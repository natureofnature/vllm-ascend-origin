"""
生成 GT:VLLM_ASCEND_KV_DEBUG=1 且 VLLM_ASCEND_CHUNKED_PREFILL=0
对比:VLLM_ASCEND_KV_DEBUG=1 且 VLLM_ASCEND_CHUNKED_PREFILL=1
覆盖已有:VLLM_ASCEND_KV_DEBUG_OVERWRITE_GT=1
输出根目录:VLLM_ASCEND_KV_DEBUG_DIR（默认 ./kv_debug）
写入文件:VLLM_ASCEND_KV_DEBUG_CTRL（默认 ./kv_debug/ctrl.json）
{
  "enabled": true,
  "chunked": false,
  "base_dir": "./kv_debug",
  "strict": false,
  "rtol": 1e-3,
  "atol": 1e-3
}

"""
import os
import json
import time
import glob
import hashlib
import pickle
from typing import Optional, Tuple

import torch
import torch_npu  # type: ignore
from vllm.logger import logger  # type: ignore


def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as exc:
        logger.warning(f"[KVDBG] mkdir failed for {path}: {exc}")


def _hash_list_int(int_list: list[int]) -> str:
    h = hashlib.sha1()
    h.update(
        (" ".join(str(x) for x in int_list)).encode("utf-8")
    )
    return h.hexdigest()[:12]


_CTRL_CACHE = {"path": None, "mtime": 0.0, "data": None}


def _ctrl_path_default() -> str:
    # Default control file under base dir
    return os.getenv("VLLM_ASCEND_KV_DEBUG_CTRL", os.path.join(os.getenv("VLLM_ASCEND_KV_DEBUG_DIR", "./kv_debug"), "ctrl.json"))


def _load_ctrl() -> dict:
    path = _ctrl_path_default()
    try:
        st = os.stat(path)
    except Exception:
        return {}
    if _CTRL_CACHE["path"] != path or _CTRL_CACHE["mtime"] < st.st_mtime:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            _CTRL_CACHE["path"] = path
            _CTRL_CACHE["mtime"] = st.st_mtime
            _CTRL_CACHE["data"] = data
        except Exception:
            return {}
    return _CTRL_CACHE.get("data") or {}


def _get_debug_base_dir() -> str:
    ctrl = _load_ctrl()
    if isinstance(ctrl, dict) and ctrl.get("base_dir"):
        return str(ctrl.get("base_dir"))
    return os.getenv("VLLM_ASCEND_KV_DEBUG_DIR", "./kv_debug")


def _is_enabled() -> bool:
    ctrl = _load_ctrl()
    if isinstance(ctrl, dict) and "enabled" in ctrl:
        return bool(ctrl.get("enabled"))
    flag = os.getenv("VLLM_ASCEND_KV_DEBUG", "0")
    return flag in ("1", "true", "True")


def _chunked_prefill_enabled() -> bool:
    # Prefer control file, then env
    ctrl = _load_ctrl()
    if isinstance(ctrl, dict):
        if "chunked" in ctrl:
            return bool(ctrl.get("chunked"))
        if "is_chunked" in ctrl:
            return bool(ctrl.get("is_chunked"))
    # Prefer explicit env from user; support multiple names for convenience
    for name in ("VLLM_ASCEND_CHUNKED_PREFILL", "VLLM_ASCEND_IS_CHUNKED", "VLLM_ASCEND_FORCE_CHUNKED"):
        val = os.getenv(name)
        if val is not None:
            return val in ("1", "true", "True")
    return False


def _ground_truth_mode() -> bool:
    # Ground truth defined as: chunked prefill disabled
    return not _chunked_prefill_enabled()


def _gt_paths(base_dir: str) -> tuple[str, str]:
    gt_dir = os.path.join(base_dir, "groundtruth")
    _ensure_dir(gt_dir)
    manifest = os.path.join(gt_dir, "manifest_gt.json")
    return gt_dir, manifest


def _cmp_dir(base_dir: str) -> str:
    cmp_dir = os.path.join(base_dir, "compare", time.strftime("%Y%m%d-%H%M%S"))
    _ensure_dir(cmp_dir)
    return cmp_dir


def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_pickle(path: str, obj) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _load_manifest_json(path: str) -> dict:
    if not os.path.exists(path):
        return {"version": 1, "entries": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"version": 1, "entries": []}


def _save_manifest_json(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def _upsert_manifest_entry(manifest: dict, entry: dict) -> dict:
    entries = manifest.get("entries", [])
    key_fields = ["layer_idx", "cp_rank", "cp_size", "sp_rank", "sp_size", "tag"]
    def same_key(a, b):
        return all(a.get(k) == b.get(k) for k in key_fields)
    for i, e in enumerate(entries):
        if same_key(e, entry):
            entries[i] = entry
            break
    else:
        entries.append(entry)
    manifest["entries"] = entries
    return manifest


def _get_tol() -> tuple[float, float, bool]:
    ctrl = _load_ctrl()
    if isinstance(ctrl, dict):
        strict = bool(ctrl.get("strict", False))
        rtol = float(ctrl.get("rtol", 1e-3))
        atol = float(ctrl.get("atol", 1e-3))
        return rtol, atol, strict
    strict = os.getenv("VLLM_ASCEND_KV_DEBUG_STRICT", "0") in ("1", "true", "True")
    try:
        rtol = float(os.getenv("VLLM_ASCEND_KV_DEBUG_RTOL", "1e-3"))
    except Exception:
        rtol = 1e-3
    try:
        atol = float(os.getenv("VLLM_ASCEND_KV_DEBUG_ATOL", "1e-3"))
    except Exception:
        atol = 1e-3
    return rtol, atol, strict


def _first_mismatch_index(
    a: torch.Tensor,
    b: torch.Tensor,
    rtol: float,
    atol: float,
    strict: bool,
):
    # a, b: [tokens, ...]
    if a.shape != b.shape or a.numel() == 0:
        return 0
    a2 = a.view(a.shape[0], -1).to(torch.float32)
    b2 = b.view(b.shape[0], -1).to(torch.float32)
    if strict:
        diff_mask = (a2 != b2).any(dim=1)
    else:
        isclose_elem = torch.isclose(a2, b2, rtol=rtol, atol=atol)
        diff_mask = (~isclose_elem).any(dim=1)
    idx = torch.nonzero(diff_mask, as_tuple=False)
    if idx.numel() == 0:
        return None
    return int(idx[0].item())


def dump_or_compare_kv(
    attn_metadata,
    kv_cache: Tuple[torch.Tensor, torch.Tensor],
    cp_rank: int,
    cp_size: int,
    sp_rank: int,
    sp_size: int,
    device: torch.device,
    tag: str,
    layer_idx: int = -1,
    tp_rank: int | None = None,
    tp_size: int | None = None,
) -> None:
    if not _is_enabled():
        return

    try:
        num_reqs = attn_metadata.num_decodes + attn_metadata.num_prefills
        if num_reqs <= 0:
            return
    except Exception:
        return

    # Per-rank seq_len per request
    try:
        if hasattr(attn_metadata, "prefill") and attn_metadata.prefill is not None and \
           getattr(attn_metadata.prefill, "num_computed_tokens_of_cp_sp", None) is not None and cp_size > 1:
            # shape [bs, cp, sp]
            cp_sp = attn_metadata.prefill.num_computed_tokens_of_cp_sp
            seq_len_list = [int(cp_sp[i][cp_rank][sp_rank]) for i in range(num_reqs)]
        elif hasattr(attn_metadata, "decode") and attn_metadata.decode is not None and \
             getattr(attn_metadata.decode, "num_computed_tokens_of_cp_sp", None) is not None and cp_size > 1:
            cp_sp = attn_metadata.decode.num_computed_tokens_of_cp_sp
            seq_len_list = [int(cp_sp[i][cp_rank][sp_rank]) for i in range(num_reqs)]
        else:
            # No CP split: use cumulative seq_lens per request
            seq_len_list = [int(x) for x in attn_metadata.seq_lens[:num_reqs].tolist()]
    except Exception as exc:
        logger.warning(f"[KVDBG] seq_len resolve failed: {exc}")
        return

    total_tokens = sum(seq_len_list)

    block_table = attn_metadata.block_tables  # [bs, max_blocks_per_req]
    cache_kv_c, cache_k_pe = kv_cache[0], kv_cache[1]

    # Load KV for current rank by requested per-req lengths
    seq_len_tensor = torch.tensor(seq_len_list, dtype=torch.int32, device=device)
    zero_starts = torch.zeros([len(seq_len_list)], dtype=torch.int32, device=device)
    try:
        kv_c_normed, k_pe = torch_npu.atb.npu_paged_cache_load(
            cache_kv_c,
            cache_k_pe,
            block_table,
            seq_len_tensor,
            seq_starts=zero_starts,
        )
    except Exception as exc:
        logger.warning(f"[KVDBG] npu_paged_cache_load failed: {exc}")
        return

    kv_c_normed = kv_c_normed.squeeze()
    k_pe = k_pe.squeeze()

    base_dir = _get_debug_base_dir()
    _ensure_dir(base_dir)
    gt_dir, manifest_gt = _gt_paths(base_dir)

    # Effective TP: if not provided, fallback to SP (some codebases alias SP to TP)
    eff_tp_rank = sp_rank if tp_rank is None else int(tp_rank)
    eff_tp_size = sp_size if tp_size is None else int(tp_size)

    meta = {
        "mode": "gt" if _ground_truth_mode() else "chk",
        "num_reqs": len(seq_len_list),
        "seq_len": seq_len_list,
        "cp_rank": cp_rank,
        "cp_size": cp_size,
        "sp_rank": sp_rank,
        "sp_size": sp_size,
        "tp_rank": eff_tp_rank,
        "tp_size": eff_tp_size,
        "tag": tag,
        "layer_idx": int(layer_idx),
    }

    tensor_dump = {
        "kv_c_normed": kv_c_normed.detach().cpu().to(torch.float16),
        "k_pe": k_pe.detach().cpu().to(torch.float16),
    }

    if _ground_truth_mode():
        # Write/merge manifest entry for this (layer, cp, sp, tag)
        overwrite = os.getenv("VLLM_ASCEND_KV_DEBUG_OVERWRITE_GT", "0") in ("1", "true", "True")
        manifest_data = {} if overwrite else _load_manifest_json(manifest_gt)
        entry = {
            "layer_idx": int(layer_idx),
            "cp_rank": int(cp_rank),
            "cp_size": int(cp_size),
            "sp_rank": int(sp_rank),
            "sp_size": int(sp_size),
            "tp_rank": eff_tp_rank,
            "tp_size": eff_tp_size,
            "tag": str(tag),
            "num_reqs": int(meta["num_reqs"]),
            "seq_len": seq_len_list,
            "kv_path": f"L{layer_idx}/tp{eff_tp_rank}/cp{cp_rank}-sp{sp_rank}/kv.pkl",
        }
        manifest_data = _upsert_manifest_entry(manifest_data, entry)
        _save_manifest_json(manifest_gt, manifest_data)
        # Per-layer, per-(cp,sp) subdir
        out_dir = os.path.join(gt_dir, f"L{layer_idx}", f"tp{eff_tp_rank}", f"cp{cp_rank}-sp{sp_rank}")
        _ensure_dir(out_dir)
        out_path = os.path.join(out_dir, "kv.pkl")
        # If this rank has 0 tokens this step, still write an empty tensor to mark presence
        if total_tokens <= 0:
            _save_pickle(out_path, {
                "kv_c_normed": torch.empty((0,), dtype=torch.float16),
                "k_pe": torch.empty((0,), dtype=torch.float16),
            })
        else:
            _save_pickle(out_path, tensor_dump)
        logger.info(f"[KVDBG] saved GT kv to {out_path}")
        return

    # Compare against latest GT with same key
    gt_dir, manifest_path = _gt_paths(base_dir)
    if not os.path.exists(manifest_path):
        logger.warning("[KVDBG] no GT manifest found; run with chunked prefill disabled first")
        return
    try:
        manifest_data = _load_manifest_json(manifest_path)
        # Select matching entry
        entries = manifest_data.get("entries", [])
        candidates = [
            e for e in entries
            if (e.get("layer_idx") == int(layer_idx) and
                e.get("cp_rank") == int(cp_rank) and
                e.get("cp_size") == int(cp_size) and
                e.get("sp_rank") == int(sp_rank) and
                e.get("sp_size") == int(sp_size) and
                e.get("tp_rank", eff_tp_rank) == eff_tp_rank and
                e.get("tp_size", eff_tp_size) == eff_tp_size)
        ]
        if not candidates:
            # Print brief manifest summary to help align GT/compare settings
            avail = []
            for e in entries:
                if e.get("layer_idx") == int(layer_idx) and e.get("tag") == str(tag):
                    avail.append(f"cp{e.get('cp_rank')}/{e.get('cp_size')} sp{e.get('sp_rank')}/{e.get('sp_size')} tp{e.get('tp_rank', 'NA')}/{e.get('tp_size', 'NA')}")
            logger.warning(
                f"[KVDBG] GT entry not found for L{layer_idx} cp{cp_rank}/{cp_size} sp{sp_rank}/{sp_size} tp{eff_tp_rank}/{eff_tp_size} tag={tag}. "
                f"Available for this layer/tag: {', '.join(avail) if avail else 'none'}. "
                f"Ensure GT run used the same cp/sp/tp sizes and all ranks had VLLM_ASCEND_KV_DEBUG=1.")
            return
        # Prefer tag match, otherwise take the one with largest total seq tokens
        tagged = [e for e in candidates if e.get("tag") == str(tag)]
        chosen = tagged if tagged else candidates
        match = max(chosen, key=lambda e: sum(e.get("seq_len", [])))
        gt_seq = match.get("seq_len", [])
        kv_rel_path = match.get("kv_path")
        gt_path = os.path.join(gt_dir, kv_rel_path) if kv_rel_path else os.path.join(gt_dir, f"L{layer_idx}", f"tp{eff_tp_rank}", f"cp{cp_rank}-sp{sp_rank}", "kv.pkl")
        if not os.path.exists(gt_path):
            logger.warning(f"[KVDBG] GT file missing at {gt_path}")
            return
        gt = _load_pickle(gt_path)
    except Exception as exc:
        logger.warning(f"[KVDBG] load GT failed: {exc}")
        return

    # Build expected prefix by slicing GT per-request
    try:
        if len(gt_seq) != len(seq_len_list):
            logger.warning(f"[KVDBG] GT seq_len size mismatch: gt={len(gt_seq)} cur={len(seq_len_list)}")
            return

        def _slice_prefix(full: torch.Tensor, gt_lens: list[int], cur_lens: list[int]) -> torch.Tensor:
            # full: [sum(gt_lens), ...]; return [sum(cur_lens), ...]
            dim_tail = full.shape[1:]
            out = []
            off = 0
            for gl, cl in zip(gt_lens, cur_lens):
                take = min(gl, cl)
                if take > 0:
                    out.append(full[off:off + take])
                off += gl
            if not out:
                return full[:0]
            return torch.cat(out, dim=0)

        kv_gt_full = gt["kv_c_normed"].to(torch.float16)
        kpe_gt_full = gt["k_pe"].to(torch.float16)
        kv_gt_expect = _slice_prefix(kv_gt_full, gt_seq, seq_len_list)
        kpe_gt_expect = _slice_prefix(kpe_gt_full, gt_seq, seq_len_list)
    except Exception as exc:
        logger.warning(f"[KVDBG] build expected prefix failed: {exc}")
        return

    rtol, atol, strict = _get_tol()

    def _tensor_close(a: torch.Tensor, b: torch.Tensor) -> bool:
        if a.shape != b.shape:
            return False
        try:
            if strict:
                return torch.equal(a, b)
            return torch.allclose(a, b, rtol=rtol, atol=atol)
        except Exception:
            return False

    kv_ok = _tensor_close(tensor_dump["kv_c_normed"], kv_gt_expect)
    pe_ok = _tensor_close(tensor_dump["k_pe"], kpe_gt_expect)

    if kv_ok and pe_ok:
        logger.info(f"[KVDBG] OK rank({cp_rank},{sp_rank}) L{layer_idx} tokens={total_tokens}")
    else:
        # Save diff snapshot for inspection
        run_dir = _cmp_dir(base_dir)
        out_dir = os.path.join(run_dir, f"L{layer_idx}", f"cp{cp_rank}-sp{sp_rank}")
        _ensure_dir(out_dir)
        out_path = os.path.join(out_dir, "kv_cmp.pkl")
        # Find first mismatch token indices
        kv_first_bad = _first_mismatch_index(tensor_dump["kv_c_normed"], kv_gt_expect, rtol, atol, strict)
        pe_first_bad = _first_mismatch_index(tensor_dump["k_pe"], kpe_gt_expect, rtol, atol, strict)
        # Map token index to (req_id, offset)
        def _tok_to_req_off(token_idx: int, lens: list[int]):
            if token_idx is None:
                return None
            cum = 0
            for rid, L in enumerate(lens):
                if token_idx < cum + L:
                    return {"req": rid, "offset": token_idx - cum}
                cum += L
            return {"req": -1, "offset": -1}
        kv_loc = _tok_to_req_off(kv_first_bad, seq_len_list)
        pe_loc = _tok_to_req_off(pe_first_bad, seq_len_list)
        _save_pickle(out_path, {
            "cur": tensor_dump,
            "expect": {"kv_c_normed": kv_gt_expect, "k_pe": kpe_gt_expect},
            "gt_full": gt,
            "meta": meta,
            "meta_gt": meta_gt,
            "kv_first_bad_token": kv_first_bad,
            "pe_first_bad_token": pe_first_bad,
            "kv_first_bad_loc": kv_loc,
            "pe_first_bad_loc": pe_loc,
        })
        detail = []
        if kv_first_bad is not None:
            detail.append(f"kv@tok{kv_first_bad}:{kv_loc}")
        if pe_first_bad is not None:
            detail.append(f"pe@tok{pe_first_bad}:{pe_loc}")
        logger.warning(
            f"[KVDBG] MISMATCH rank({cp_rank},{sp_rank}) L{layer_idx} tokens={total_tokens} -> {out_path} | "
            + ", ".join(detail))


