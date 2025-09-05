"""
生成 GT:VLLM_ASCEND_KV_DEBUG=1 且 VLLM_ASCEND_CHUNKED_PREFILL=0
对比:VLLM_ASCEND_KV_DEBUG=1 且 VLLM_ASCEND_CHUNKED_PREFILL=1
覆盖已有:VLLM_ASCEND_KV_DEBUG_OVERWRITE_GT=1
输出根目录:VLLM_ASCEND_KV_DEBUG_DIR（默认 ./kv_debug）
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


def _get_debug_base_dir() -> str:
    return os.getenv("VLLM_ASCEND_KV_DEBUG_DIR", "./kv_debug")


def _is_enabled() -> bool:
    flag = os.getenv("VLLM_ASCEND_KV_DEBUG", "0")
    return flag in ("1", "true", "True")


def _chunked_prefill_enabled() -> bool:
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
    if total_tokens <= 0:
        return

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

    meta = {
        "mode": "gt" if _ground_truth_mode() else "chk",
        "key": _hash_list_int(seq_len_list),
        "num_reqs": len(seq_len_list),
        "seq_len": seq_len_list,
        "cp_rank": cp_rank,
        "cp_size": cp_size,
        "sp_rank": sp_rank,
        "sp_size": sp_size,
        "tag": tag,
        "layer_idx": int(layer_idx),
    }

    tensor_dump = {
        "kv_c_normed": kv_c_normed.detach().cpu().to(torch.float16),
        "k_pe": k_pe.detach().cpu().to(torch.float16),
    }

    if _ground_truth_mode():
        # Write single shared GT set (overwritable by env)
        overwrite = os.getenv("VLLM_ASCEND_KV_DEBUG_OVERWRITE_GT", "0") in ("1", "true", "True")
        mode = "w" if overwrite or (not os.path.exists(manifest_gt)) else None
        if mode is None:
            logger.info("[KVDBG] GT exists; skip (set VLLM_ASCEND_KV_DEBUG_OVERWRITE_GT=1 to overwrite)")
            return
        with open(manifest_gt, mode) as f:
            json.dump(meta, f)
        out_path = os.path.join(gt_dir, f"kv_L{layer_idx}_rank{cp_rank}-{sp_rank}.pkl")
        _save_pickle(out_path, tensor_dump)
        logger.info(f"[KVDBG] saved GT kv to {out_path}")
        return

    # Compare against latest GT with same key
    gt_dir, manifest_path = _gt_paths(base_dir)
    if not os.path.exists(manifest_path):
        logger.warning("[KVDBG] no GT manifest found; run with chunked prefill disabled first")
        return
    try:
        with open(manifest_path, "r") as f:
            meta_gt = json.load(f)
        gt_path = os.path.join(gt_dir, f"kv_L{layer_idx}_rank{cp_rank}-{sp_rank}.pkl")
        if not os.path.exists(gt_path):
            # fallback legacy name
            gt_path = os.path.join(gt_dir, f"kv_rank{cp_rank}-{sp_rank}.pkl")
        if not os.path.exists(gt_path):
            logger.warning(f"[KVDBG] GT file not found for rank {cp_rank}-{sp_rank}")
            return
        gt = _load_pickle(gt_path)
    except Exception as exc:
        logger.warning(f"[KVDBG] load GT failed: {exc}")
        return

    # Build expected prefix by slicing GT per-request
    try:
        gt_seq = meta_gt.get("seq_len", [])
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

    def _tensor_close(a: torch.Tensor, b: torch.Tensor) -> bool:
        if a.shape != b.shape:
            return False
        try:
            return torch.allclose(a, b, rtol=1e-3, atol=1e-3)
        except Exception:
            return False

    kv_ok = _tensor_close(tensor_dump["kv_c_normed"], kv_gt_expect)
    pe_ok = _tensor_close(tensor_dump["k_pe"], kpe_gt_expect)

    if kv_ok and pe_ok:
        logger.info(f"[KVDBG] OK rank({cp_rank},{sp_rank}) L{layer_idx} tokens={total_tokens}")
    else:
        # Save diff snapshot for inspection
        run_dir = _cmp_dir(base_dir)
        out_path = os.path.join(run_dir, f"kv_cmp_L{layer_idx}_rank{cp_rank}-{sp_rank}.pkl")
        _save_pickle(out_path, {
            "cur": tensor_dump,
            "expect": {"kv_c_normed": kv_gt_expect, "k_pe": kpe_gt_expect},
            "gt_full": gt,
            "meta": meta,
            "meta_gt": meta_gt,
        })
        logger.warning(
            f"[KVDBG] MISMATCH rank({cp_rank},{sp_rank}) L{layer_idx} tokens={total_tokens} -> {out_path}")


