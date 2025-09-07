import argparse
import glob
import os
import pickle
from typing import Dict, Tuple, List

import torch


def _scan(dir_path: str) -> Dict[Tuple[int, int, int, int], str]:
    """
    Return mapping: (layer_idx, step_idx, cp_rank, sp_rank) -> file_path
    Expect files named like: layer{L}_step{S}_cp{C}_sp{P}_{tag}.pkl
    Backward compatible with layer{L}_step{S}_{tag}.pkl (cp,sp default 0).
    """
    out: Dict[Tuple[int, int, int, int], str] = {}
    for fp in glob.glob(os.path.join(dir_path, "*.pkl")):
        bn = os.path.basename(fp)
        try:
            parts = bn.split("_")
            layer = int(parts[0].replace("layer", ""))
            step = int(parts[1].replace("step", ""))
            cp = 0
            sp = 0
            for part in parts[2:]:
                if part.startswith("cp"):
                    cp = int(part.replace("cp", ""))
                elif part.startswith("sp"):
                    sp = int(part.replace("sp", ""))
            out[(layer, step, cp, sp)] = fp
        except Exception:
            try:
                ls, rest = bn.split("_step", 1)
                layer = int(ls.replace("layer", ""))
                s_part = rest.split("_", 1)[0]
                step = int(s_part)
                out[(layer, step, 0, 0)] = fp
            except Exception:
                continue
    return out


def _first_mismatch(a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float):
    if a.shape != b.shape:
        return (False, f"shape mismatch: {a.shape} vs {b.shape}")
    if torch.allclose(a, b, atol=atol, rtol=rtol):
        return (True, None)
    diff = (a - b).abs()
    mask = diff > (atol + rtol * b.abs())
    idx = mask.view(-1).nonzero(as_tuple=False)
    first = int(idx[0]) if idx.numel() > 0 else -1
    return (False, f"value mismatch, first_flat_index={first}")


def _pick_rep_per_step(m: Dict[Tuple[int, int, int, int], str]) -> Dict[Tuple[int, int], Dict[str, torch.Tensor]]:
    merged: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
    buckets: Dict[Tuple[int, int], List[Tuple[int, int, str]]] = {}
    for (layer, step, cp, sp), fp in m.items():
        buckets.setdefault((layer, step), []).append((cp, sp, fp))
    for (layer, step), shard_list in buckets.items():
        shard_list.sort()
        kv_fp = None
        attn_fps: List[str] = []
        # prefer cp0,sp0 for KV; collect all shards for ATTN
        for cp, sp, fp in shard_list:
            if "_attn_" in fp:
                attn_fps.append(fp)
            else:
                if kv_fp is None and cp == 0 and sp == 0:
                    kv_fp = fp
        if kv_fp is None:
            for _, _, fp in shard_list:
                if "_attn_" not in fp:
                    kv_fp = fp
                    break
        payload: Dict[str, torch.Tensor] = {}
        if kv_fp is not None:
            try:
                with open(kv_fp, "rb") as f:
                    d = pickle.load(f)
                if isinstance(d, dict):
                    if "kv_c" in d:
                        payload["kv_c"] = d["kv_c"]
                    if "k_pe" in d:
                        payload["k_pe"] = d["k_pe"]
            except Exception:
                pass
        if attn_fps:
            try:
                attn_list = []
                for fp in attn_fps:
                    with open(fp, "rb") as f:
                        d = pickle.load(f)
                    if isinstance(d, dict) and "attn" in d:
                        attn_list.append(d["attn"])
                if attn_list:
                    payload["attn"] = torch.cat(attn_list, dim=0)
            except Exception:
                pass
        merged[(layer, step)] = payload
    return merged


def _series_by_layer(rep: Dict[Tuple[int, int], Dict[str, torch.Tensor]]):
    by_layer: Dict[int, List[Tuple[int, Dict[str, torch.Tensor]]]] = {}
    for (layer, step), payload in rep.items():
        by_layer.setdefault(layer, []).append((step, payload))
    for layer in by_layer:
        by_layer[layer].sort(key=lambda x: x[0])
    return by_layer


def _concat_upto(series: List[Tuple[int, Dict[str, torch.Tensor]]], target_len: int):
    kv_list = []
    kpe_list = []
    acc = 0
    for _, p in series:
        kv = p["kv_c"]
        kpe = p["k_pe"]
        need = target_len - acc
        if need <= 0:
            break
        take = min(need, kv.size(0))
        kv_list.append(kv[:take])
        kpe_list.append(kpe[:take])
        acc += take
    kv_cat = torch.cat(kv_list, dim=0) if len(kv_list) > 1 else (kv_list[0] if kv_list else torch.empty(0))
    kpe_cat = torch.cat(kpe_list, dim=0) if len(kpe_list) > 1 else (kpe_list[0] if kpe_list else torch.empty(0))
    return kv_cat, kpe_cat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir_a", type=str, help="ground truth dir (chunked prefill OFF)")
    ap.add_argument("dir_b", type=str, help="compare dir (chunked prefill ON)")
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--expected_tokens", type=int, default=None, help="Optional total tokens to compare (truncate)")
    ap.add_argument("--compare_attn", default=True)
    args = ap.parse_args()

    map_a = _scan(args.dir_a)
    map_b = _scan(args.dir_b)

    rep_a = _pick_rep_per_step(map_a)
    rep_b = _pick_rep_per_step(map_b)

    A = _series_by_layer(rep_a)
    B = _series_by_layer(rep_b)

    layers = sorted(set(A.keys()) & set(B.keys()))
    pair_mode = False
    if not layers:
        if not A or not B:
            print("No overlapping layers. Check inputs.")
            return 2
        la = min(A.keys())
        lb = min(B.keys())
        layers = [(la, lb)]
        pair_mode = True

    if pair_mode:
        la, lb = layers[0]
        sa = A[la]
        sb = B[lb]
        total_a = sum([p["kv_c"].size(0) for _, p in sa if "kv_c" in p])
        total_b = sum([p["kv_c"].size(0) for _, p in sb if "kv_c" in p])
        if total_a == 0 or total_b == 0:
            # fallback to attention length if kv missing
            attn_total_a = sum([p["attn"].size(0) for _, p in sa if "attn" in p])
            attn_total_b = sum([p["attn"].size(0) for _, p in sb if "attn" in p])
            total_a = attn_total_a
            total_b = attn_total_b
        target = args.expected_tokens if args.expected_tokens is not None else min(total_a, total_b)
        if any(("kv_c" in p) for _, p in sa) and any(("kv_c" in p) for _, p in sb):
            a_kv, a_kpe = _concat_upto(sa, target)
            b_kv, b_kpe = _concat_upto(sb, target)
            for key, ta, tb in (("kv_c", a_kv, b_kv), ("k_pe", a_kpe, b_kpe)):
                if ta.numel() and tb.numel():
                    ok, msg = _first_mismatch(ta, tb, args.atol, args.rtol)
                    if not ok:
                        print(f"First mismatch (pair) key={key}, target_tokens={target}. {msg}")
                        return 1
        if args.compare_attn:
            # accumulate attn outputs similarly if available
            a_attn_list = [p.get("attn") for _, p in sa if p.get("attn") is not None]
            b_attn_list = [p.get("attn") for _, p in sb if p.get("attn") is not None]
            if a_attn_list and b_attn_list:
                a_attn = torch.cat(a_attn_list, dim=0)
                b_attn = torch.cat(b_attn_list, dim=0)
                trim = min(a_attn.size(0), b_attn.size(0), target)
                ok, msg = _first_mismatch(a_attn[:trim], b_attn[:trim], args.atol, args.rtol)
                if not ok:
                    print(f"First mismatch (pair) key=attn, target_tokens={trim}. {msg}")
                    return 1
        print("All compared KV are consistent (pair mode).")
        return 0

    for layer in layers:
        sa = A[layer]
        sb = B[layer]
        total_a = sum([p["kv_c"].size(0) for _, p in sa if "kv_c" in p])
        total_b = sum([p["kv_c"].size(0) for _, p in sb if "kv_c" in p])
        if total_a == 0 or total_b == 0:
            attn_total_a = sum([p["attn"].size(0) for _, p in sa if "attn" in p])
            attn_total_b = sum([p["attn"].size(0) for _, p in sb if "attn" in p])
            total_a = attn_total_a
            total_b = attn_total_b
        target = args.expected_tokens if args.expected_tokens is not None else min(total_a, total_b)
        if any(("kv_c" in p) for _, p in sa) and any(("kv_c" in p) for _, p in sb):
            a_kv, a_kpe = _concat_upto(sa, target)
            b_kv, b_kpe = _concat_upto(sb, target)
            for key, ta, tb in (("kv_c", a_kv, b_kv), ("k_pe", a_kpe, b_kpe)):
                if ta.numel() and tb.numel():
                    ok, msg = _first_mismatch(ta, tb, args.atol, args.rtol)
                    if not ok:
                        print(f"First mismatch at layer={layer}, key={key}, target_tokens={target}. {msg}")
                        return 1
        if args.compare_attn:
            a_attn_list = [p.get("attn") for _, p in sa if p.get("attn") is not None]
            b_attn_list = [p.get("attn") for _, p in sb if p.get("attn") is not None]
            if a_attn_list and b_attn_list:
                a_attn = torch.cat(a_attn_list, dim=0)
                b_attn = torch.cat(b_attn_list, dim=0)
                trim = min(a_attn.size(0), b_attn.size(0), target)
                ok, msg = _first_mismatch(a_attn[:trim], b_attn[:trim], args.atol, args.rtol)
                if not ok:
                    print(f"First mismatch at layer={layer}, key=attn, target_tokens={trim}. {msg}")
                    return 1

    print("All compared KV are consistent across layers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


