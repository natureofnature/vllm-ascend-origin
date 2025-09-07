import argparse
import glob
import os
import pickle
from typing import Dict, Tuple, List

import torch


def _scan_attn(dir_path: str) -> Dict[Tuple[int, int, int, int], str]:
    """
    Map (layer, step, cp, sp) -> attn file path
    Expect names like: layer{L}_step{S}_cp{C}_sp{P}_attn_{tag}.pkl
    """
    out: Dict[Tuple[int, int, int, int], str] = {}
    for fp in glob.glob(os.path.join(dir_path, "*.pkl")):
        bn = os.path.basename(fp)
        if "_attn_" not in bn:
            continue
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
            continue
    return out


def _merge_per_step_attn(m: Dict[Tuple[int, int, int, int], str]) -> Dict[Tuple[int, int], torch.Tensor]:
    buckets: Dict[Tuple[int, int], List[Tuple[int, int, str]]] = {}
    for (layer, step, cp, sp), fp in m.items():
        buckets.setdefault((layer, step), []).append((cp, sp, fp))
    merged: Dict[Tuple[int, int], torch.Tensor] = {}
    for (layer, step), shard_list in buckets.items():
        shard_list.sort()
        attn_list = []
        for _, _, fp in shard_list:
            with open(fp, "rb") as f:
                d = pickle.load(f)
            if isinstance(d, dict) and "attn" in d:
                attn_list.append(d["attn"])  # [tokens, H*V]
        if attn_list:
            merged[(layer, step)] = torch.cat(attn_list, dim=0)
    return merged


def _series_by_layer(merged: Dict[Tuple[int, int], torch.Tensor]):
    by_layer: Dict[int, List[Tuple[int, torch.Tensor]]] = {}
    for (layer, step), t in merged.items():
        by_layer.setdefault(layer, []).append((step, t))
    for layer in by_layer:
        by_layer[layer].sort(key=lambda x: x[0])
    return by_layer


def _concat_upto(series: List[Tuple[int, torch.Tensor]], target_len: int) -> torch.Tensor:
    out_list = []
    acc = 0
    for _, t in series:
        need = target_len - acc
        if need <= 0:
            break
        take = min(need, t.size(0))
        out_list.append(t[:take])
        acc += take
    if not out_list:
        return torch.empty(0)
    return torch.cat(out_list, dim=0)


def main():
    ap = argparse.ArgumentParser(description="Compare attention outputs with/without chunked prefill")
    ap.add_argument("dir_off", type=str, help="dump dir for chunked prefill OFF")
    ap.add_argument("dir_on", type=str, help="dump dir for chunked prefill ON")
    ap.add_argument("--expected_tokens", type=int, default=2974)
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--rtol", type=float, default=1e-3)
    args = ap.parse_args()

    off_map = _scan_attn(args.dir_off)
    on_map = _scan_attn(args.dir_on)

    off_step = _merge_per_step_attn(off_map)
    on_step = _merge_per_step_attn(on_map)

    off_series = _series_by_layer(off_step)
    on_series = _series_by_layer(on_step)

    layers = sorted(set(off_series.keys()) & set(on_series.keys()))
    if not layers:
        print("No overlapping layers found.")
        return 2

    any_mismatch = False
    for layer in layers:
        a = _concat_upto(off_series[layer], args.expected_tokens)
        b = _concat_upto(on_series[layer], args.expected_tokens)
        trim = min(a.size(0), b.size(0), args.expected_tokens)
        if trim == 0:
            print(f"Layer {layer}: no data to compare.")
            continue
        a = a[:trim]
        b = b[:trim]
        if a.shape != b.shape:
            print(f"Layer {layer}: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
            any_mismatch = True
            continue
        diff = (a - b).abs()
        mae = diff.mean().item()
        maxe = diff.max().item()
        denom = b.abs().mean().clamp_min(1e-6)
        rel = (diff / denom).mean().item()
        ok = torch.allclose(a, b, atol=args.atol, rtol=args.rtol)
        status = "OK" if ok else "DIFF"
        print(f"Layer {layer}: status={status}, tokens={trim}, mae={mae:.4e}, max={maxe:.4e}, rel_mean={rel:.4e}")
        if not ok:
            any_mismatch = True

    return 1 if any_mismatch else 0


if __name__ == "__main__":
    raise SystemExit(main())


