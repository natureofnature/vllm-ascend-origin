import argparse
import glob
import os
import pickle
from typing import Dict, Tuple

import torch

'''
流程：
关闭 chunked prefill，创建 flag：
echo /tmp/kv_off > /tmp/vllm_ascend_kv_dump_dir
echo off > /tmp/vllm_ascend_kv_dump_tag
跑一次同样输入，生成 ground truth 于 /tmp/kv_off
开启 chunked prefill，创建 flag：
echo /tmp/kv_on > /tmp/vllm_ascend_kv_dump_dir
echo on > /tmp/vllm_ascend_kv_dump_tag
跑同样输入，生成对比对象于 /tmp/kv_on
对比：
python tools/kv_compare_dir.py /tmp/kv_off /tmp/kv_on --atol 1e-3 --rtol 1e-3
'''

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
            # layer{L}
            layer = int(parts[0].replace("layer", ""))
            # step{S}
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
            # fallback older naming: layer{L}_step{S}_{tag}.pkl
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir_a", type=str, help="ground truth dir (chunked prefill OFF)")
    ap.add_argument("dir_b", type=str, help="compare dir (chunked prefill ON)")
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--rtol", type=float, default=1e-3)
    args = ap.parse_args()

    map_a = _scan(args.dir_a)
    map_b = _scan(args.dir_b)

    # Merge shards (cp,sp) per (layer, step) by concatenating on sequence dim
    def load_merge(m: Dict[Tuple[int, int, int, int], str]):
        merged: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
        buckets: Dict[Tuple[int, int], list[Tuple[int, int, str]]] = {}
        for (layer, step, cp, sp), fp in m.items():
            buckets.setdefault((layer, step), []).append((cp, sp, fp))
        for (layer, step), shard_list in buckets.items():
            shards = []
            for _, _, fp in sorted(shard_list):
                with open(fp, "rb") as f:
                    shards.append(pickle.load(f))
            kv_c_list = [s["kv_c"] for s in shards]
            k_pe_list = [s["k_pe"] for s in shards]
            kv_c = torch.cat(kv_c_list, dim=0) if len(kv_c_list) > 1 else kv_c_list[0]
            k_pe = torch.cat(k_pe_list, dim=0) if len(k_pe_list) > 1 else k_pe_list[0]
            merged[(layer, step)] = {"kv_c": kv_c, "k_pe": k_pe}
        return merged

    A = load_merge(map_a)
    B = load_merge(map_b)

    keys = sorted(set(A.keys()) & set(B.keys()))
    if not keys:
        print("No overlapping (layer, step) after merge. Check inputs.")
        return 2

    for (layer, step) in keys:
        pa = A[(layer, step)]
        pb = B[(layer, step)]
        for key in ("kv_c", "k_pe"):
            if key not in pa or key not in pb:
                print(f"[L{layer} S{step}] missing key {key}")
                return 2
            ok, msg = _first_mismatch(pa[key], pb[key], args.atol, args.rtol)
            if not ok:
                a = pa[key]
                flat_len_per_token = a.shape[1] if a.dim() == 2 else 1
                diff = (pa[key] - pb[key]).abs()
                mask = diff > (args.atol + args.rtol * pb[key].abs())
                idx = mask.view(-1).nonzero(as_tuple=False)
                first = int(idx[0]) if idx.numel() > 0 else -1
                token_idx = first // flat_len_per_token if flat_len_per_token > 0 and first >= 0 else -1
                print(f"First mismatch at layer={layer}, step={step}, key={key}, token_idx={token_idx}. {msg}")
                return 1

    print("All compared KV are consistent across layers/steps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


