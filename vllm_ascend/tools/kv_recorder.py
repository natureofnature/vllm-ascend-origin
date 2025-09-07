import os
import pickle
from typing import Dict, List, Tuple

import torch
import torch_npu  # type: ignore


class KVRecorder:
    """
    Minimal KV cache recorder for MLA on Ascend.
    - Pulls per-layer KV for current step via npu_paged_cache_load given block_table and seq_lens
    - Appends to an in-memory list; can be flushed to pickle per step
    """

    def __init__(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.steps: List[Dict[str, torch.Tensor]] = []

    def record_layer(
        self,
        layer_idx: int,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        cache_kv_c, cache_k_pe = kv_cache
        # seq_lens: [bs]
        seq_starts = torch.zeros_like(seq_lens, dtype=torch.int32, device=device)
        kv_c_normed, k_pe = torch_npu.atb.npu_paged_cache_load(
            cache_kv_c,
            cache_k_pe,
            block_table,
            seq_lens.to(device).to(torch.int32),
            seq_starts,
        )
        # [tot_seq, 1, D] -> [tot_seq, D]
        kv_c_normed = kv_c_normed.squeeze(1)
        k_pe = k_pe.squeeze(1)
        return {
            "kv_c": kv_c_normed.detach().cpu(),
            "k_pe": k_pe.detach().cpu(),
        }

    def push_step(self, payload: Dict[str, Dict[str, torch.Tensor]]):
        # payload: {layer_name: {"kv_c": Tensor, "k_pe": Tensor}}
        self.steps.append({k: {kk: vv.cpu() for kk, vv in v.items()} for k, v in payload.items()})

    def flush(self, tag: str) -> str:
        path = os.path.join(self.out_dir, f"kv_steps_{tag}.pkl")
        with open(path, "wb") as f:
            pickle.dump(self.steps, f)
        return path


def compare_pickles(p1: str, p2: str, atol: float = 1e-3, rtol: float = 1e-3):
    with open(p1, "rb") as f:
        a = pickle.load(f)
    with open(p2, "rb") as f:
        b = pickle.load(f)

    if len(a) != len(b):
        return {
            "ok": False,
            "reason": f"num_steps mismatch: {len(a)} vs {len(b)}",
        }
    # step-wise
    for s, (sa, sb) in enumerate(zip(a, b)):
        if sa.keys() != sb.keys():
            return {"ok": False, "reason": f"layers mismatch at step {s}"}
        for layer in sa.keys():
            for key in ("kv_c", "k_pe"):
                ta = sa[layer][key]
                tb = sb[layer][key]
                if ta.shape != tb.shape:
                    return {
                        "ok": False,
                        "reason": f"shape mismatch at step {s}, layer {layer}, key {key}: {ta.shape} vs {tb.shape}",
                    }
                if not torch.allclose(ta, tb, atol=atol, rtol=rtol):
                    # find first mismatch token index
                    diff = (ta - tb).abs()
                    idx = (diff.view(-1) > (atol + rtol * tb.abs().view(-1))).nonzero(as_tuple=False)
                    first = int(idx[0]) if idx.numel() > 0 else -1
                    return {
                        "ok": False,
                        "reason": f"value mismatch at step {s}, layer {layer}, key {key}, first_flat_index {first}",
                    }
    return {"ok": True}


