"""
Standalone simulator for CP + chunked prefill scheduling and mapping.

This file does not import vLLM. It reproduces the essential logic used in
Ascend scheduler and runner to help diagnose mismatches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclass
class CPChunkStep:
    num_new_tokens: int
    cp_size: int
    sp_size: int
    block_size: int

    def scheduler_distribution_tokens(self) -> List[List[int]]:
        """
        Simulate AscendScheduler splitting a step's chunk by blocks, then
        flattening to tokens per (cp, sp) rank.
        """
        total_blocks = ceil_div(self.num_new_tokens, self.block_size)
        total_ranks = self.cp_size * self.sp_size
        base = total_blocks // total_ranks
        rem = total_blocks % total_ranks
        blocks_flat: List[int] = [base + (1 if i < rem else 0) for i in range(total_ranks)]
        # reshape to [cp][sp]
        by_cp_sp: List[List[int]] = []
        idx = 0
        for _ in range(self.cp_size):
            row = []
            for _ in range(self.sp_size):
                row.append(blocks_flat[idx] * self.block_size)
                idx += 1
            by_cp_sp.append(row)
        return by_cp_sp

    def runner_even_distribution_tokens(self) -> List[List[int]]:
        """
        Simulate NPUModelRunner._slot_mapping_prefill_cp using an even split of
        the CP-padded tokens across cp*sp ranks.
        """
        total_padded = ceil_div(self.num_new_tokens, 2 * self.cp_size) * (2 * self.cp_size)
        total_ranks = self.cp_size * self.sp_size
        per_rank = total_padded // total_ranks
        # reshape to [cp][sp]
        by_cp_sp: List[List[int]] = []
        for _ in range(self.cp_size):
            row = []
            for _ in range(self.sp_size):
                row.append(per_rank)
            by_cp_sp.append(row)
        return by_cp_sp

    def cp_pad(self) -> int:
        total_padded = ceil_div(self.num_new_tokens, 2 * self.cp_size) * (2 * self.cp_size)
        return total_padded - self.num_new_tokens


def accumulate_lengths(per_rank_tokens_steps: List[List[List[int]]]) -> List[List[int]]:
    """
    Accumulate per-rank token lengths across steps. Shape: steps x cp x sp -> cp x sp
    """
    if not per_rank_tokens_steps:
        return []
    cp_size = len(per_rank_tokens_steps[0])
    sp_size = len(per_rank_tokens_steps[0][0]) if cp_size > 0 else 0
    acc = [[0 for _ in range(sp_size)] for _ in range(cp_size)]
    for step in per_rank_tokens_steps:
        for i in range(cp_size):
            for j in range(sp_size):
                acc[i][j] += step[i][j]
    return acc


def logits_index_for_cp_prefill(cu_num_tokens: List[int], cp_size: int, cp_pads: List[int]) -> List[int]:
    """
    Mirror runner logic:
      logits_indices = cu_num_tokens * cp_size - num_cp_pads - 1
    where inputs are per-request cumulative scheduled tokens for the current step
    and the cp padding for each request.
    """
    assert len(cu_num_tokens) == len(cp_pads)
    out = []
    for cu, pad in zip(cu_num_tokens, cp_pads):
        out.append(cu * cp_size - pad - 1)
    return out


def cu_scheduled(num_scheduled_each_req: List[int]) -> List[int]:
    out = []
    s = 0
    for n in num_scheduled_each_req:
        s += n
        out.append(s)
    return out


