from __future__ import annotations

import pytest

from tests.standalone.cp_chunk_sim import (
    CPChunkStep,
    accumulate_lengths,
    cu_scheduled,
    logits_index_for_cp_prefill,
)


@pytest.mark.parametrize("cp_size", [2, 4])
def test_scheduler_vs_runner_distribution_for_chunk(cp_size: int):
    # Prompt 2497, chunk 1024 -> first step schedules 1024 tokens.
    step = CPChunkStep(num_new_tokens=1024, cp_size=cp_size, sp_size=1, block_size=16)
    sched = step.scheduler_distribution_tokens()
    run = step.runner_even_distribution_tokens()

    # Compare total per (cp,sp)
    sched_tot = sum(sum(r) for r in sched)
    run_tot = sum(sum(r) for r in run)
    assert sched_tot >= run_tot  # scheduler may over-allocate blocks to cover tokens

    # The runner saves exactly even tokens per rank for CP-padded amount
    total_padded = (1024 + 2 * cp_size - 1) // (2 * cp_size) * (2 * cp_size)
    per_rank = total_padded // (cp_size * 1)
    for i in range(cp_size):
        assert run[i][0] == per_rank


def test_logits_indices_bounds_cp2_cp4_chunk_first_step():
    # One request, scheduled tokens = 1024, cp_pads depend on cp_size.
    num_scheduled = [1024]
    cu = cu_scheduled(num_scheduled)
    for cp_size in (2, 4):
        # cp_pad = ceil_div(1024, 2*cp) * (2*cp) - 1024
        total_padded = ((1024 + 2 * cp_size - 1) // (2 * cp_size)) * (2 * cp_size)
        cp_pad = total_padded - 1024
        logits_idx = logits_index_for_cp_prefill(cu, cp_size, [cp_pad])
        # Must be within [0, total_padded*cp_size - 1] after gather across cp
        max_idx = total_padded * cp_size - 1
        assert 0 <= logits_idx[0] <= max_idx


def test_multi_step_accumulation_and_even_saving_cp4():
    # Simulate 3 steps for cp=4 with lengths [1024, 1024, 449]
    steps = [
        CPChunkStep(1024, 4, 1, 16),
        CPChunkStep(1024, 4, 1, 16),
        CPChunkStep(449, 4, 1, 16),
    ]
    per_step = [s.runner_even_distribution_tokens() for s in steps]
    acc = accumulate_lengths(per_step)
    # Each step pads to multiple of 8 tokens (2*cp=8) and splits across 4 ranks.
    # Step1: 1024 -> padded 1024, per_rank 256
    # Step2: 1024 -> padded 1024, per_rank 256
    # Step3: 449  -> padded 456,  per_rank 114
    assert acc == [[256 + 256 + 114], [256 + 256 + 114], [256 + 256 + 114], [256 + 256 + 114]]


def test_mismatch_scheduler_vs_runner_cp4_uneven_last_chunk():
    # Last chunk 449 with block_size=16, cp=4
    s = CPChunkStep(449, 4, 1, 16)
    sched = s.scheduler_distribution_tokens()
    run = s.runner_even_distribution_tokens()
    # Scheduler by blocks -> [128,112,112,112], Runner by padded-even -> [114,114,114,114]
    assert sched != run
    assert sched[0][0] == 128 and all(v == 112 for v in [sched[i][0] for i in range(1, 4)])
    assert all(run[i][0] == 114 for i in range(4))


