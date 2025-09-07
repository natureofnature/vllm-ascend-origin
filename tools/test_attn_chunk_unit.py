import argparse
import os
import sys

import torch

try:
    import torch_npu  # type: ignore
except Exception as e:
    print("torch_npu not available:", e)
    sys.exit(2)


@torch.inference_mode()
def ring_attn(q_nope, q_rope, k_nope, k_rope, v, q_len: int, kv_len: int, head_num: int, scale: float):
    device = q_nope.device
    dtype = q_nope.dtype
    mask = torch.ones(512, 512, device=device, dtype=dtype).triu(1)  # not used when mask_type=no_mask
    seqlen = torch.tensor([q_len, kv_len], dtype=torch.int32, device=device).unsqueeze(1)  # [2,1]
    out, lse = torch_npu.atb.npu_ring_mla(
        q_nope=q_nope,
        q_rope=q_rope,
        k_nope=k_nope,
        k_rope=k_rope,
        value=v,
        mask=mask,
        seqlen=seqlen,
        head_num=head_num,
        kv_head_num=head_num,
        qk_scale=scale,
        kernel_type="kernel_type_high_precision",
        mask_type="no_mask",
        input_layout="type_bsnd",
        calc_type="calc_type_default",
    )
    # Convert lse to [T, H, 1] for stable update broadcasting
    lse_bt = lse.permute(1, 0).unsqueeze(-1).contiguous()
    return out, lse_bt


def stable_update(out, lse, block_out, block_lse):
    # all in fp32 for stability
    if out is None:
        return block_out.to(torch.float32), block_lse.to(torch.float32)
    out_f = out.to(torch.float32)
    lse_f = lse.to(torch.float32)
    block_out_f = block_out.to(torch.float32)
    block_lse_f = block_lse.to(torch.float32)
    out_new = out_f - torch.sigmoid(block_lse_f - lse_f) * (out_f - block_out_f)
    lse_new = lse_f - torch.logsigmoid(lse_f - block_lse_f)
    return out_new, lse_new


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description="Unit test: chunked vs full ring MLA attention")
    ap.add_argument("--total_tokens", type=int, default=2974)
    ap.add_argument("--chunk_size", type=int, default=1024)
    ap.add_argument("--num_heads", type=int, default=16)
    ap.add_argument("--qk_nope", type=int, default=128)
    ap.add_argument("--qk_rope", type=int, default=64)
    ap.add_argument("--v_dim", type=int, default=128)
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--rtol", type=float, default=1e-3)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("npu")
    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    T = args.total_tokens
    H = args.num_heads
    dn = args.qk_nope
    dr = args.qk_rope
    dv = args.v_dim
    scale = 1.0 / (dn ** 0.5)

    # Random inputs
    q_nope = torch.randn(T, H, dn, device=device, dtype=dtype)
    q_rope = torch.randn(T, H, dr, device=device, dtype=dtype)
    K = T  # let kv_len == total tokens for test
    k_nope_full = torch.randn(K, H, dn, device=device, dtype=dtype)
    k_rope_full = torch.randn(K, H, dr, device=device, dtype=dtype)
    v_full = torch.randn(K, H, dv, device=device, dtype=dtype)

    # Baseline: one full call
    out_full, lse_full = ring_attn(q_nope, q_rope, k_nope_full, k_rope_full, v_full, T, K, H, scale)
    out_full = out_full.to(torch.float32)

    # Chunked: split KV into chunks and accumulate
    out_acc = None
    lse_acc = None
    remaining = K
    start = 0
    while remaining > 0:
        ck = min(args.chunk_size, remaining)
        k_nope = k_nope_full[start:start+ck]
        k_rope = k_rope_full[start:start+ck]
        v = v_full[start:start+ck]
        out_blk, lse_blk = ring_attn(q_nope, q_rope, k_nope, k_rope, v, T, ck, H, scale)
        out_acc, lse_acc = stable_update(out_acc, lse_acc, out_blk, lse_blk)
        start += ck
        remaining -= ck

    # Compare
    diff = (out_full - out_acc).abs()
    mae = diff.mean().item()
    maxe = diff.max().item()
    ok = torch.allclose(out_full, out_acc, atol=args.atol, rtol=args.rtol)
    print(f"tokens={T}, chunk={args.chunk_size}, mae={mae:.4e}, max={maxe:.4e}, allclose={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())


