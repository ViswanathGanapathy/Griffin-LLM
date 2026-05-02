"""SMPNN-Griffin warm-start sanity check.

Run BEFORE kicking off A/B training. Verifies:
  1. The migration loader can read a legacy Griffin checkpoint into a
     ``use_smpnn=True`` GriffinMod with zero unexpected keys.
  2. SMPNN-only params (ln1, ln2, alpha_ff) keep their identity-style init
     after warm-start (otherwise the "warm-start ≈ identity at step 0"
     property the paper relies on is broken).
  3. Both legacy and SMPNN warm-started models produce finite output of the
     expected shape on a synthetic forward pass.
  4. At step 0 the SMPNN block is approximately identity per-layer
     (per-layer ``|x_out - x_in|_inf`` should be on the order of alpha_ff).

Usage:
    python smpnn_sanity_check.py [--ckpt PATH] [--hiddim H] [--num_mp L]

Defaults to the ``checkpoints/single-completion`` checkpoint that ships with
the repo (hiddim=512, num_mp=4). For the deep-SMPNN A/B (num_mp=6) pass
``--num_mp 6`` — the extra layers stay at identity init.
"""

import argparse
import sys

import torch

import accelerate
from hmaintask_completion import _load_legacy_into_smpnn
from hmodel import GriffinMod


def build_synthetic_batch(hiddim: int, seed: int = 0):
    """Synthetic heterogeneous batch with the structure GriffinMod expects.

    Returns args matching ``GriffinMod.forward(node, mask, taskfeat, edge_index, edge_attr_type, edge_attr)``.
    """
    g = torch.Generator().manual_seed(seed)
    NUM_TYPES = 3
    NUM_FEATS = [4, 6, 5]
    NUM_NODES = [8, 10, 7]
    NUM_EDGES = 32
    NUM_RELATIONS = 5
    node = [
        (
            torch.randn(NUM_FEATS[t], hiddim, generator=g),
            torch.randn(NUM_NODES[t], NUM_FEATS[t], hiddim, generator=g),
        )
        for t in range(NUM_TYPES)
    ]
    mask = [None] * NUM_TYPES
    taskfeat = [None] * NUM_TYPES
    total = sum(NUM_NODES)
    edge_index = torch.randint(0, total, (2, NUM_EDGES), generator=g)
    edge_attr_type = torch.randint(0, NUM_RELATIONS, (NUM_EDGES,), generator=g)
    edge_attr = torch.randn(NUM_RELATIONS, hiddim, generator=g)
    return node, mask, taskfeat, edge_index, edge_attr_type, edge_attr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="checkpoints/single-completion")
    ap.add_argument("--hiddim", type=int, default=512)
    ap.add_argument("--num_mp", type=int, default=4,
                    help="Layers in the SMPNN model. Pass 6 to test the deep-SMPNN config; "
                         "the extra layers will be at identity init.")
    ap.add_argument("--use_rev", action="store_true", default=True)
    ap.add_argument("--use_gate", action="store_true", default=True)
    args = ap.parse_args()

    print(f"== SMPNN warm-start sanity check ==")
    print(f"  ckpt   = {args.ckpt}")
    print(f"  hiddim = {args.hiddim}")
    print(f"  num_mp = {args.num_mp}")
    print()

    failures: list[str] = []

    # --- 1. Build legacy and SMPNN models, warm-start both -----------------
    torch.manual_seed(0)
    m_legacy = GriffinMod(hiddim=args.hiddim, num_mp=args.num_mp,
                          use_rev=args.use_rev, use_gate=args.use_gate,
                          use_smpnn=False)
    accelerate.load_checkpoint_in_model(m_legacy, args.ckpt)
    m_legacy.eval()

    torch.manual_seed(0)
    m_smpnn = GriffinMod(hiddim=args.hiddim, num_mp=args.num_mp,
                         use_rev=args.use_rev, use_gate=args.use_gate,
                         use_smpnn=True)
    # Snapshot SMPNN-only params BEFORE load to verify they're untouched
    ln1_w_pre = [m_smpnn.ln1[i].weight.detach().clone() for i in range(args.num_mp)]
    ln2_b_pre = [m_smpnn.ln2[i].bias.detach().clone() for i in range(args.num_mp)]
    alpha_pre = m_smpnn.alpha_ff.detach().clone()

    _load_legacy_into_smpnn(m_smpnn, args.ckpt)
    m_smpnn.eval()

    # --- 2. SMPNN-only params should be untouched by warm-start ------------
    for i in range(args.num_mp):
        if not torch.equal(ln1_w_pre[i], m_smpnn.ln1[i].weight):
            failures.append(f"ln1[{i}].weight mutated by warm-start")
        if not torch.equal(ln2_b_pre[i], m_smpnn.ln2[i].bias):
            failures.append(f"ln2[{i}].bias mutated by warm-start")
    if not torch.equal(alpha_pre, m_smpnn.alpha_ff):
        failures.append("alpha_ff mutated by warm-start")
    if not torch.allclose(m_smpnn.alpha_ff, torch.full_like(m_smpnn.alpha_ff, 1e-6)):
        failures.append(f"alpha_ff != 1e-6 (got {m_smpnn.alpha_ff.tolist()})")
    print(f"[2/4] SMPNN-only init preserved: alpha_ff={m_smpnn.alpha_ff.detach().tolist()}")

    # --- 3. Forward pass on synthetic batch produces finite output ---------
    batch = build_synthetic_batch(args.hiddim)
    with torch.no_grad():
        out_legacy = m_legacy(*batch)
        # Rebuild taskfeat (it's a list mutated in-place during forward)
        batch2 = build_synthetic_batch(args.hiddim)
        out_smpnn = m_smpnn(*batch2)

    n_total = sum(t[1].shape[0] for t in batch[0])
    expect_shape = (n_total, args.hiddim)
    if out_legacy.shape != expect_shape:
        failures.append(f"legacy out shape {tuple(out_legacy.shape)} != {expect_shape}")
    if out_smpnn.shape != expect_shape:
        failures.append(f"smpnn out shape {tuple(out_smpnn.shape)} != {expect_shape}")
    if not torch.isfinite(out_legacy).all():
        failures.append("legacy output has non-finite entries")
    if not torch.isfinite(out_smpnn).all():
        failures.append("smpnn output has non-finite entries")

    legacy_stats = (out_legacy.mean().item(), out_legacy.std().item(), out_legacy.abs().max().item())
    smpnn_stats = (out_smpnn.mean().item(), out_smpnn.std().item(), out_smpnn.abs().max().item())
    diff_stats = ((out_legacy - out_smpnn).abs().mean().item(),
                  (out_legacy - out_smpnn).abs().max().item())
    print(f"[3/4] Forward shapes OK: {expect_shape}")
    print(f"      legacy out  mean/std/|max| = {legacy_stats[0]:+.3e}  {legacy_stats[1]:.3e}  {legacy_stats[2]:.3e}")
    print(f"      smpnn  out  mean/std/|max| = {smpnn_stats[0]:+.3e}  {smpnn_stats[1]:.3e}  {smpnn_stats[2]:.3e}")
    print(f"      |legacy - smpnn|  mean/max = {diff_stats[0]:.3e}  {diff_stats[1]:.3e}")
    print(f"      ^ NOTE: legacy adds an unscaled mlp2 contribution per layer;")
    print(f"        SMPNN scales it by alpha_ff=1e-6. Output divergence here is")
    print(f"        EXPECTED — it's the entire reason SMPNN is identity-at-init.")

    # --- 4. SMPNN per-layer block is approximately identity at step 0 ------
    # Bypass the cross-attention path; feed random x straight into _block_update.
    N = 16
    x = torch.randn(N, args.hiddim)
    edge_index = torch.randint(0, N, (2, 24))
    edge_attr_type = torch.randint(0, 5, (24,))
    edge_attr = torch.randn(5, args.hiddim)
    max_per_layer_delta = 0.0
    with torch.no_grad():
        for i in range(args.num_mp):
            x_in = x.clone()
            x_out = m_smpnn._block_update(i, x_in, edge_index, edge_attr_type, edge_attr)
            d = (x_out - x_in).abs().max().item()
            max_per_layer_delta = max(max_per_layer_delta, d)
            x = x_out
    # Bound: alpha_ff * mlp2_output_max. With alpha=1e-6 and unit-scale x,
    # |delta| should be O(1e-5) or smaller. Anything larger means gates aren't
    # zero (e.g. ckpt overwrote gate weights with non-zero values) or
    # alpha_ff isn't 1e-6.
    BOUND = 1e-3
    if max_per_layer_delta > BOUND:
        failures.append(
            f"SMPNN block update is NOT identity-at-init: max per-layer "
            f"|x_out - x_in|_inf = {max_per_layer_delta:.3e} > {BOUND}. "
            f"Gates may have been loaded with non-zero values, or alpha_ff drifted."
        )
    print(f"[4/4] SMPNN identity-at-init: max per-layer |Δx|_inf = {max_per_layer_delta:.3e}  (bound: {BOUND:.0e})")

    # --- summary ---
    print()
    if failures:
        print("FAIL — issues found:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("PASS — warm-start is safe to use for A/B training.")


if __name__ == "__main__":
    main()
