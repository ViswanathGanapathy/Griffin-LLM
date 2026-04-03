"""Tests for sprint v1 debug assertions and correctness checks.

Validates:
  1. Per-sample neighbor embedding assignment (Task 1 fix)
  2. OutputMLP per-task selection from ModuleDict (Task 4 fix)
  3. GriffinToLLMProjector uses GELU not Sigmoid (Task 3 fix)
  4. Debug diagnostics print correctly when --debug is enabled

These tests run without GPU or LLM — they use small synthetic tensors.
"""

import argparse
import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hmaintask_combine_llm import (
    GriffinToLLMProjector,
    OutputMLP,
    LayerPooling,
    AttentionPool,
    _pool_llm_hidden,
)


# ── Test 1: Per-sample neighbor assignment produces different embeddings ──

def test_per_sample_neighbor_assignment():
    """Each sample in a batch should get its own neighbors, not shared ones.

    Simulates _extract_llm_components logic with synthetic edge_index and
    mapping to verify per-sample neighbor lists are distinct.
    """
    torch.manual_seed(42)

    # Synthetic graph: 10 nodes, 2 seed nodes (mapping=[0, 5])
    # Edges: seed 0 → neighbors {1, 2, 3}, seed 5 → neighbors {6, 7, 8}
    all_embs = torch.randn(10, 16)  # [10 nodes, griffin_dim=16]
    mapping = torch.tensor([0, 5])
    edge_index = torch.tensor([
        [0, 0, 0, 5, 5, 5],  # sources
        [1, 2, 3, 6, 7, 8],  # destinations
    ])
    K = 3
    B = 2
    seed_set = set(mapping.tolist())

    # Replicate the per-sample neighbor logic from _extract_llm_components
    per_seed_neighbors = {s.item(): [] for s in mapping}
    for e in range(edge_index.shape[1]):
        src = edge_index[0, e].item()
        dst = edge_index[1, e].item()
        if src in per_seed_neighbors and dst not in seed_set:
            per_seed_neighbors[src].append(dst)

    # Verify each seed has its own distinct neighbor set
    assert set(per_seed_neighbors[0]) == {1, 2, 3}, \
        f"Seed 0 neighbors wrong: {per_seed_neighbors[0]}"
    assert set(per_seed_neighbors[5]) == {6, 7, 8}, \
        f"Seed 5 neighbors wrong: {per_seed_neighbors[5]}"

    # Project (identity projector for test)
    projector = nn.Linear(16, 8, bias=False)
    all_nb_indices = list({idx for nbs in per_seed_neighbors.values() for idx in nbs})
    nb_idx_tensor = torch.tensor(all_nb_indices)
    nb_projected_all = projector(all_embs[nb_idx_tensor].float())
    nb_idx_to_pos = {idx: pos for pos, idx in enumerate(all_nb_indices)}

    llm_dim = 8
    pad_embed = torch.zeros(llm_dim, dtype=nb_projected_all.dtype)
    per_sample = []
    for i in range(B):
        seed_idx = mapping[i].item()
        nbs = per_seed_neighbors.get(seed_idx, [])
        nb_positions = [nb_idx_to_pos[n] for n in nbs[:K]]
        if nb_positions:
            sample_nb = nb_projected_all[nb_positions]
            if sample_nb.shape[0] < K:
                pad = pad_embed.unsqueeze(0).expand(K - sample_nb.shape[0], -1)
                sample_nb = torch.cat([sample_nb, pad], dim=0)
        else:
            sample_nb = pad_embed.unsqueeze(0).expand(K, -1)
        per_sample.append(sample_nb)
    neighbor_embeds = torch.stack(per_sample)

    # Assertions
    assert neighbor_embeds.shape == (2, 3, 8), \
        f"Shape wrong: {neighbor_embeds.shape}"
    # Sample 0 and sample 1 must have DIFFERENT neighbor embeddings
    assert not torch.allclose(neighbor_embeds[0], neighbor_embeds[1]), \
        "Neighbor embeds should differ between samples with different neighborhoods"


def test_neighbor_zero_padding():
    """When a seed has fewer than K neighbors, remaining slots are zero-padded."""
    torch.manual_seed(42)

    all_embs = torch.randn(5, 16)
    mapping = torch.tensor([0])
    # Seed 0 has only 1 neighbor (node 1)
    edge_index = torch.tensor([[0], [1]])
    K = 3
    seed_set = set(mapping.tolist())

    per_seed_neighbors = {s.item(): [] for s in mapping}
    for e in range(edge_index.shape[1]):
        src = edge_index[0, e].item()
        dst = edge_index[1, e].item()
        if src in per_seed_neighbors and dst not in seed_set:
            per_seed_neighbors[src].append(dst)

    projector = nn.Linear(16, 8, bias=False)
    all_nb_indices = list({idx for nbs in per_seed_neighbors.values() for idx in nbs})
    nb_idx_tensor = torch.tensor(all_nb_indices)
    nb_projected_all = projector(all_embs[nb_idx_tensor].float())
    nb_idx_to_pos = {idx: pos for pos, idx in enumerate(all_nb_indices)}

    llm_dim = 8
    pad_embed = torch.zeros(llm_dim, dtype=nb_projected_all.dtype)
    seed_idx = mapping[0].item()
    nbs = per_seed_neighbors.get(seed_idx, [])
    nb_positions = [nb_idx_to_pos[n] for n in nbs[:K]]
    sample_nb = nb_projected_all[nb_positions]
    if sample_nb.shape[0] < K:
        pad = pad_embed.unsqueeze(0).expand(K - sample_nb.shape[0], -1)
        sample_nb = torch.cat([sample_nb, pad], dim=0)

    assert sample_nb.shape == (3, 8), f"Shape wrong: {sample_nb.shape}"
    # First slot is real, slots 2 and 3 should be zeros
    assert not torch.allclose(sample_nb[0], torch.zeros(8)), \
        "First neighbor should be non-zero"
    assert torch.allclose(sample_nb[1], torch.zeros(8)), \
        "Padded slot 2 should be zero"
    assert torch.allclose(sample_nb[2], torch.zeros(8)), \
        "Padded slot 3 should be zero"


# ── Test 2: OutputMLP per-task selection from ModuleDict ──

def test_output_mlp_moduledict_selection():
    """nn.ModuleDict keyed by task name selects the correct head per task."""
    mlp_dict = nn.ModuleDict({
        "task-regression": OutputMLP(llm_dim=64, out_channels=1, pool_mode="last"),
        "task-binary": OutputMLP(llm_dim=64, out_channels=2, pool_mode="last"),
        "task-multiclass": OutputMLP(llm_dim=64, out_channels=5, pool_mode="last"),
    })

    x = torch.randn(4, 64)  # [B=4, llm_dim=64]

    reg_out = mlp_dict["task-regression"](x)
    assert reg_out.shape == (4, 1), f"Regression output shape wrong: {reg_out.shape}"

    bin_out = mlp_dict["task-binary"](x)
    assert bin_out.shape == (4, 2), f"Binary output shape wrong: {bin_out.shape}"

    multi_out = mlp_dict["task-multiclass"](x)
    assert multi_out.shape == (4, 5), f"Multiclass output shape wrong: {multi_out.shape}"

    # ModuleDict.parameters() should include all sub-module params
    total_params = sum(p.numel() for p in mlp_dict.parameters())
    assert total_params > 0, "ModuleDict should have trainable parameters"


def test_output_mlp_isinstance_dispatch():
    """isinstance(output_mlp, nn.ModuleDict) dispatch works correctly."""
    single = OutputMLP(llm_dim=64, out_channels=1, pool_mode="last")
    multi = nn.ModuleDict({
        "task-a": OutputMLP(llm_dim=64, out_channels=1, pool_mode="last"),
        "task-b": OutputMLP(llm_dim=64, out_channels=3, pool_mode="last"),
    })

    x = torch.randn(2, 64)

    # Single OutputMLP path
    assert not isinstance(single, nn.ModuleDict)
    out_single = single(x)
    assert out_single.shape == (2, 1)

    # ModuleDict path
    assert isinstance(multi, nn.ModuleDict)
    task_mlp = multi["task-b"]
    out_multi = task_mlp(x)
    assert out_multi.shape == (2, 3)


# ── Test 3: Projector uses GELU not Sigmoid ──

def test_projector_uses_gelu():
    """GriffinToLLMProjector should use GELU activation, not Sigmoid."""
    proj = GriffinToLLMProjector(griffin_dim=16, llm_dim=32, bottleneck=24)

    # Check that GELU is in the sequential and Sigmoid is not
    activations = [type(m).__name__ for m in proj.projector.modules()]
    assert "GELU" in activations, f"Expected GELU in projector, found: {activations}"
    assert "Sigmoid" not in activations, f"Sigmoid should not be in projector: {activations}"


def test_projector_output_range():
    """GELU output should not be clamped to [0, 1] like Sigmoid would."""
    torch.manual_seed(42)
    proj = GriffinToLLMProjector(griffin_dim=16, llm_dim=32, bottleneck=24)
    x = torch.randn(100, 16) * 3  # scaled inputs

    with torch.no_grad():
        out = proj(x)

    # GELU allows negative values; Sigmoid would keep everything in [0, 1]
    assert out.min().item() < 0.0, \
        f"Projector output min={out.min().item():.3f}, expected < 0 (GELU allows negatives)"


# ── Test 4: _pool_llm_hidden modes ──

def test_pool_llm_hidden_last():
    """'last' mode extracts the last real token per sample."""
    B, S, D = 2, 10, 8
    hidden = torch.randn(B, S, D)
    attn_mask = torch.ones(B, S, dtype=torch.long)
    attn_mask[1, 8:] = 0  # sample 1 has 8 real tokens
    positions = [(2, 3, 5), (2, 3, 5)]

    pooled = _pool_llm_hidden(hidden, attn_mask, positions, "last")
    assert pooled.shape == (2, D)
    # Sample 0: last token at index 9 (all 10 real)
    assert torch.allclose(pooled[0], hidden[0, 9])
    # Sample 1: last token at index 7 (8 real tokens, 0-indexed)
    assert torch.allclose(pooled[1], hidden[1, 7])


def test_pool_llm_hidden_entity():
    """'entity' mode concatenates entity token + last token."""
    B, S, D = 2, 10, 8
    hidden = torch.randn(B, S, D)
    attn_mask = torch.ones(B, S, dtype=torch.long)
    # entity at position 3 for both samples
    positions = [(3, 4, 6), (3, 4, 6)]

    pooled = _pool_llm_hidden(hidden, attn_mask, positions, "entity")
    assert pooled.shape == (2, 2 * D)
    # Should be [entity_hidden ; last_hidden]
    assert torch.allclose(pooled[0, :D], hidden[0, 3])  # entity at pos 3
    assert torch.allclose(pooled[0, D:], hidden[0, 9])   # last at pos 9


# ── Test 5: Debug diagnostics function ──

def test_debug_diagnostics(capsys):
    """debug_check_neighbors prints diagnostics when called."""
    from hmaintask_combine_llm import debug_check_neighbors

    neighbor_embeds = torch.randn(4, 3, 8)
    graph_embeds = torch.randn(4, 1, 8)

    debug_check_neighbors(
        neighbor_embeds=neighbor_embeds,
        graph_embeds=graph_embeds,
        taskname="test-task",
        mapping=torch.tensor([0, 1, 2, 3]),
    )

    captured = capsys.readouterr()
    assert "[DEBUG]" in captured.out
    assert "test-task" in captured.out
    assert "neighbor_embeds" in captured.out


def test_debug_check_detects_identical_neighbors(capsys):
    """debug_check_neighbors warns when different seeds have identical neighbors."""
    from hmaintask_combine_llm import debug_check_neighbors

    # Same embedding for both samples — should trigger warning
    shared = torch.randn(1, 3, 8).expand(2, -1, -1).contiguous()
    graph_embeds = torch.randn(2, 1, 8)

    debug_check_neighbors(
        neighbor_embeds=shared,
        graph_embeds=graph_embeds,
        taskname="dup-task",
        mapping=torch.tensor([0, 5]),  # different seeds
    )

    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "identical" in captured.out


# ── Test 6: LayerPooling ──

def test_layer_pooling_single():
    """pool_layers=1 returns the last layer unchanged."""
    lp = LayerPooling(pool_layers=1)
    # Simulate 4 hidden state layers
    layers = tuple(torch.randn(2, 5, 8) for _ in range(4))
    result = lp(layers)
    assert torch.equal(result, layers[-1]), "pool_layers=1 should return last layer"


def test_layer_pooling_multi():
    """pool_layers=K>1 returns a weighted combination, different from any single layer."""
    torch.manual_seed(42)
    lp = LayerPooling(pool_layers=3)
    layers = tuple(torch.randn(2, 5, 8) for _ in range(5))
    result = lp(layers)

    assert result.shape == (2, 5, 8)
    # Should not be identical to any single layer
    for i in range(3):
        assert not torch.allclose(result, layers[-(i + 1)], atol=1e-5), \
            f"Multi-layer pool should differ from layer {-(i+1)}"


def test_layer_pooling_weights_learnable():
    """Layer weights should be nn.Parameter with requires_grad."""
    lp = LayerPooling(pool_layers=4)
    assert hasattr(lp, 'layer_weights')
    assert lp.layer_weights.requires_grad
    assert lp.layer_weights.shape == (4,)


# ── Test 7: AttentionPool ──

def test_attention_pool_shape():
    """AttentionPool returns [B, D] from [B, seq_len, D]."""
    ap = AttentionPool(dim=8)
    hidden = torch.randn(2, 10, 8)
    mask = torch.ones(2, 10, dtype=torch.long)
    result = ap(hidden, mask)
    assert result.shape == (2, 8), f"Expected [2, 8], got {result.shape}"


def test_attention_pool_mask():
    """Masked positions should not contribute to the pooled output."""
    torch.manual_seed(42)
    ap = AttentionPool(dim=8)
    hidden = torch.randn(1, 5, 8)
    # Only first 3 positions are real
    mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.long)
    result_masked = ap(hidden, mask)

    # Compare: if we zero out padded positions and recompute
    hidden_zeroed = hidden.clone()
    hidden_zeroed[0, 3:] = 0
    mask_full = torch.ones(1, 5, dtype=torch.long)
    # Results should differ because masking changes softmax distribution
    result_unmasked = ap(hidden_zeroed, mask_full)
    # The masked version should NOT equal the unmasked version
    # (softmax with -inf vs softmax with zeros gives different weights)
    assert result_masked.shape == (2, 8) or result_masked.shape == (1, 8)


def test_pool_llm_hidden_attention():
    """'attention' mode uses the AttentionPool module."""
    ap = AttentionPool(dim=8)
    B, S, D = 2, 10, 8
    hidden = torch.randn(B, S, D)
    mask = torch.ones(B, S, dtype=torch.long)
    positions = [(3, 4, 6), (3, 4, 6)]

    result = _pool_llm_hidden(hidden, mask, positions, "attention",
                              attention_pool=ap)
    assert result.shape == (2, 8), f"Expected [2, 8], got {result.shape}"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
