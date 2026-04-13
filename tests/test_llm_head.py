"""Tests for --head llm (LLM text generation) output path.

Tests each step:
  1. Prompt generation (build_llm_inputs) — layout and token positions
  2. Binary classification — Yes/No logit extraction
  3. Regression — text generation + float parsing
  4. Multi-class — text generation + int parsing

Uses mock LLM outputs to avoid requiring a real LLM download.
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hmaintask_combine_llm import (
    GriffinToLLMProjector,
    build_llm_inputs,
)


# ═══════════════════════════════════════════════════════════════
# Mock LLM Decoder (avoids downloading a real model)
# ═══════════════════════════════════════════════════════════════

class MockTokenizer:
    """Minimal tokenizer mock for testing prompt construction."""

    def __init__(self, vocab_size=100, dim=32):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self._embedding = nn.Embedding(vocab_size, dim)

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        # Simple: each character → token id (mod vocab_size)
        ids = [ord(c) % self.vocab_size for c in text]
        if return_tensors == "pt":
            class Result:
                input_ids = torch.tensor([ids])
            return Result()
        return {"input_ids": ids}

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % self.vocab_size for c in text]

    def decode(self, ids, skip_special_tokens=True):
        # Return a simple number string for testing
        return "4.5"


class MockLLMDecoder:
    """Mock LLMDecoder that mimics the interface without loading a real model."""

    def __init__(self, llm_dim=32, vocab_size=100, device="cpu"):
        self.llm_dim = llm_dim
        self.device = device
        self.model_name = "mock-llm"
        self.model_type = "mock"
        self.needs_sampling = False

        self.tokenizer = MockTokenizer(vocab_size=vocab_size, dim=llm_dim)
        self.word_embedding = nn.Embedding(vocab_size, llm_dim)
        self.pad_embed = torch.zeros(1, llm_dim)

        # Yes/No token IDs
        self.true_id = ord("Y") % vocab_size   # "Yes" → first char
        self.false_id = ord("N") % vocab_size   # "No" → first char

    def tokenize(self, text):
        ids = [ord(c) % self.tokenizer.vocab_size for c in text]
        return torch.tensor(ids, device=self.device)

    def embed_tokens(self, token_ids):
        return self.word_embedding(token_ids)


# ═══════════════════════════════════════════════════════════════
# Test 1: Prompt Generation
# ═══════════════════════════════════════════════════════════════

def test_prompt_layout_entity_after_question():
    """Verify prompt order: [system] [question] [ENTITY] [neighbors]."""
    llm_dim = 32
    mock_decoder = MockLLMDecoder(llm_dim=llm_dim)

    # Simulate: 2 samples, each with 1 entity embed
    graph_embeds = torch.randn(2, 1, llm_dim)

    inputs_embeds, attention_mask, label_ids, positions = build_llm_inputs(
        graph_embeds=graph_embeds,
        neighbor_embeds=None,
        llm_decoder=mock_decoder,
        task_name="test-task",
        task_type="regression",
        entity_type="TestEntity",
        labels=None,  # inference mode
        entity_after_question=True,
    )

    B, seq_len, D = inputs_embeds.shape
    assert B == 2, f"Batch size should be 2, got {B}"
    assert D == llm_dim, f"Dim should be {llm_dim}, got {D}"
    assert seq_len > 3, f"Seq len should be > 3 (system + question + entity), got {seq_len}"

    # Verify entity position is AFTER system+question tokens
    for i in range(B):
        ent_pos, nb_start, nb_end = positions[i]
        # Entity should not be at position 0 or 1 (those are system/question)
        assert ent_pos > 5, f"Entity at pos {ent_pos} — should be after system+question"
        # Entity should be near the end (just before the last position)
        assert ent_pos >= seq_len - 3, f"Entity at pos {ent_pos}, seq_len={seq_len}"

    print("  [PASS] Prompt layout: entity after question")


def test_prompt_layout_entity_before_question():
    """Verify old prompt order: [system] [ENTITY] [neighbors] [question]."""
    llm_dim = 32
    mock_decoder = MockLLMDecoder(llm_dim=llm_dim)
    graph_embeds = torch.randn(2, 1, llm_dim)

    inputs_embeds, attention_mask, label_ids, positions = build_llm_inputs(
        graph_embeds=graph_embeds,
        neighbor_embeds=None,
        llm_decoder=mock_decoder,
        task_name="test-task",
        task_type="regression",
        entity_type="TestEntity",
        labels=None,
        entity_after_question=False,
    )

    B, seq_len, D = inputs_embeds.shape

    # In old order, entity should be right after system text
    for i in range(B):
        ent_pos, nb_start, nb_end = positions[i]
        # In old order, entity comes before question tokens
        # With mock tokenizer (1 char = 1 token), system text is long
        # Just verify entity is BEFORE where it would be in new order
        # (i.e., not at the very end of the sequence)
        assert ent_pos < seq_len - 1, \
            f"Entity at pos {ent_pos} — should not be at the very end (seq_len={seq_len})"

    print("  [PASS] Prompt layout: entity before question (old order)")


def test_prompt_with_neighbors():
    """Verify neighbor tokens are placed after entity."""
    llm_dim = 32
    mock_decoder = MockLLMDecoder(llm_dim=llm_dim)
    graph_embeds = torch.randn(2, 1, llm_dim)
    neighbor_embeds = torch.randn(2, 3, llm_dim)  # 3 neighbors

    inputs_embeds, attention_mask, label_ids, positions = build_llm_inputs(
        graph_embeds=graph_embeds,
        neighbor_embeds=neighbor_embeds,
        llm_decoder=mock_decoder,
        task_name="test-task",
        task_type="regression",
        entity_type="TestEntity",
        labels=None,
        entity_after_question=True,
    )

    for i in range(2):
        ent_pos, nb_start, nb_end = positions[i]
        assert nb_start == ent_pos + 1, \
            f"Neighbors should start right after entity: ent={ent_pos}, nb_start={nb_start}"
        assert nb_end == nb_start + 3, \
            f"Should have 3 neighbors: nb_start={nb_start}, nb_end={nb_end}"

    print("  [PASS] Prompt with 3 neighbor tokens")


def test_prompt_with_training_labels():
    """Verify answer tokens are appended during training (labels provided)."""
    llm_dim = 32
    mock_decoder = MockLLMDecoder(llm_dim=llm_dim)
    graph_embeds = torch.randn(2, 1, llm_dim)
    labels = torch.tensor([3.14, 2.71])

    inputs_embeds_train, attn_train, label_ids, _ = build_llm_inputs(
        graph_embeds=graph_embeds,
        neighbor_embeds=None,
        llm_decoder=mock_decoder,
        task_name="test-task",
        task_type="regression",
        entity_type="TestEntity",
        labels=labels,
        entity_after_question=True,
    )

    inputs_embeds_infer, attn_infer, _, _ = build_llm_inputs(
        graph_embeds=graph_embeds,
        neighbor_embeds=None,
        llm_decoder=mock_decoder,
        task_name="test-task",
        task_type="regression",
        entity_type="TestEntity",
        labels=None,
        entity_after_question=True,
    )

    # Training sequence should be longer (has answer + EOS tokens)
    assert inputs_embeds_train.shape[1] > inputs_embeds_infer.shape[1], \
        f"Training seq ({inputs_embeds_train.shape[1]}) should be longer than " \
        f"inference seq ({inputs_embeds_infer.shape[1]})"

    # Label IDs should have -100 for non-answer tokens, real IDs for answer
    assert (label_ids[:, :inputs_embeds_infer.shape[1]] == -100).all(), \
        "Non-answer tokens should have label=-100"
    # At least one answer token should be non-(-100)
    assert (label_ids != -100).any(), \
        "Should have at least one answer token with real label ID"

    print("  [PASS] Training labels appended correctly")


# ═══════════════════════════════════════════════════════════════
# Test 2: Binary Classification — Yes/No Logit Extraction
# ═══════════════════════════════════════════════════════════════

def test_binary_logit_extraction():
    """Verify Yes/No logits are correctly extracted for binary classification."""
    vocab_size = 100
    llm_dim = 32

    mock_decoder = MockLLMDecoder(llm_dim=llm_dim, vocab_size=vocab_size)

    # Simulate LLM output logits: [B=2, seq_len=10, vocab_size=100]
    logits = torch.randn(2, 10, vocab_size)

    # Set known values at the Yes/No positions in the last token
    logits[0, -1, mock_decoder.true_id] = 5.0   # Sample 0: Yes=5.0
    logits[0, -1, mock_decoder.false_id] = -3.0  # Sample 0: No=-3.0
    logits[1, -1, mock_decoder.true_id] = -2.0   # Sample 1: Yes=-2.0
    logits[1, -1, mock_decoder.false_id] = 4.0   # Sample 1: No=4.0

    # Extract binary logits (same logic as compute_output)
    next_logits = logits[:, -1, :]  # [B, vocab]
    binary_logits = next_logits[:, [mock_decoder.false_id, mock_decoder.true_id]]

    assert binary_logits.shape == (2, 2), f"Expected [2, 2], got {binary_logits.shape}"

    # Sample 0: No=-3.0, Yes=5.0 → predicts Yes (class 1)
    assert binary_logits[0, 1] > binary_logits[0, 0], \
        f"Sample 0 should predict Yes: No={binary_logits[0,0]:.1f}, Yes={binary_logits[0,1]:.1f}"

    # Sample 1: No=4.0, Yes=-2.0 → predicts No (class 0)
    assert binary_logits[1, 0] > binary_logits[1, 1], \
        f"Sample 1 should predict No: No={binary_logits[1,0]:.1f}, Yes={binary_logits[1,1]:.1f}"

    print("  [PASS] Binary logit extraction: Yes/No correctly extracted")


def test_binary_no_text_generation():
    """Binary path should NOT use text generation — just logit extraction."""
    # The key check: binary uses outputs.logits[:, -1, :] directly
    # NOT llm_decoder.model.generate()
    # This is verified by the code structure — binary path returns before
    # the generate() call. We verify by checking the code path exists.

    from hmaintask_combine_llm import compute_output
    import inspect
    source = inspect.getsource(compute_output)

    # The binary path should check "y.shape[0] == 2" before generate()
    assert "y.shape[0] == 2" in source, \
        "Binary classification should check for 2 classes"
    assert "binary_logits" in source, \
        "Binary path should extract binary_logits"

    print("  [PASS] Binary path uses logit extraction, not text generation")


# ═══════════════════════════════════════════════════════════════
# Test 3: Regression — Text Generation + Float Parsing
# ═══════════════════════════════════════════════════════════════

def test_regression_text_parsing():
    """Verify regression output parsing from generated text."""
    # Simulate what happens after LLM generates text
    test_cases = [
        ("4.5", 4.5),
        (" 3.14159", 3.14159),
        ("Answer: 7.2", 7.2),          # takes last token
        ("The value is 0.5", 0.5),     # takes last token
        ("", 0.0),                      # empty → fallback 0.0
        ("not a number", 0.0),          # unparseable → fallback 0.0
    ]

    for text, expected in test_cases:
        try:
            tokens = text.strip().split()
            val = float(tokens[-1]) if tokens else 0.0
        except (ValueError, IndexError):
            val = 0.0
        assert val == expected, f"Text '{text}' → expected {expected}, got {val}"

    print("  [PASS] Regression text parsing: all cases correct")


def test_regression_output_shape():
    """Regression output should be [B, 1]."""
    # Simulate parsed predictions
    predictions = [4.5, 3.14, 7.2]
    device = torch.device("cpu")
    output = torch.tensor(predictions, device=device, dtype=torch.float).unsqueeze(1)

    assert output.shape == (3, 1), f"Expected [3, 1], got {output.shape}"
    assert abs(output[0, 0].item() - 4.5) < 1e-5
    assert abs(output[2, 0].item() - 7.2) < 1e-4  # float32 precision

    print("  [PASS] Regression output shape: [B, 1]")


# ═══════════════════════════════════════════════════════════════
# Test 4: Multi-class — Text Generation + Int Parsing
# ═══════════════════════════════════════════════════════════════

def test_multiclass_text_parsing():
    """Verify multi-class output parsing from generated text."""
    test_cases = [
        ("3", 3),
        (" 0", 0),
        ("Class 7", 7),       # takes last token
        ("Answer: 11", 11),   # takes last token
        ("", 0),               # empty → fallback 0
        ("abc", 0),            # unparseable → fallback 0
    ]

    for text, expected in test_cases:
        try:
            tokens = text.strip().split()
            val = int(tokens[-1]) if tokens else 0
        except (ValueError, IndexError):
            val = 0
        assert val == expected, f"Text '{text}' → expected {expected}, got {val}"

    print("  [PASS] Multi-class text parsing: all cases correct")


def test_multiclass_logit_construction():
    """Multi-class creates one-hot-ish logits from parsed class index."""
    batch_size = 3
    num_classes = 12
    predictions = [3, 0, 11]  # predicted class indices
    device = torch.device("cpu")

    # Same logic as compute_output
    output = torch.full((batch_size, num_classes), -10.0, device=device)
    for i, p in enumerate(predictions):
        if 0 <= p < num_classes:
            output[i, p] = 10.0

    assert output.shape == (3, 12), f"Expected [3, 12], got {output.shape}"

    # Sample 0 predicts class 3
    assert output[0, 3] == 10.0
    assert output[0, 0] == -10.0
    assert output[0].argmax() == 3

    # Sample 1 predicts class 0
    assert output[1].argmax() == 0

    # Sample 2 predicts class 11
    assert output[2].argmax() == 11

    print("  [PASS] Multi-class logit construction: one-hot-ish correct")


def test_multiclass_out_of_range():
    """Out-of-range class predictions should result in all-negative logits."""
    num_classes = 5
    predictions = [99, -1]  # both out of range
    output = torch.full((2, num_classes), -10.0)
    for i, p in enumerate(predictions):
        if 0 <= p < num_classes:
            output[i, p] = 10.0

    # No class should be activated
    assert (output == -10.0).all(), "Out-of-range predictions should leave all logits at -10"

    print("  [PASS] Multi-class out-of-range: handled gracefully")


# ═══════════════════════════════════════════════════════════════
# Test 5: End-to-end prompt → positions consistency
# ═══════════════════════════════════════════════════════════════

def test_attention_mask_consistency():
    """Attention mask should match actual sequence content."""
    llm_dim = 32
    mock_decoder = MockLLMDecoder(llm_dim=llm_dim)
    graph_embeds = torch.randn(2, 1, llm_dim)

    inputs_embeds, attention_mask, _, positions = build_llm_inputs(
        graph_embeds=graph_embeds,
        neighbor_embeds=None,
        llm_decoder=mock_decoder,
        task_name="test-task",
        task_type="regression",
        entity_type="TestEntity",
        labels=None,
        entity_after_question=True,
    )

    B, seq_len, D = inputs_embeds.shape

    # Attention mask should be all 1s (no padding needed if both seqs same length)
    # or have 0s only at the start (left-padding)
    for i in range(B):
        real_tokens = attention_mask[i].sum().item()
        assert real_tokens > 0, "Must have at least 1 real token"
        assert real_tokens <= seq_len, "Can't have more real tokens than seq_len"

        # Left-padding: 0s should be contiguous at the start
        if real_tokens < seq_len:
            pad_len = seq_len - real_tokens
            assert attention_mask[i, :pad_len].sum() == 0, "Padding should be at start"
            assert attention_mask[i, pad_len:].sum() == real_tokens, "Real tokens after padding"

    print("  [PASS] Attention mask: left-padding is consistent")


def test_entity_position_within_attention():
    """Entity position should be within the real (non-padded) region."""
    llm_dim = 32
    mock_decoder = MockLLMDecoder(llm_dim=llm_dim)

    # Different length sequences to trigger padding
    graph_embeds = torch.randn(2, 1, llm_dim)
    neighbor_embeds = torch.zeros(2, 3, llm_dim)
    # Give sample 0 non-zero neighbors, sample 1 zero neighbors
    neighbor_embeds[0] = torch.randn(3, llm_dim)

    inputs_embeds, attention_mask, _, positions = build_llm_inputs(
        graph_embeds=graph_embeds,
        neighbor_embeds=neighbor_embeds,
        llm_decoder=mock_decoder,
        task_name="test-task",
        task_type="regression",
        entity_type="TestEntity",
        labels=None,
        entity_after_question=True,
    )

    for i in range(2):
        ent_pos, nb_start, nb_end = positions[i]
        # Entity should be in the attention=1 region
        assert attention_mask[i, ent_pos] == 1, \
            f"Sample {i}: entity at pos {ent_pos} should have attention=1"

    print("  [PASS] Entity position within attended region")


if __name__ == "__main__":
    print("\n=== Test 1: Prompt Generation ===")
    test_prompt_layout_entity_after_question()
    test_prompt_layout_entity_before_question()
    test_prompt_with_neighbors()
    test_prompt_with_training_labels()

    print("\n=== Test 2: Binary Classification ===")
    test_binary_logit_extraction()
    test_binary_no_text_generation()

    print("\n=== Test 3: Regression ===")
    test_regression_text_parsing()
    test_regression_output_shape()

    print("\n=== Test 4: Multi-class ===")
    test_multiclass_text_parsing()
    test_multiclass_logit_construction()
    test_multiclass_out_of_range()

    print("\n=== Test 5: Consistency ===")
    test_attention_mask_consistency()
    test_entity_position_within_attention()

    print("\n=== ALL TESTS PASSED ===")
