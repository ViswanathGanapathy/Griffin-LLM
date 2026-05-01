"""
hmaintask_combine_llm.py — Griffin Unified Training Pipeline

Supports FIVE prediction heads selectable via --head:

  Gradient-based heads (standard training loop):
    1. "default"  — Original Griffin heads (regression dec / classification @y.T)
    2. "llm"      — Projection → LLM text generation (autoregressive)
    3. "llm_mlp"  — Projection → LLM hidden states → MLP output head

  ICL-based heads (extract embeddings → fit ICL model):
    4. "tabpfn"   — Griffin embeddings → TabPFN v2 (in-context learning)
    5. "tabicl"   — Griffin embeddings → TabICL v2 (in-context learning)

Additional features beyond hmaintask_combine.py:
  - In-context learning (ICL) via --num_demo (LLM heads)
  - Neighbor embedding injection via --neighbor_tokens (LLM heads)
  - 2-phase training: projector warmup then full fine-tune via --warmup_epochs
  - Token-logit extraction for classification (faster than text parsing)
  - Rich task-specific prompts from task_prompts.py
  - TabPFN v2 fine-tuning via --tabpfn_finetune
  - Linear probe for Griffin fine-tuning with ICL heads

Usage examples:

  # Default Griffin heads (backward-compatible with hmaintask_combine.py):
  python hmaintask_combine_llm.py \\
      datasets/joint-v65 logs/default default-run \\
      --head default --tasks rel-f1-driver-position \\
      --loadpath checkpoints/pretrained --savepath checkpoints/default-ft

  # LLM text generation head:
  python hmaintask_combine_llm.py \\
      datasets/joint-v65 logs/llm llm-run \\
      --head llm --llm_model meta-llama/Llama-3.2-1B \\
      --tasks rel-f1-driver-position --num_demo 3 \\
      --loadpath checkpoints/pretrained --savepath checkpoints/llm-ft

  # LLM + MLP output head:
  python hmaintask_combine_llm.py \\
      datasets/joint-v65 logs/llm-mlp llm-mlp-run \\
      --head llm_mlp --llm_model meta-llama/Llama-3.2-1B \\
      --tasks rel-f1-driver-position --neighbor_tokens 5 \\
      --loadpath checkpoints/pretrained --savepath checkpoints/llm-mlp-ft

  # TabPFN v2 head (frozen Griffin, ICL prediction):
  python hmaintask_combine_llm.py \\
      datasets/joint-v65 logs/tabpfn tabpfn-run \\
      --head tabpfn --tasks rel-f1-driver-position \\
      --loadpath checkpoints/pretrained --freeze_griffin

  # TabPFN v2 with fine-tuning + Griffin linear probe:
  python hmaintask_combine_llm.py \\
      datasets/joint-v65 logs/tabpfn-ft tabpfn-ft-run \\
      --head tabpfn --tasks rel-f1-driver-position \\
      --loadpath checkpoints/pretrained --tabpfn_finetune \\
      --savepath checkpoints/tabpfn-ft

  # TabICL v2 head:
  python hmaintask_combine_llm.py \\
      datasets/joint-v65 logs/tabicl tabicl-run \\
      --head tabicl --tasks rel-f1-driver-position \\
      --loadpath checkpoints/pretrained --freeze_griffin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from hdataset import Graph, Task
from hloaderwrapper import LoaderWrapperTask, LoaderWrapperTaskLLM
from hmodel import GriffinMod
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from hFloatEmb import SimpleRepeater, getfloatdec
import numpy as np
import accelerate
import argparse
import os
import os.path as osp
from typing import Union, Optional
from metric import compute_metric
from task_prompts import get_task_description, get_task_question, get_task_reasoning, build_rich_system_prompt, audit_task_prompts
from tabular_heads import (
    TabPFNHead, TabICLHead, LinearProbe, ICLProjection,
    extract_embeddings, eval_with_icl_head,
)
import yaml


# ═══════════════════════════════════════════════════════════════════════════════
# Debug Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def debug_check_neighbors(
    neighbor_embeds: torch.Tensor,
    graph_embeds: torch.Tensor,
    taskname: str,
    mapping: torch.Tensor,
) -> None:
    """Print diagnostic info about neighbor embeddings when --debug is set.

    Checks:
      - neighbor_embeds shape is [B, K, llm_dim]
      - neighbor_embeds differ across samples when seeds differ
      - no NaN/Inf values
    """
    B = graph_embeds.shape[0]
    print(f"[DEBUG] Task: {taskname}")
    print(f"[DEBUG]   graph_embeds shape: {list(graph_embeds.shape)}")
    print(f"[DEBUG]   neighbor_embeds shape: {list(neighbor_embeds.shape)}")
    print(f"[DEBUG]   mapping: {mapping.tolist()[:8]}{'...' if len(mapping) > 8 else ''}")

    assert neighbor_embeds.dim() == 3, \
        f"neighbor_embeds should be 3D [B, K, D], got {neighbor_embeds.dim()}D"
    assert neighbor_embeds.shape[0] == B, \
        f"neighbor_embeds batch dim {neighbor_embeds.shape[0]} != graph_embeds batch {B}"
    assert neighbor_embeds.shape[2] == graph_embeds.shape[2], \
        f"neighbor_embeds dim {neighbor_embeds.shape[2]} != llm_dim {graph_embeds.shape[2]}"

    if torch.isnan(neighbor_embeds).any():
        print("[DEBUG]   WARNING: neighbor_embeds contains NaN!")
    if torch.isinf(neighbor_embeds).any():
        print("[DEBUG]   WARNING: neighbor_embeds contains Inf!")

    # Check diversity: if seeds differ, neighbors should differ
    if B >= 2 and mapping[0] != mapping[1]:
        same = torch.allclose(neighbor_embeds[0], neighbor_embeds[1], atol=1e-6)
        if same:
            print("[DEBUG]   WARNING: samples 0 and 1 have different seeds "
                  "but identical neighbor embeddings!")
        else:
            print("[DEBUG]   OK: samples 0 and 1 have distinct neighbor embeddings")

    # Count non-zero neighbor slots per sample
    zero_counts = []
    for i in range(min(B, 4)):
        norms = neighbor_embeds[i].norm(dim=-1)  # [K]
        n_zero = (norms < 1e-8).sum().item()
        zero_counts.append(n_zero)
    K = neighbor_embeds.shape[1]
    print(f"[DEBUG]   zero-padded slots (first {min(B,4)} samples): "
          f"{zero_counts} / {K}")


def focal_cross_entropy(logits: torch.Tensor, target: torch.Tensor,
                        gamma: float = 2.0, alpha: float = None) -> torch.Tensor:
    """Multi-class focal loss (Lin et al. 2017).

    loss_i = -alpha_i * (1 - p_i)^gamma * log(p_i)
    where p_i is the softmax probability of the true class for example i.

    When gamma=0 this reduces to standard cross-entropy.
    When alpha is None, no class re-weighting is applied (pure focal).
    """
    log_probs = F.log_softmax(logits, dim=-1)
    log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
    pt = log_pt.exp()
    focal_term = (1.0 - pt) ** gamma
    loss = -focal_term * log_pt
    if alpha is not None:
        # alpha is a tensor [num_classes]; index by target
        alpha_t = alpha[target]
        loss = alpha_t * loss
    return loss.mean()


# ═══════════════════════════════════════════════════════════════════════════════
# LLM Components
# ═══════════════════════════════════════════════════════════════════════════════

class GriffinToLLMProjector(nn.Module):
    """Projects Griffin MPNN embeddings → LLM embedding space.

    Uses a bottleneck architecture with GELU activation following
    Rel-LLM's MLP design for bridging GNN and LLM representation spaces.
    """

    def __init__(self, griffin_dim: int = 512, llm_dim: int = 2048,
                 bottleneck: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(griffin_dim, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, llm_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


class OutputMLP(nn.Module):
    """MLP head on LLM hidden states for direct prediction.

    Used with --head llm_mlp: extracts the last token's hidden state
    from the LLM and maps it to the target output dimension.

    When hidden_dim is None, uses a single Linear layer (minimal params,
    better for few-shot transfer).  When set, uses a 2-layer MLP with
    hidden_dim as intermediate size.

    pool_mode controls how many LLM tokens feed into the head:
      "last"    — last real token only (default, standard causal LM approach)
      "entity"  — entity embed token + last token, concatenated
      "graph"   — entity + all neighbor tokens + last token, mean-pooled
                  then concatenated with last token
    """

    def __init__(self, llm_dim: int, out_channels: int = 1,
                 hidden_dim: int = None, dropout: float = 0.1,
                 pool_mode: str = "last"):
        super().__init__()
        self.pool_mode = pool_mode

        if pool_mode == "last":
            in_dim = llm_dim
        elif pool_mode == "entity":
            in_dim = llm_dim * 2   # [entity_token ; last_token]
        elif pool_mode == "graph":
            in_dim = llm_dim * 2   # [mean(entity+neighbors) ; last_token]
        elif pool_mode == "attention":
            in_dim = llm_dim       # learned attention over all positions
        else:
            raise ValueError(f"Unknown pool_mode: {pool_mode}")

        if hidden_dim is None:
            self.head = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, out_channels),
            )
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_channels),
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """hidden_states: [B, in_dim] — already pooled by caller."""
        return self.head(hidden_states)


class TaskTypeHeads(nn.Module):
    """Three separate heads for regression, binary classification, and multi-class.

    Each task is routed to the appropriate head based on its output structure,
    not its identity. This enables transfer to unseen tasks of the same type.

    Routing:
      - regression (y is None)     → regression_head → [B, 1]
      - binary (num_classes == 2)  → binary_head     → [B, 2]
      - multi-class (num_classes > 2) → multiclass_head → [B, num_classes]

    The multi-class head has max_classes outputs. For tasks with fewer classes,
    the output is sliced to [:num_classes].
    """

    def __init__(self, llm_dim: int, max_classes: int = 12,
                 hidden_dim: int = None, dropout: float = 0.1,
                 pool_mode: str = "entity"):
        super().__init__()
        self.pool_mode = pool_mode
        self.max_classes = max_classes

        if pool_mode == "last" or pool_mode == "attention":
            in_dim = llm_dim
        elif pool_mode in ("entity", "graph"):
            in_dim = llm_dim * 2
        else:
            raise ValueError(f"Unknown pool_mode: {pool_mode}")

        def _make_head(out_dim):
            if hidden_dim is None:
                return nn.Sequential(
                    nn.LayerNorm(in_dim),
                    nn.Linear(in_dim, out_dim),
                )
            else:
                return nn.Sequential(
                    nn.LayerNorm(in_dim),
                    nn.Linear(in_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, out_dim),
                )

        self.regression_head = _make_head(1)
        self.binary_head = _make_head(2)
        self.multiclass_head = _make_head(max_classes)

    def forward(self, hidden_states: torch.Tensor,
                num_classes: int = None) -> torch.Tensor:
        """Route to appropriate head based on num_classes.

        Args:
            hidden_states: [B, in_dim] — already pooled by caller.
            num_classes: None for regression, 2 for binary, >2 for multi-class.

        Returns:
            [B, out_dim] where out_dim depends on task type.
        """
        # Compute all three heads so all parameters receive gradients
        # (required for DDP multi-GPU — unused params cause errors).
        # The dummy term adds zero-weighted outputs from unused heads,
        # ensuring gradient flow without affecting the actual prediction.
        reg_out = self.regression_head(hidden_states)
        bin_out = self.binary_head(hidden_states)
        multi_out = self.multiclass_head(hidden_states)
        dummy = 0.0 * (reg_out.sum() + bin_out.sum() + multi_out.sum())

        if num_classes is None:
            return reg_out + dummy
        elif num_classes == 2:
            return bin_out + dummy
        else:
            return multi_out[:, :num_classes] + dummy


class UnifiedDecoder(nn.Module):
    """Griffin-style unified task decoder for the LLM pipeline.

    Projects LLM pooled hidden states back to Griffin's embedding space,
    then uses Griffin's original decoders:
      - Regression: dec(back_projected) → scalar
      - Classification: back_projected @ y.T → [B, K] logits

    Classification requires no task-specific parameters — class embeddings
    y come from the graph (embeddings of the K candidate entities).
    Automatically handles any number of classes without retraining.

    Based on Section 3.3 of the Griffin paper (arxiv:2505.05568).
    """

    def __init__(self, pool_dim: int, griffin_dim: int = 512,
                 dropout: float = 0.1, pool_mode: str = "entity",
                 load_pretrained_dec: bool = True):
        super().__init__()
        self.pool_mode = pool_mode
        self.griffin_dim = griffin_dim

        # Back-projection: LLM pool space → Griffin embedding space
        self.back_proj = nn.Sequential(
            nn.LayerNorm(pool_dim),
            nn.Linear(pool_dim, griffin_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Regression decoder: same structure as Griffin's getfloatdec
        # Optionally load pretrained weights from floatdec-{dim}.pt
        self.reg_dec = nn.Sequential(
            nn.LayerNorm(griffin_dim, elementwise_affine=False),
            nn.Linear(griffin_dim, 1, bias=False),
        )
        if load_pretrained_dec:
            dec_path = f"floatdec-{griffin_dim}.pt"
            if os.path.exists(dec_path):
                self.reg_dec.load_state_dict(
                    torch.load(dec_path, map_location="cpu", weights_only=True),
                )
                print(f"[UnifiedDecoder] Loaded pretrained regression decoder from {dec_path}")
            else:
                print(f"[UnifiedDecoder] {dec_path} not found — using random init")

    def forward(self, hidden_states: torch.Tensor,
                y: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, pool_dim] — pooled LLM hidden states.
            y: [K, griffin_dim] class embeddings from graph, or None for regression.

        Returns:
            Regression (y is None): [B, 1] scalar predictions.
            Classification (y given): [B, K] logits via dot product.
        """
        # Project back to Griffin embedding space
        h = self.back_proj(hidden_states)  # [B, griffin_dim]

        if y is None:
            # Regression: learned linear decoder
            return self.reg_dec(h)  # [B, 1]
        else:
            # Classification: similarity with class embeddings
            # y: [K, griffin_dim] — embeddings of K candidate class entities
            return h @ y.T  # [B, K]


class LLMDecoder:
    """Wraps a HuggingFace causal LM + tokenizer.

    Handles model loading, optional freezing/LoRA, and embedding utilities.

    Tested with:
      - meta-llama/Llama-3.2-1B       (hidden=2048, no pad token)
      - google/gemma-3-270m            (hidden=640, has pad token)
      - Qwen/Qwen3-0.6B               (hidden=1024, has pad token, no greedy)

    Model-specific handling:
      - pad_token: auto-set to eos_token if missing (Llama)
      - LoRA target_modules: auto-detected per architecture
      - Generation: uses sampling for Qwen3 (greedy causes loops)
    """

    # LoRA target modules per model architecture.
    # All three use q_proj/v_proj but naming may vary in future models.
    LORA_TARGETS = {
        "llama":       ["q_proj", "v_proj"],
        "gemma3_text": ["q_proj", "v_proj"],
        "gemma3":      ["q_proj", "v_proj"],
        "gemma4":      ["q_proj", "v_proj"],
        "qwen3":       ["q_proj", "v_proj"],
        # Fallback for unknown architectures
        "default":     ["q_proj", "v_proj"],
    }

    def __init__(self, model_name: str, frozen: bool = True,
                 use_lora: bool = False, lora_r: int = 8,
                 lora_alpha: int = 16, device: str = "cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[LLM] Loading {model_name}...")
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=False, padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"[LLM] No pad_token found — using eos_token: "
                  f"'{self.tokenizer.eos_token}'")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        ).to(device)

        # hidden_size may be at top level or under text_config (multimodal models like Gemma-4)
        cfg = self.model.config
        self.llm_dim = getattr(cfg, "hidden_size", None)
        if self.llm_dim is None and hasattr(cfg, "text_config"):
            self.llm_dim = cfg.text_config.hidden_size
        if self.llm_dim is None:
            raise ValueError(f"Cannot determine hidden_size from {model_name} config")
        self.model_type = getattr(cfg, "model_type", "unknown")
        self.word_embedding = self.model.get_input_embeddings()
        self.device = device

        print(f"[LLM] model_type={self.model_type}, hidden_size={self.llm_dim}")

        if frozen and not use_lora:
            print("[LLM] Freezing all LLM weights")
            for p in self.model.parameters():
                p.requires_grad = False
        elif use_lora:
            print("[LLM] Applying LoRA")
            from peft import LoraConfig, get_peft_model
            # Auto-detect LoRA target modules for this architecture
            target_modules = self.LORA_TARGETS.get(
                self.model_type, self.LORA_TARGETS["default"],
            )
            print(f"[LLM] LoRA target_modules for {self.model_type}: {target_modules}")
            config = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
                target_modules=target_modules,
                bias="none", task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, config)
            self.model.print_trainable_parameters()

        # Cache pad embedding
        with torch.no_grad():
            pad_id = torch.tensor([self.tokenizer.pad_token_id], device=device)
            self.pad_embed = self.word_embedding(pad_id)  # [1, llm_dim]

        # Cache Yes/No token IDs for classification logit extraction
        self.true_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
        self.false_id = self.tokenizer.encode("No", add_special_tokens=False)[0]

        # Qwen3 produces degenerate output with greedy decoding.
        # Flag it so generate() uses sampling instead.
        self.needs_sampling = "qwen" in self.model_type.lower()
        if self.needs_sampling:
            print("[LLM] Qwen model detected — will use sampling instead of greedy")

    def tokenize(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text, add_special_tokens=False, return_tensors="pt",
        ).input_ids[0].to(self.device)

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.word_embedding(token_ids)


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt Construction
# ═══════════════════════════════════════════════════════════════════════════════

def build_llm_inputs(
    graph_embeds: torch.Tensor,
    neighbor_embeds: Optional[torch.Tensor],
    llm_decoder: LLMDecoder,
    task_name: str,
    task_type: str,
    entity_type: str,
    labels: Optional[torch.Tensor] = None,
    demo_embeds: Optional[torch.Tensor] = None,
    demo_labels: Optional[torch.Tensor] = None,
    metanode: Optional[dict] = None,
    metaadj: Optional[dict] = None,
    metatask: Optional[dict] = None,
    feature_names: Optional[list[str]] = None,
    neighbor_entity_types: Optional[list[str]] = None,
    entity_after_question: bool = True,
    cot_prompt: bool = False,
):
    """Build input embeddings for the LLM.

    Prompt layout per sample:
      [system_text] [ENTITY_EMBED] [NEIGHBOR_EMBED_1..K]
      [DEMO_EMBED_1 → label_1] ... [DEMO_EMBED_D → label_D]  (if ICL)
      [question_text]
      [answer_text + EOS]  (training only)

    When metanode/metaadj/metatask are provided, the system prompt is enriched
    with relational schema metadata (entity types, features, relationships)
    following the Rel-LLM approach.  This is critical for generalization to
    unseen datasets.

    Args:
        graph_embeds:    [B, 1, llm_dim] — projected seed node embeddings
        neighbor_embeds: [B, K, llm_dim] or None — projected neighbor embeddings
        llm_decoder:     LLMDecoder instance
        task_name:       e.g. "rel-f1-driver-position"
        task_type:       "regression" or "retrieval"
        entity_type:     e.g. "driver" — for prompt context
        labels:          ground truth (Tensor) — None for inference
        demo_embeds:     [B, D, llm_dim] or None — projected demo embeddings
        demo_labels:     [B, D] or None — demo labels
        metanode:        Graph.metanode dict (entity schema metadata)
        metaadj:         Adjacency metadata dict (relationship info)
        metatask:        Task.metatask dict (task metadata)
        feature_names:   Visible feature names for the root entity
        neighbor_entity_types: Entity types of neighbors in the subgraph

    Returns:
        inputs_embeds:  [B, max_seq_len, llm_dim]
        attention_mask: [B, max_seq_len]
        label_ids:      [B, max_seq_len] (-100 for non-answer tokens)
    """
    ld = llm_decoder
    device = graph_embeds.device
    batch_size = graph_embeds.shape[0]
    is_training = labels is not None

    # Build system prompt — use rich metadata version when available
    if metanode is not None and metatask is not None:
        system_text = build_rich_system_prompt(
            task_name=task_name,
            task_type=task_type,
            root_entity=entity_type,
            metanode=metanode,
            metaadj=metaadj if metaadj is not None else {},
            metatask=metatask,
            feature_names=feature_names,
            neighbor_entity_types=neighbor_entity_types,
        )
    else:
        # Fallback: generic prompt (backward compatibility)
        description = get_task_description(task_name)
        entity_clean = entity_type.replace("_", " ").replace("-", " ")
        system_text = (
            f"You are a predictive model analyzing relational database records.\n"
            f"Task: {description}\n"
            f"Entity type: {entity_clean}\n"
            f"Database record embeddings follow.\n"
        )

    question = get_task_question(task_name, task_type)
    if cot_prompt:
        reasoning = get_task_reasoning(task_name)
        if reasoning:
            question_text = (
                f"\nQuestion: {question}\n"
                f"Let's reason step by step.\n{reasoning}\nAnswer:"
            )
        else:
            # Generic CoT scaffolding for tasks without curated hints
            question_text = (
                f"\nQuestion: {question}\n"
                f"Let's reason step by step. Consider the entity's features, "
                f"its relational neighbors, and any temporal patterns.\nAnswer:"
            )
    else:
        question_text = f"\nQuestion: {question}\nAnswer:"

    # Tokenize + embed fixed text
    sys_ids = ld.tokenize(system_text)
    sys_emb = ld.embed_tokens(sys_ids)          # [S, llm_dim]
    q_ids = ld.tokenize(question_text)
    q_emb = ld.embed_tokens(q_ids)              # [Q, llm_dim]

    # ICL: tokenize demo separator/label format
    demo_arrow_ids = ld.tokenize(" -> ")
    demo_arrow_emb = ld.embed_tokens(demo_arrow_ids)  # [A, llm_dim]
    demo_sep_ids = ld.tokenize("\n")
    demo_sep_emb = ld.embed_tokens(demo_sep_ids)      # [1, llm_dim]

    all_embeds = []
    all_labels = []
    all_graph_token_positions = []  # (entity_pos, neighbor_start, neighbor_end) per sample
    max_len = 0

    for i in range(batch_size):
        parts = [sys_emb]  # system description
        label_parts = [
            torch.full((sys_emb.size(0),), -100, dtype=torch.long, device=device),
        ]
        cur_pos = sys_emb.size(0)

        # Helper: append ICL demo examples
        def _append_demos():
            nonlocal cur_pos
            if demo_embeds is not None and demo_labels is not None:
                num_demo = demo_embeds.shape[1]
                for d in range(num_demo):
                    parts.append(demo_embeds[i, d].unsqueeze(0))
                    parts.append(demo_arrow_emb)
                    if task_type == "regression":
                        dl_text = f"{demo_labels[i, d].item():.4f}"
                    else:
                        dl_text = f"{demo_labels[i, d].item()}"
                    dl_ids = ld.tokenize(dl_text)
                    dl_emb = ld.embed_tokens(dl_ids)
                    parts.extend([dl_emb, demo_sep_emb])
                    demo_tok_len = 1 + demo_arrow_emb.size(0) + dl_emb.size(0) + demo_sep_emb.size(0)
                    label_parts.append(
                        torch.full((demo_tok_len,), -100, dtype=torch.long, device=device),
                    )
                    cur_pos += demo_tok_len

        # Helper: append question
        def _append_question():
            nonlocal cur_pos
            parts.append(q_emb)
            label_parts.append(
                torch.full((q_emb.size(0),), -100, dtype=torch.long, device=device),
            )
            cur_pos += q_emb.size(0)

        # Helper: append entity + neighbor embeddings
        def _append_graph_tokens():
            nonlocal cur_pos
            nonlocal entity_pos, neighbor_start, neighbor_end
            entity_pos = cur_pos
            parts.append(graph_embeds[i])  # [1, llm_dim]
            label_parts.append(
                torch.full((1,), -100, dtype=torch.long, device=device),
            )
            cur_pos += 1
            neighbor_start = cur_pos
            if neighbor_embeds is not None and neighbor_embeds.shape[1] > 0:
                n_emb = neighbor_embeds[i]  # [K, llm_dim]
                parts.append(n_emb)
                label_parts.append(
                    torch.full((n_emb.size(0),), -100, dtype=torch.long, device=device),
                )
                cur_pos += n_emb.size(0)
            neighbor_end = cur_pos

        entity_pos = neighbor_start = neighbor_end = 0

        if entity_after_question:
            # New order: [system] [demos] [question] [ENTITY] [neighbors]
            _append_demos()
            _append_question()
            _append_graph_tokens()
        else:
            # Old order: [system] [ENTITY] [neighbors] [demos] [question]
            _append_graph_tokens()
            _append_demos()
            _append_question()

        all_graph_token_positions.append((entity_pos, neighbor_start, neighbor_end))

        if is_training:
            # Answer text (supervised)
            if task_type == "regression":
                ans_text = f" {labels[i].item():.4f}"
            else:
                ans_text = f" {labels[i].item()}"
            ans_ids = ld.tokenize(ans_text)
            ans_emb = ld.embed_tokens(ans_ids)
            eos_id = torch.tensor([ld.tokenizer.eos_token_id], device=device)
            eos_emb = ld.embed_tokens(eos_id)

            parts.extend([ans_emb, eos_emb])
            label_parts.extend([ans_ids, eos_id])

        seq_emb = torch.cat(parts, dim=0)
        seq_lab = torch.cat(label_parts, dim=0)

        all_embeds.append(seq_emb)
        all_labels.append(seq_lab)
        max_len = max(max_len, seq_emb.size(0))

    # Left-pad to max_len
    padded_embeds = []
    padded_labels = []
    attention_masks = []

    for i in range(batch_size):
        pad_len = max_len - all_embeds[i].size(0)
        real_len = all_embeds[i].size(0)

        if pad_len > 0:
            pad_e = ld.pad_embed.expand(pad_len, -1)
            padded_embeds.append(torch.cat([pad_e, all_embeds[i]], dim=0))
            padded_labels.append(torch.cat([
                torch.full((pad_len,), -100, dtype=torch.long, device=device),
                all_labels[i],
            ]))
            attention_masks.append(torch.cat([
                torch.zeros(pad_len, dtype=torch.long, device=device),
                torch.ones(real_len, dtype=torch.long, device=device),
            ]))
        else:
            padded_embeds.append(all_embeds[i])
            padded_labels.append(all_labels[i])
            attention_masks.append(
                torch.ones(real_len, dtype=torch.long, device=device),
            )

    # Adjust graph token positions for left-padding offset
    adjusted_positions = []
    for i in range(batch_size):
        pad_len = max_len - all_embeds[i].size(0)
        ent_pos, nb_start, nb_end = all_graph_token_positions[i]
        adjusted_positions.append((
            ent_pos + pad_len,
            nb_start + pad_len,
            nb_end + pad_len,
        ))

    return (
        torch.stack(padded_embeds),     # [B, max_len, llm_dim]
        torch.stack(attention_masks),   # [B, max_len]
        torch.stack(padded_labels),     # [B, max_len]
        adjusted_positions,             # [(entity_pos, nb_start, nb_end)] per sample
    )


class LayerPooling(nn.Module):
    """Learnable weighted combination of the last K transformer layers.

    When pool_layers=1, returns the last layer unchanged (zero overhead).
    When pool_layers=K>1, learns K scalar weights via softmax and returns
    the weighted sum of the last K hidden states.

    Args:
        pool_layers: Number of layers to pool (1 = last layer only).
    """

    def __init__(self, pool_layers: int = 1):
        super().__init__()
        self.pool_layers = pool_layers
        if pool_layers > 1:
            self.layer_weights = nn.Parameter(torch.ones(pool_layers) / pool_layers)

    def forward(self, hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Args:
            hidden_states: tuple of [B, seq_len, D] tensors from each layer.
                           Length = num_hidden_layers + 1 (embedding layer + N).
        Returns:
            [B, seq_len, D] — pooled hidden states.
        """
        if self.pool_layers == 1:
            return hidden_states[-1]
        weights = F.softmax(self.layer_weights, dim=0)
        pooled = sum(
            w * hidden_states[-(i + 1)]
            for i, w in enumerate(weights)
        )
        return pooled


class AttentionPool(nn.Module):
    """Learned attention pooling over sequence positions.

    Computes a weighted sum over all token positions using a learned
    query vector. This lets the model learn which positions (entity,
    neighbors, text tokens) carry the most predictive signal.

    Args:
        dim: Hidden dimension of the LLM (e.g. 2048).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim) * 0.01)
        self.scale = dim ** -0.5

    def forward(self, hidden: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden:          [B, seq_len, D]
            attention_mask:  [B, seq_len]
        Returns:
            [B, D] — attention-weighted pooled representation.
        """
        scores = (hidden @ self.query) * self.scale  # [B, seq_len]
        scores = scores.masked_fill(~attention_mask.bool(), -1e9)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)  # [B, seq_len, 1]
        return (hidden * weights).sum(dim=1)  # [B, D]


def _pool_llm_hidden(last_hidden, attention_mask, graph_positions, pool_mode,
                     attention_pool=None):
    """Extract and pool LLM hidden states based on pool_mode.

    Args:
        last_hidden:     [B, seq_len, llm_dim] — pooled hidden states
        attention_mask:  [B, seq_len]
        graph_positions: [(entity_pos, nb_start, nb_end)] per sample
        pool_mode:       "last" | "entity" | "graph" | "attention"
        attention_pool:  AttentionPool module (required for pool_mode="attention")

    Returns:
        pooled: [B, in_dim] where in_dim depends on pool_mode
    """
    B = last_hidden.size(0)
    device = last_hidden.device

    # Last real token per sample (always needed)
    seq_lens = attention_mask.sum(dim=1) - 1  # [B]
    last_token = last_hidden[torch.arange(B, device=device), seq_lens]  # [B, D]

    if pool_mode == "last":
        return last_token

    elif pool_mode == "entity":
        # [entity_hidden ; last_hidden]
        entity_tokens = torch.stack([
            last_hidden[i, graph_positions[i][0]]
            for i in range(B)
        ])  # [B, D]
        return torch.cat([entity_tokens, last_token], dim=-1)  # [B, 2D]

    elif pool_mode == "graph":
        # mean(entity + all neighbor hidden states) ; last_hidden
        graph_pooled = []
        for i in range(B):
            ent_pos, nb_start, nb_end = graph_positions[i]
            # Gather entity token + neighbor tokens
            indices = [ent_pos] + list(range(nb_start, nb_end))
            tokens = last_hidden[i, indices]  # [1+K, D]
            graph_pooled.append(tokens.mean(dim=0))  # [D]
        graph_pooled = torch.stack(graph_pooled)  # [B, D]
        return torch.cat([graph_pooled, last_token], dim=-1)  # [B, 2D]

    elif pool_mode == "attention":
        # Learned attention over all positions → [B, D]
        return attention_pool(last_hidden, attention_mask)

    raise ValueError(f"Unknown pool_mode: {pool_mode}")


# ═══════════════════════════════════════════════════════════════════════════════
# Loss and Output — supports all three head modes
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_llm_components(model, projector, data, args):
    """Shared logic: run Griffin MPNN → project seed + neighbor embeddings.

    Returns:
        seed_emb:       [B, griffin_dim]
        graph_embeds:   [B, 1, llm_dim]
        neighbor_embeds:[B, K, llm_dim] or None
        label, y, mapping, taskname, rootnodetype,
        feature_names, neighbor_entity_types
    """
    # Unpack LLM-extended data tuple (now includes metadata)
    (node, mask, taskfeat, edge_index, edge_attr_type, edge_attr,
     label, y, mapping,
     taskname, rootnodetype, neighbor_mask,
     feature_names, neighbor_entity_types) = data

    all_embs = model(node, mask, taskfeat, edge_index, edge_attr_type, edge_attr)
    seed_emb = all_embs[mapping]  # [B, griffin_dim]

    graph_embeds = projector(seed_emb.float()).unsqueeze(1)  # [B, 1, llm_dim]

    # Neighbor embeddings — per-sample assignment using edge_index
    neighbor_embeds = None
    if args.neighbor_tokens > 0 and edge_index is not None and edge_index.shape[1] > 0:
        B = seed_emb.shape[0]
        K = args.neighbor_tokens
        seed_set = set(mapping.tolist())
        llm_dim = graph_embeds.shape[-1]

        # Build per-seed neighbor index lists from edge_index
        # edges where source is a seed → target is its 1-hop neighbor
        per_seed_neighbors = {s.item(): [] for s in mapping}
        for e in range(edge_index.shape[1]):
            src = edge_index[0, e].item()
            dst = edge_index[1, e].item()
            if src in per_seed_neighbors and dst not in seed_set:
                per_seed_neighbors[src].append(dst)

        # Project all neighbor nodes at once (deduplicated)
        all_nb_indices = list({idx for nbs in per_seed_neighbors.values() for idx in nbs})
        if all_nb_indices:
            nb_idx_tensor = torch.tensor(all_nb_indices, device=all_embs.device)
            nb_projected_all = projector(all_embs[nb_idx_tensor].float())  # [N_unique, llm_dim]
            nb_idx_to_pos = {idx: pos for pos, idx in enumerate(all_nb_indices)}

            # Build [B, K, llm_dim] with per-sample neighbors, zero-padded
            pad_embed = torch.zeros(llm_dim, device=all_embs.device, dtype=nb_projected_all.dtype)
            per_sample = []
            for i in range(B):
                seed_idx = mapping[i].item()
                nbs = per_seed_neighbors.get(seed_idx, [])
                # Take up to K unique neighbors
                nb_positions = [nb_idx_to_pos[n] for n in nbs[:K]]
                if nb_positions:
                    sample_nb = nb_projected_all[nb_positions]  # [<=K, llm_dim]
                    if sample_nb.shape[0] < K:
                        pad = pad_embed.unsqueeze(0).expand(K - sample_nb.shape[0], -1)
                        sample_nb = torch.cat([sample_nb, pad], dim=0)
                else:
                    sample_nb = pad_embed.unsqueeze(0).expand(K, -1)
                per_sample.append(sample_nb)
            neighbor_embeds = torch.stack(per_sample)  # [B, K, llm_dim]

    # Debug diagnostics
    if getattr(args, "debug", False) and neighbor_embeds is not None:
        debug_check_neighbors(
            neighbor_embeds=neighbor_embeds,
            graph_embeds=graph_embeds,
            taskname=taskname,
            mapping=mapping,
        )

    return (seed_emb, graph_embeds, neighbor_embeds,
            label, y, mapping, taskname, rootnodetype,
            feature_names, neighbor_entity_types)


def compute_loss(model, dec, data, args,
                 projector=None, llm_decoder=None, output_mlp=None,
                 task_type_dict=None, linear_probe=None,
                 icl_projection=None,
                 metanode=None, metaadj=None, metatask=None,
                 layer_pooling=None, attention_pool=None):
    """Unified loss function supporting all five head modes.

    --head default:       Uses dec (getfloatdec) for regression, @y.T for classification.
    --head llm:           Projection → LLM → language modeling loss on answer tokens.
    --head llm_mlp:       Projection → LLM hidden states → MLP → MSE/CE loss.
    --head tabpfn/tabicl: Linear probe loss (proxy for Griffin fine-tuning).
    """
    if args.head in ("default", "tabpfn", "tabicl"):
        # Original Griffin path — data comes from LoaderWrapperTask
        label, y, mapping = data[-3:]
        model_data = data[:-3]
        if args.head in ("tabpfn", "tabicl") and linear_probe is not None:
            # Griffin → ICLProjection → LinearProbe → loss
            # Gradients flow through projection (and Griffin if not frozen)
            seed_emb = model(*model_data)[mapping]
            projected = icl_projection(seed_emb) if icl_projection is not None else seed_emb
            output = linear_probe(projected)
            if y is None:
                loss = F.mse_loss(output.flatten(), label.float().flatten())
            else:
                loss = F.cross_entropy(output, label)
        elif y is None:
            output = dec(model(*model_data)[mapping])
            loss = F.mse_loss(output.flatten(), label.flatten())
        else:
            output = model(*model_data)[mapping] @ y.T
            loss = F.cross_entropy(output, label)
        return loss

    # LLM paths — data comes from LoaderWrapperTaskLLM
    (seed_emb, graph_embeds, neighbor_embeds,
     label, y, mapping, taskname, rootnodetype,
     feature_names, neighbor_entity_types) = _extract_llm_components(
        model, projector, data, args,
    )

    task_type = task_type_dict[taskname]

    # Common metadata kwargs for rich prompt construction
    _meta_kwargs = dict(
        metanode=metanode, metaadj=metaadj, metatask=metatask,
        feature_names=feature_names,
        neighbor_entity_types=neighbor_entity_types,
        entity_after_question=getattr(args, "entity_after_question", True),
        cot_prompt=getattr(args, "cot_prompt", False),
    )

    if args.head == "llm":
        # Build prompt with answer → LM loss
        inputs_embeds, attention_mask, labels_ids, _ = build_llm_inputs(
            graph_embeds, neighbor_embeds, llm_decoder,
            task_name=taskname, task_type=task_type,
            entity_type=rootnodetype, labels=label,
            **_meta_kwargs,
        )
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = llm_decoder.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels_ids,
                return_dict=True,
            )
        return outputs.loss

    elif args.head == "llm_mlp":
        # Build prompt (no answer) → extract hidden states → MLP
        inputs_embeds, attention_mask, _, graph_positions = build_llm_inputs(
            graph_embeds, neighbor_embeds, llm_decoder,
            task_name=taskname, task_type=task_type,
            entity_type=rootnodetype, labels=None,
            **_meta_kwargs,
        )
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = llm_decoder.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )
        # Multi-layer pooling: combine last K layers with learned weights
        _lp = layer_pooling if layer_pooling is not None else LayerPooling(1)
        last_hidden = _lp(outputs.hidden_states)
        # Select OutputMLP: TaskTypeHeads, shared head, or per-task ModuleDict
        # Unwrap DDP wrapper if present (multi-GPU)
        _mlp = getattr(output_mlp, 'module', output_mlp)
        task_mlp = _mlp[taskname] if isinstance(_mlp, nn.ModuleDict) else output_mlp
        _task_mlp_unwrapped = getattr(task_mlp, 'module', task_mlp)
        pooled = _pool_llm_hidden(
            last_hidden, attention_mask, graph_positions,
            _task_mlp_unwrapped.pool_mode, attention_pool=attention_pool,
        )

        num_classes = y.shape[0] if y is not None else None
        if isinstance(_task_mlp_unwrapped, UnifiedDecoder):
            pred = task_mlp(pooled.float(), y=y)
        elif isinstance(_task_mlp_unwrapped, TaskTypeHeads):
            pred = task_mlp(pooled.float(), num_classes=num_classes)
        else:
            pred = task_mlp(pooled.float())

        # Loss computation
        if num_classes is None:
            loss = F.mse_loss(pred[:, 0], label.float())
        else:
            if getattr(args, "focal_loss", False) and num_classes == 2:
                loss = focal_cross_entropy(
                    pred[:, :num_classes], label,
                    gamma=getattr(args, "focal_gamma", 2.0),
                )
            else:
                loss = F.cross_entropy(pred[:, :num_classes], label)
        return loss

    raise ValueError(f"Unknown head: {args.head}")


def compute_output(model, dec, data, args,
                   projector=None, llm_decoder=None, output_mlp=None,
                   task_type_dict=None, linear_probe=None,
                   metanode=None, metaadj=None, metatask=None,
                   layer_pooling=None, attention_pool=None):
    """Unified output function for evaluation.

    Returns (output, label) matching the format expected by eval_task.
    Note: tabpfn/tabicl heads bypass this function — they use
    extract_embeddings + eval_with_icl_head instead.
    """
    if args.head in ("default", "tabpfn", "tabicl"):
        label, y, mapping = data[-3:]
        model_data = data[:-3]
        if y is None:
            output = dec(model(*model_data)[mapping])
        else:
            output = model(*model_data)[mapping] @ y.T
        return output, label

    # LLM paths
    (seed_emb, graph_embeds, neighbor_embeds,
     label, y, mapping, taskname, rootnodetype,
     feature_names, neighbor_entity_types) = _extract_llm_components(
        model, projector, data, args,
    )

    task_type = task_type_dict[taskname]
    batch_size = seed_emb.shape[0]
    device = seed_emb.device

    # Common metadata kwargs for rich prompt construction
    _meta_kwargs = dict(
        metanode=metanode, metaadj=metaadj, metatask=metatask,
        feature_names=feature_names,
        neighbor_entity_types=neighbor_entity_types,
        entity_after_question=getattr(args, "entity_after_question", True),
        cot_prompt=getattr(args, "cot_prompt", False),
    )

    if args.head == "llm_mlp":
        inputs_embeds, attention_mask, _, graph_positions = build_llm_inputs(
            graph_embeds, neighbor_embeds, llm_decoder,
            task_name=taskname, task_type=task_type,
            entity_type=rootnodetype, labels=None,
            **_meta_kwargs,
        )
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = llm_decoder.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )
        _lp = layer_pooling if layer_pooling is not None else LayerPooling(1)
        last_hidden = _lp(outputs.hidden_states)
        # Select OutputMLP: TaskTypeHeads, shared head, or per-task ModuleDict
        # Unwrap DDP wrapper if present (multi-GPU)
        _mlp = getattr(output_mlp, 'module', output_mlp)
        task_mlp = _mlp[taskname] if isinstance(_mlp, nn.ModuleDict) else output_mlp
        _task_mlp_unwrapped = getattr(task_mlp, 'module', task_mlp)
        pooled = _pool_llm_hidden(
            last_hidden, attention_mask, graph_positions,
            _task_mlp_unwrapped.pool_mode, attention_pool=attention_pool,
        )

        num_classes = y.shape[0] if y is not None else None
        if isinstance(_task_mlp_unwrapped, UnifiedDecoder):
            pred = task_mlp(pooled.float(), y=y)
        elif isinstance(_task_mlp_unwrapped, TaskTypeHeads):
            pred = task_mlp(pooled.float(), num_classes=num_classes)
        else:
            pred = task_mlp(pooled.float())

        if num_classes is None:
            return pred[:, :1], label  # [B, 1], [B]
        else:
            return pred[:, :num_classes], label  # [B, num_classes], [B]

    elif args.head == "llm":
        inputs_embeds, attention_mask, _, _gp = build_llm_inputs(
            graph_embeds, neighbor_embeds, llm_decoder,
            task_name=taskname, task_type=task_type,
            entity_type=rootnodetype, labels=None,
            **_meta_kwargs,
        )

        # For binary classification: extract Yes/No logits directly
        if task_type == "retrieval" and y is not None and y.shape[0] == 2:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = llm_decoder.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
            # Logits for the next token after the prompt
            next_logits = outputs.logits[:, -1, :]  # [B, vocab]
            binary_logits = next_logits[:, [llm_decoder.false_id, llm_decoder.true_id]]
            return binary_logits, label  # [B, 2], [B]

        # General case: generate text
        # Qwen3 produces degenerate output with greedy decoding —
        # use sampling with temperature for those models.
        gen_kwargs = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=llm_decoder.tokenizer.pad_token_id,
        )
        if llm_decoder.needs_sampling:
            gen_kwargs.update(do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
        else:
            gen_kwargs.update(do_sample=False)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            generated_ids = llm_decoder.model.generate(**gen_kwargs)

        if task_type == "regression" or y is None:
            predictions = []
            for i in range(batch_size):
                text = llm_decoder.tokenizer.decode(
                    generated_ids[i], skip_special_tokens=True,
                )
                try:
                    tokens = text.strip().split()
                    val = float(tokens[-1]) if tokens else 0.0
                except (ValueError, IndexError):
                    val = 0.0
                predictions.append(val)
            output = torch.tensor(
                predictions, device=device, dtype=torch.float,
            ).unsqueeze(1)
        else:
            num_classes = y.shape[0]
            predictions = []
            for i in range(batch_size):
                text = llm_decoder.tokenizer.decode(
                    generated_ids[i], skip_special_tokens=True,
                )
                try:
                    tokens = text.strip().split()
                    val = int(tokens[-1]) if tokens else 0
                except (ValueError, IndexError):
                    val = 0
                predictions.append(val)
            output = torch.full(
                (batch_size, num_classes), -10.0, device=device,
            )
            for i, p in enumerate(predictions):
                if 0 <= p < num_classes:
                    output[i, p] = 10.0

        return output, label

    raise ValueError(f"Unknown head: {args.head}")


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def eval_task(model, dec, dataset, args, accelerator, metric,
              projector=None, llm_decoder=None, output_mlp=None,
              task_type_dict=None,
              metanode=None, metaadj=None, metatask=None,
              layer_pooling=None, attention_pool=None):
    model.eval()
    if projector is not None:
        projector.eval()
    dataset.rebuild_indice(accelerator)
    batchsize: int = dataset.batch_size
    loader = DataLoader(
        dataset, shuffle=False, batch_size=1,
        collate_fn=lambda xlist: xlist[0],
        num_workers=4, persistent_workers=False,
    )
    loader = accelerator.prepare(loader)
    outputs = []
    labels = []
    with torch.no_grad():
        for data in loader:
            output, label = compute_output(
                model, dec, data, args,
                projector=projector, llm_decoder=llm_decoder,
                output_mlp=output_mlp, task_type_dict=task_type_dict,
                metanode=metanode, metaadj=metaadj, metatask=metatask,
                layer_pooling=layer_pooling, attention_pool=attention_pool,
            )
            if output.shape[0] < batchsize:
                assert output.ndim == 2
                assert label.ndim == 1
                padnum = batchsize - output.shape[0]
                if torch.is_floating_point(label):
                    label = torch.concat((
                        label,
                        torch.empty_like(label[[0]].expand(padnum)).fill_(torch.nan),
                    ), dim=0)
                else:
                    label = torch.concat((
                        label,
                        torch.empty_like(label[[0]].expand(padnum)).fill_(-1),
                    ), dim=0)
                output = torch.concat((
                    output,
                    torch.empty_like(output[[0]].expand(padnum, -1)).fill_(torch.nan),
                ), dim=0)
            output, label = output.unsqueeze(0), label.unsqueeze(0)
            output, label = accelerator.gather_for_metrics((output, label))
            output, label = output.flatten(0, 1), label.flatten(0, 1)
            if accelerator.is_main_process:
                if torch.is_floating_point(label):
                    mask = torch.isnan(label).logical_not_()
                else:
                    mask = label >= 0
                output, label = output[mask], label[mask]
                outputs.append(output.cpu())
                labels.append(label.cpu())
    if accelerator.is_main_process:
        labels = torch.concat(labels, dim=0)
        outputs = torch.concat(outputs, dim=0)
        return compute_metric(outputs, labels, metric)
    else:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset Construction
# ═══════════════════════════════════════════════════════════════════════════════

def construct_dataset(graph, task, tasknames, split, args, floatembmodel):
    # ICL heads (tabpfn/tabicl) use the standard LoaderWrapperTask since
    # they only need Griffin embeddings, not LLM metadata.
    if args.head in ("llm", "llm_mlp"):
        LoaderClass = LoaderWrapperTaskLLM
    else:
        LoaderClass = LoaderWrapperTask
    return LoaderClass(
        graph,
        batch_size=args.batchsize,
        subgraphargs={
            "floatemb": floatembmodel,
            "fanout": args.fanout,
            "hop": args.hop,
        },
        shuffle=True if split == "train" else False,
        task=task,
        tasknames=tasknames,
        split=split,
        fewshotfanout=args.fewshotfanout,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ICL Head Evaluation (TabPFN / TabICL)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_icl_evaluation(model, train_dataset, valid_dataset_dict,
                        test_dataset_dict, tasknames, task_type_dict,
                        metric_dict, args, accelerator, tbtracker, step=0,
                        icl_projection=None,
                        graph=None, task_obj=None):
    """Extract Griffin embeddings, project, and evaluate with TabPFN or TabICL.

    Pipeline:
      1. Extract embeddings from Griffin for train/valid/test splits
      2. Project through ICLProjection (trainable, compresses 512→icl_proj_dim)
      3. Create ICL head (TabPFN or TabICL) per task
      4. Fit on projected train embeddings (subsampled to context limit)
      5. Predict on projected test embeddings, report metrics
    """
    for tn in tasknames:
        task_type = task_type_dict[tn]
        metric_name = metric_dict[tn]

        if accelerator.is_main_process:
            print(f"\n[ICL] Processing task: {tn} (type={task_type})")

        # Extract embeddings for each split
        # Build a train-split dataset for this specific task
        # Reuse already-loaded graph/task objects to avoid disk I/O
        _graph = graph if graph is not None else Graph(args.dataset)
        _task = task_obj if task_obj is not None else Task(args.dataset)
        floatembmodel = SimpleRepeater(args.hiddim)
        train_ds_task = construct_dataset(
            _graph, _task,
            [tn], "train", args, floatembmodel,
        )

        train_embs, train_labels, train_y, _ = extract_embeddings(
            model, train_ds_task, accelerator, icl_projection=icl_projection,
        )
        valid_embs, valid_labels, _, _ = extract_embeddings(
            model, valid_dataset_dict[tn], accelerator, icl_projection=icl_projection,
        )
        test_embs, test_labels, _, _ = extract_embeddings(
            model, test_dataset_dict[tn], accelerator, icl_projection=icl_projection,
        )

        if not accelerator.is_main_process:
            continue

        print(f"  Train: {train_embs.shape}, Valid: {valid_embs.shape}, "
              f"Test: {test_embs.shape}")

        # Create ICL head
        if args.head == "tabpfn":
            icl_head = TabPFNHead(
                task_type=task_type,
                device=str(accelerator.device),
                n_estimators=args.icl_n_estimators,
                max_context_size=args.icl_max_context,
                model_version=args.tabpfn_version,
                finetune=args.tabpfn_finetune,
                finetune_epochs=args.tabpfn_finetune_epochs,
                finetune_lr=args.tabpfn_finetune_lr,
            )
        else:
            icl_head = TabICLHead(
                task_type=task_type,
                device=str(accelerator.device),
                n_estimators=args.icl_n_estimators,
                max_context_size=args.icl_max_context,
            )

        # Fit on train embeddings
        icl_head.fit(train_embs, train_labels)

        # Evaluate on valid
        valid_score = eval_with_icl_head(
            icl_head, train_embs, train_labels,
            valid_embs, valid_labels, metric_name,
        )
        print(f"  valid_metric/{tn}/{metric_name}: {valid_score}")
        tbtracker.log(
            {f"valid_metric/{tn}/{metric_name}": valid_score}, step=step,
        )

        # Evaluate on test
        test_score = eval_with_icl_head(
            icl_head, train_embs, train_labels,
            test_embs, test_labels, metric_name,
        )
        print(f"  test_metric/{tn}/{metric_name}: {test_score}")
        tbtracker.log(
            {f"test_metric/{tn}/{metric_name}": test_score}, step=step,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main(args):
    tbconfig = ProjectConfiguration(
        project_dir=args.logdir, logging_dir=args.logdir,
    )
    accelerator = Accelerator(log_with="tensorboard", project_config=tbconfig)
    accelerator.init_trackers(args.logname)
    tbtracker = accelerator.get_tracker("tensorboard")

    # ── Griffin MPNN ──
    model = GriffinMod(
        hiddim=args.hiddim, num_mp=args.num_mp,
        use_rev=args.use_rev, use_gate=args.use_gate,
    )
    if args.loadpath is not None:
        accelerate.load_checkpoint_in_model(model, args.loadpath)

    # ── Head-specific components ──
    dec = None
    projector = None
    llm_decoder = None
    output_mlp = None
    linear_probe = None
    icl_projection = None
    layer_pooling = None
    attention_pool = None

    if args.head == "default":
        dec = getfloatdec(args.hiddim)

    elif args.head in ("tabpfn", "tabicl"):
        # ICL heads: trainable projection + optional linear probe for
        # Griffin fine-tuning. dec is kept for evaluation fallback.
        dec = getfloatdec(args.hiddim)
        icl_projection = ICLProjection(
            in_dim=args.hiddim,
            out_dim=args.icl_proj_dim,
            dropout=0.1,
        )
        if args.probe_epochs > 0:
            # Linear probe operates on projected dim, not raw Griffin dim
            linear_probe = LinearProbe(
                in_dim=args.icl_proj_dim,
                out_dim=args.output_mlp_dim,
            )

    else:
        # LLM heads: build projector + LLM
        llm_decoder = LLMDecoder(
            model_name=args.llm_model,
            frozen=args.llm_frozen,
            use_lora=args.use_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            device=accelerator.device,
        )
        # Adapt bottleneck: should not exceed LLM dim (e.g. Gemma 640)
        effective_bottleneck = min(args.projector_bottleneck, llm_decoder.llm_dim)
        projector = GriffinToLLMProjector(
            griffin_dim=args.hiddim,
            llm_dim=llm_decoder.llm_dim,
            bottleneck=effective_bottleneck,
            dropout=0.1,
        )
        if effective_bottleneck != args.projector_bottleneck:
            if accelerator.is_main_process:
                print(f"[Projector] Bottleneck clamped: {args.projector_bottleneck} "
                      f"→ {effective_bottleneck} (LLM dim={llm_decoder.llm_dim})")
        # OutputMLP creation is deferred until after task metadata is loaded,
        # so that out_channels can be auto-detected per task (see below).

        # Multi-layer pooling and attention pooling
        _cfg = llm_decoder.model.config
        max_layers = getattr(_cfg, "num_hidden_layers", None)
        if max_layers is None and hasattr(_cfg, "text_config"):
            max_layers = _cfg.text_config.num_hidden_layers
        effective_pool_layers = min(args.pool_layers, max_layers)
        if effective_pool_layers != args.pool_layers and accelerator.is_main_process:
            print(f"[Pool] pool_layers clamped: {args.pool_layers} → {effective_pool_layers} "
                  f"(LLM has {max_layers} layers)")
        layer_pooling = LayerPooling(pool_layers=effective_pool_layers)
        attention_pool = None
        if args.pool_mode == "attention":
            attention_pool = AttentionPool(dim=llm_decoder.llm_dim)

    # ── Optimizer (created after task metadata is loaded for OutputMLP) ──
    # Placeholder — actual optimizer creation is deferred to after task loading
    optimizer = None

    # ── Data loading ──
    graph = Graph(args.dataset)
    task = Task(args.dataset)

    # Audit prompts vs task metadata once on rank 0
    if accelerator.is_main_process:
        audit_task_prompts(task.metatask, verbose=True)

    # Load adjacency metadata for rich LLM prompts
    _metaadj_path = osp.join(args.dataset, "metaadj.yaml")
    if osp.exists(_metaadj_path):
        with open(_metaadj_path) as f:
            metaadj = yaml.safe_load(f)
    else:
        metaadj = {}

    tasknames = args.tasks
    if len(tasknames) == 1:
        if tasknames[0] == "ALLTASK":
            tasknames = [tn for tn in task.metatask]
        elif tasknames[0] == "RETTASK":
            tasknames = [
                tn for tn in task.metatask
                if task.metatask[tn]["task_type"] == "retrieval"
            ]
        elif tasknames[0] == "REGTASK":
            tasknames = [
                tn for tn in task.metatask
                if task.metatask[tn]["task_type"] == "regression"
            ]
        elif tasknames[0].startswith("EXCEPT__"):
            except_tn = tasknames[0][len("EXCEPT__"):]
            tasknames = [tn for tn in task.metatask if tn != except_tn]
        elif tasknames[0] in ["commerce-1", "commerce-2", "others-1", "others-2",
                               "binary-source", "binary-target",
                               "rel-train-broad", "rel-amazon-eval",
                               "rel-train-no-hm", "rel-hm-eval",
                               "rel-train-no-stack", "rel-stack-eval",
                               "rel-train-no-f1", "rel-f1-eval",
                               "rel-train-no-avito", "rel-avito-eval"]:
            with open("task_names.yaml", "r") as f:
                tasks_dict = yaml.load(f, Loader=yaml.FullLoader)
            tasknames = tasks_dict[tasknames[0]]

    if accelerator.is_main_process:
        print(f"Train tasks: {tasknames}")

    # ── Resolve eval_tasks (for cross-dataset transfer) ──
    def _resolve_task_keywords(task_list):
        """Expand keywords like ALLTASK, others-1, etc. into task name lists."""
        if len(task_list) == 1:
            kw = task_list[0]
            if kw == "ALLTASK":
                return [tn for tn in task.metatask]
            elif kw == "RETTASK":
                return [tn for tn in task.metatask if task.metatask[tn]["task_type"] == "retrieval"]
            elif kw == "REGTASK":
                return [tn for tn in task.metatask if task.metatask[tn]["task_type"] == "regression"]
            elif kw.startswith("EXCEPT__"):
                return [tn for tn in task.metatask if tn != kw[len("EXCEPT__"):]]
            elif kw in ["commerce-1", "commerce-2", "others-1", "others-2",
                        "binary-source", "binary-target",
                        "rel-train-broad", "rel-amazon-eval",
                        "rel-train-no-hm", "rel-hm-eval",
                        "rel-train-no-stack", "rel-stack-eval",
                        "rel-train-no-f1", "rel-f1-eval",
                        "rel-train-no-avito", "rel-avito-eval"]:
                with open("task_names.yaml", "r") as f:
                    tasks_dict = yaml.load(f, Loader=yaml.FullLoader)
                return tasks_dict[kw]
        return task_list

    eval_tasknames = tasknames  # default: evaluate on same tasks as training
    if args.eval_tasks is not None:
        eval_tasknames = _resolve_task_keywords(args.eval_tasks)
        if accelerator.is_main_process:
            print(f"Eval tasks: {eval_tasknames}")
    else:
        if accelerator.is_main_process:
            print(f"Eval tasks: (same as train)")

    # All tasks that need type lookup and OutputMLPs
    all_tasknames = list(dict.fromkeys(tasknames + eval_tasknames))  # deduplicated, ordered

    # Task type lookup (needed for LLM heads)
    task_type_dict = {}
    for tn in all_tasknames:
        tt = task.metatask[tn].get("task_type", "regression")
        task_type_dict[tn] = tt
        if accelerator.is_main_process:
            print(f"  {tn}: {tt}")

    # ── Deferred OutputMLP creation: auto-detect out_channels per task ──
    if args.head == "llm_mlp" and llm_decoder is not None:
        if args.unified_decoder:
            # Griffin-style unified decoder: back-project to Griffin space
            if args.pool_mode in ("last", "attention"):
                pool_dim = llm_decoder.llm_dim
            else:
                pool_dim = llm_decoder.llm_dim * 2  # entity/graph mode
            output_mlp = UnifiedDecoder(
                pool_dim=pool_dim,
                griffin_dim=args.hiddim,
                dropout=0.1,
                pool_mode=args.pool_mode,
            )
            if accelerator.is_main_process:
                print(f"  UnifiedDecoder: pool_dim={pool_dim} → griffin_dim={args.hiddim} "
                      f"(regression: dec, classification: @y.T)")
        elif args.task_type_heads:
            # Three per-task-type heads: regression / binary / multi-class
            max_classes = max(
                task.metatask[tn].get("num_class", 2) if task_type_dict[tn] != "regression" else 1
                for tn in all_tasknames
            )
            max_classes = max(max_classes, 3)  # at least 3 for multi-class head
            output_mlp = TaskTypeHeads(
                llm_dim=llm_decoder.llm_dim,
                max_classes=max_classes,
                hidden_dim=args.output_mlp_hidden if args.output_mlp_hidden > 0 else None,
                dropout=0.1,
                pool_mode=args.pool_mode,
            )
            if accelerator.is_main_process:
                print(f"  TaskTypeHeads: regression(1) + binary(2) + multiclass({max_classes})")
        elif args.shared_head:
            # Single shared head: out_channels = max(num_classes) across all tasks
            max_classes = max(
                task.metatask[tn].get("num_class", 2) if task_type_dict[tn] != "regression" else 1
                for tn in all_tasknames
            )
            max_classes = max(max_classes, 2)
            output_mlp = OutputMLP(
                llm_dim=llm_decoder.llm_dim,
                out_channels=max_classes,
                hidden_dim=args.output_mlp_hidden if args.output_mlp_hidden > 0 else None,
                dropout=0.1,
                pool_mode=args.pool_mode,
            )
            if accelerator.is_main_process:
                print(f"  SharedOutputMLP: out_channels={max_classes}")
        else:
            # Per-task heads (default)
            output_mlp_dict = nn.ModuleDict()
            for tn in all_tasknames:
                if task_type_dict[tn] == "regression":
                    out_ch = 1
                else:
                    out_ch = task.metatask[tn].get("num_class", 2)
                output_mlp_dict[tn] = OutputMLP(
                    llm_dim=llm_decoder.llm_dim,
                    out_channels=out_ch,
                    hidden_dim=args.output_mlp_hidden if args.output_mlp_hidden > 0 else None,
                    dropout=0.1,
                    pool_mode=args.pool_mode,
                )
                if accelerator.is_main_process:
                    print(f"  OutputMLP[{tn}]: out_channels={out_ch}")
            output_mlp = output_mlp_dict

    # ── Optimizer (deferred from above so OutputMLP dict is available) ──
    trainable_params = []

    if args.head == "default":
        trainable_params.extend(model.parameters())

    elif args.head in ("tabpfn", "tabicl"):
        if args.freeze_griffin:
            for p in model.parameters():
                p.requires_grad = False
            if accelerator.is_main_process:
                print("[Griffin] Frozen — using pretrained weights only")
        else:
            trainable_params.extend(model.parameters())
        trainable_params.extend(icl_projection.parameters())
        if linear_probe is not None:
            trainable_params.extend(linear_probe.parameters())

    else:
        # LLM heads
        if args.freeze_griffin:
            for p in model.parameters():
                p.requires_grad = False
            if accelerator.is_main_process:
                print("[Griffin] Frozen — using pretrained weights only")
        else:
            trainable_params.extend(model.parameters())

        trainable_params.extend(projector.parameters())

        if output_mlp is not None:
            trainable_params.extend(output_mlp.parameters())

        if args.use_lora:
            trainable_params.extend(
                p for p in llm_decoder.model.parameters() if p.requires_grad
            )

        # Layer pooling weights (only if K > 1)
        if layer_pooling is not None and args.pool_layers > 1:
            trainable_params.extend(layer_pooling.parameters())
        # Attention pool query (only if pool_mode == "attention")
        if attention_pool is not None:
            trainable_params.extend(attention_pool.parameters())

    optimizer = torch.optim.AdamW(
        [p for p in trainable_params if p.requires_grad],
        lr=args.lr, weight_decay=args.wd,
    )

    floatembmodel = SimpleRepeater(args.hiddim)

    dataset = construct_dataset(graph, task, tasknames, "train", args, floatembmodel)
    valid_dataset_dict = {
        tn: construct_dataset(graph, task, [tn], "valid", args, floatembmodel)
        for tn in eval_tasknames
    }
    test_dataset_dict = {
        tn: construct_dataset(graph, task, [tn], "test", args, floatembmodel)
        for tn in eval_tasknames
    }
    metric_dict = {tn: task.metatask[tn]["metric"] for tn in eval_tasknames}
    best_valid_metric = -torch.inf
    best_checkpoint_path = None
    best_epoch = 0

    # ── Accelerator prepare ──
    if args.head == "default":
        model, dec, optimizer = accelerator.prepare(model, dec, optimizer)
    elif args.head in ("tabpfn", "tabicl"):
        if linear_probe is not None:
            model, dec, icl_projection, linear_probe, optimizer = accelerator.prepare(
                model, dec, icl_projection, linear_probe, optimizer,
            )
        else:
            model, dec, icl_projection, optimizer = accelerator.prepare(
                model, dec, icl_projection, optimizer,
            )
    else:
        # Move pooling modules to the right device (and DDP-wrap if needed)
        if layer_pooling is not None and args.pool_layers > 1:
            layer_pooling = accelerator.prepare(layer_pooling)
        if attention_pool is not None:
            attention_pool = accelerator.prepare(attention_pool)

        prepare_list = [model, projector, optimizer]
        if output_mlp is not None:
            prepare_list.insert(2, output_mlp)
        prepared = accelerator.prepare(*prepare_list)
        if output_mlp is not None:
            model, projector, output_mlp, optimizer = prepared
        else:
            model, projector, optimizer = prepared

    if accelerator.is_main_process:
        g_params = sum(p.numel() for p in model.parameters())
        t_params = sum(p.numel() for p in trainable_params if p.requires_grad)
        msg = f"\n[Params] Griffin: {g_params:,}"
        if projector is not None:
            msg += f" | Projector: {sum(p.numel() for p in projector.parameters()):,}"
        if output_mlp is not None:
            msg += f" | OutputMLP: {sum(p.numel() for p in output_mlp.parameters()):,}"
        if icl_projection is not None:
            msg += f" | ICLProjection: {sum(p.numel() for p in icl_projection.parameters()):,}"
        if linear_probe is not None:
            msg += f" | LinearProbe: {sum(p.numel() for p in linear_probe.parameters()):,}"
        if llm_decoder is not None:
            msg += f" | LLM: {sum(p.numel() for p in llm_decoder.model.parameters()):,}"
        msg += f"\n[Params] Trainable: {t_params:,}\n"
        print(msg)

    # ── Eval-only keyword arguments (eval_task signature) ──
    head_kwargs = dict(
        projector=projector, llm_decoder=llm_decoder,
        output_mlp=output_mlp, task_type_dict=task_type_dict,
        metanode=graph.metanode, metaadj=metaadj,
        metatask=task.metatask,
        layer_pooling=layer_pooling if args.head in ("llm_mlp",) else None,
        attention_pool=attention_pool if args.head in ("llm_mlp",) else None,
    )

    # ── Test-only mode ──
    if args.mode == "test":
        if args.head in ("tabpfn", "tabicl"):
            # ICL heads: extract embeddings → fit → predict
            _run_icl_evaluation(
                model, dataset, valid_dataset_dict, test_dataset_dict,
                tasknames, task_type_dict, metric_dict,
                args, accelerator, tbtracker, step=0,
                icl_projection=icl_projection,
                graph=graph, task_obj=task,
            )
            accelerator.end_training()
            return

        # ── Few-shot fine-tuning on eval tasks before testing ──
        if args.finetune_samples > 0 and args.head in ("llm", "llm_mlp"):
            if accelerator.is_main_process:
                print(f"\n[Fine-tune] Per-task adaptation: {args.finetune_samples} "
                      f"samples/task, {args.finetune_epochs} epochs, lr={args.finetune_lr}")

            # Freeze everything except output heads (and optionally projector)
            for p in model.parameters():
                p.requires_grad = False
            if not args.finetune_projector:
                for p in projector.parameters():
                    p.requires_grad = False
            model.eval()

            if accelerator.is_main_process and args.finetune_projector:
                print(f"[Fine-tune] Also fine-tuning projector (lr={args.finetune_lr * 0.1:.1e})")

            # Save original states so we can reload between tasks
            _mlp_unwrapped = getattr(output_mlp, 'module', output_mlp)
            original_head_state = {
                k: v.clone() for k, v in _mlp_unwrapped.state_dict().items()
            }
            _proj_unwrapped = getattr(projector, 'module', projector)
            original_proj_state = {
                k: v.clone() for k, v in _proj_unwrapped.state_dict().items()
            } if args.finetune_projector else None

            # Snapshot LoRA params (only the trainable adapters, not base LLM)
            original_lora_state = None
            if args.finetune_lora and args.use_lora:
                _llm_unwrapped = getattr(llm_decoder.model, 'module', llm_decoder.model)
                original_lora_state = {
                    n: p.detach().clone()
                    for n, p in _llm_unwrapped.named_parameters()
                    if p.requires_grad
                }

            test_metric = {}
            for tn in eval_tasknames:
                if accelerator.is_main_process:
                    print(f"\n--- {tn} (type={task_type_dict[tn]}) ---")

                # Reset to original checkpoint state before each task
                _mlp_unwrapped.load_state_dict(original_head_state)
                if args.finetune_projector:
                    _proj_unwrapped.load_state_dict(original_proj_state)
                if original_lora_state is not None:
                    _llm_unwrapped = getattr(llm_decoder.model, 'module', llm_decoder.model)
                    with torch.no_grad():
                        for n, p in _llm_unwrapped.named_parameters():
                            if n in original_lora_state:
                                p.copy_(original_lora_state[n])

                # Build single-task fine-tuning dataset
                ft_dataset = construct_dataset(
                    graph, task, [tn], "train", args, floatembmodel,
                )

                # Create fresh optimizer for this task
                ft_param_groups = [
                    {"params": list(output_mlp.parameters()), "lr": args.finetune_lr},
                ]
                if args.finetune_projector:
                    ft_param_groups.append(
                        {"params": list(projector.parameters()), "lr": args.finetune_lr * 0.1},
                    )
                if args.finetune_lora and args.use_lora:
                    # Unfreeze LoRA params (base LLM stays frozen)
                    lora_params = [
                        p for p in llm_decoder.model.parameters()
                        # LoRA params have requires_grad=True after get_peft_model;
                        # base params have requires_grad=False
                        if p.requires_grad
                    ]
                    if lora_params:
                        ft_param_groups.append(
                            {"params": lora_params, "lr": args.finetune_lr * 0.1},
                        )
                        if accelerator.is_main_process and tn == eval_tasknames[0]:
                            n_lora = sum(p.numel() for p in lora_params)
                            print(f"[Fine-tune] LoRA enabled: {n_lora:,} params at lr={args.finetune_lr * 0.1:.1e}")
                ft_optimizer = torch.optim.AdamW(ft_param_groups)
                ft_optimizer = accelerator.prepare(ft_optimizer)

                output_mlp.train()

                for ft_epoch in range(args.finetune_epochs):
                    ft_dataset.rebuild_indice_downsample_absolute(
                        accelerator, args.finetune_samples,
                        args.downsample_seed if args.downsample_seed else 42,
                    )
                    ft_loader = DataLoader(
                        ft_dataset, shuffle=True, batch_size=1,
                        collate_fn=lambda xlist: xlist[0],
                        num_workers=4, persistent_workers=False,
                    )
                    ft_loader = accelerator.prepare(ft_loader)
                    ft_step = 0
                    epoch_loss = 0.0
                    for data in ft_loader:
                        ft_step += 1
                        ft_optimizer.zero_grad()
                        loss = compute_loss(
                            model, dec, data, args,
                            projector=projector, llm_decoder=llm_decoder,
                            output_mlp=output_mlp, task_type_dict=task_type_dict,
                            metanode=graph.metanode, metaadj=metaadj,
                            metatask=task.metatask,
                            layer_pooling=layer_pooling if args.head in ("llm_mlp",) else None,
                            attention_pool=attention_pool if args.head in ("llm_mlp",) else None,
                        )
                        accelerator.backward(loss)
                        all_ft_params = [p for pg in ft_param_groups for p in pg["params"]]
                        torch.nn.utils.clip_grad_norm_(all_ft_params, 0.5)
                        ft_optimizer.step()
                        epoch_loss += loss.item()
                    if accelerator.is_main_process:
                        avg_loss = epoch_loss / max(ft_step, 1)
                        print(f"  [ft] epoch {ft_epoch}: {ft_step} steps, avg_loss={avg_loss:.4f}")
                    accelerator.wait_for_everyone()

                # Test this task with its fine-tuned heads
                output_mlp.eval()
                test_metric[tn] = eval_task(
                    model, dec, test_dataset_dict[tn], args, accelerator,
                    metric_dict[tn], **head_kwargs,
                )
                if accelerator.is_main_process:
                    print(f"  test_metric/{tn}: {test_metric[tn]}")

            if accelerator.is_main_process:
                print(f"\n[Fine-tune] All tasks complete.")
                avg = sum(test_metric.values()) / len(test_metric)
                print(f"Average test metric: {avg}")
            accelerator.end_training()
            return

        test_metric = {}
        for tn in eval_tasknames:
            if accelerator.is_main_process:
                print(f"test {tn}...")
            test_metric[tn] = eval_task(
                model, dec, test_dataset_dict[tn], args, accelerator,
                metric_dict[tn], **head_kwargs,
            )
            if accelerator.is_main_process:
                print(f"test_metric/{tn}: {test_metric[tn]}", flush=True)
        accelerator.end_training()
        return

    # ═══════════════════════════════════════════════════════════════════════════
    # ICL heads (tabpfn/tabicl): specialized training + evaluation flow
    # ═══════════════════════════════════════════════════════════════════════════
    if args.head in ("tabpfn", "tabicl"):
        # Phase 1 (optional): Fine-tune Griffin with linear probe
        if args.probe_epochs > 0 and linear_probe is not None:
            if accelerator.is_main_process:
                print(f"\n[Phase 1] Fine-tuning Griffin with linear probe "
                      f"for {args.probe_epochs} epochs")
            model.train()
            icl_projection.train()
            linear_probe.train()
            step = 0
            for epoch in range(args.probe_epochs):
                if accelerator.is_main_process:
                    print(f"Probe epoch {epoch}")
                dataset.rebuild_indice(accelerator)
                loader = DataLoader(
                    dataset, shuffle=True, batch_size=1,
                    collate_fn=lambda xlist: xlist[0],
                    num_workers=8, prefetch_factor=4,
                    persistent_workers=False, pin_memory=True,
                )
                loader = accelerator.prepare(loader)
                for data in loader:
                    step += 1
                    optimizer.zero_grad()
                    loss = compute_loss(
                        model, dec, data, args,
                        linear_probe=linear_probe,
                        icl_projection=icl_projection,
                        task_type_dict=task_type_dict,
                    )
                    accelerator.backward(loss)
                    optimizer.step()
                    if step % 50 == 0 and accelerator.is_main_process:
                        print(f"  probe step {step}: loss = {loss.item():.4f}")
                        tbtracker.log({"probe_loss": loss.item()}, step=step)
                accelerator.wait_for_everyone()

            # Save fine-tuned Griffin + projection
            if args.savepath is not None and accelerator.is_main_process:
                probe_path = osp.join(args.savepath, "probe_finetuned")
                print(f"Saving probe-finetuned Griffin + projection to {probe_path}")
                accelerator.save_model(model, probe_path)
                torch.save(
                    accelerator.unwrap_model(icl_projection).state_dict(),
                    osp.join(probe_path, "icl_projection.pt"),
                )

        # Phase 2: Extract embeddings and evaluate with ICL head
        if accelerator.is_main_process:
            print(f"\n[Phase 2] Extracting embeddings and evaluating with "
                  f"{args.head} head")

        _run_icl_evaluation(
            model, dataset, valid_dataset_dict, test_dataset_dict,
            tasknames, task_type_dict, metric_dict,
            args, accelerator, tbtracker, step=step if args.probe_epochs > 0 else 0,
            icl_projection=icl_projection,
            graph=graph, task_obj=task,
        )

        accelerator.end_training()
        return

    # ═══════════════════════════════════════════════════════════════════════════
    # Gradient-based heads (default/llm/llm_mlp): standard training loop
    # ═══════════════════════════════════════════════════════════════════════════
    model.train()
    if projector is not None:
        projector.train()
    if output_mlp is not None:
        output_mlp.train()
    step = 0

    for epoch in range(args.maxepoch):
        if accelerator.is_main_process:
            print(f"Epoch {epoch} starts")

        # ── 2-phase training: unfreeze Griffin after warmup ──
        if (args.head != "default" and not args.freeze_griffin
                and args.warmup_epochs > 0 and epoch == args.warmup_epochs):
            if accelerator.is_main_process:
                print(f"[Phase 2] Unfreezing Griffin MPNN at epoch {epoch}")
            for p in model.parameters():
                p.requires_grad = True
            # Rebuild optimizer with Griffin params at lower LR
            param_groups = [
                {"params": list(projector.parameters()), "lr": args.projector_lr},
            ]
            if output_mlp is not None:
                param_groups.append(
                    {"params": list(output_mlp.parameters()), "lr": args.lr},
                )
            if args.use_lora:
                param_groups.append({
                    "params": [p for p in llm_decoder.model.parameters() if p.requires_grad],
                    "lr": args.lr * 0.1,
                })
            param_groups.append(
                {"params": list(model.parameters()), "lr": args.lr * 0.1},
            )
            optimizer = torch.optim.AdamW(
                param_groups, weight_decay=args.wd,
            )
            optimizer = accelerator.prepare(optimizer)

        # ── Warmup phase: freeze Griffin for first N epochs ──
        if (args.head != "default" and not args.freeze_griffin
                and args.warmup_epochs > 0 and epoch < args.warmup_epochs):
            for p in model.parameters():
                p.requires_grad = False

        if args.downsample_num > 0:
            dataset.rebuild_indice_downsample_absolute(
                accelerator, args.downsample_num, args.downsample_seed,
            )
        else:
            dataset.rebuild_indice(accelerator)
        loader = DataLoader(
            dataset, shuffle=True, batch_size=1,
            collate_fn=lambda xlist: xlist[0],
            num_workers=8, prefetch_factor=4,
            persistent_workers=False, pin_memory=True,
        )
        loader = accelerator.prepare(loader)

        for data in loader:
            step += 1
            optimizer.zero_grad()

            loss = compute_loss(
                model, dec, data, args,
                projector=projector, llm_decoder=llm_decoder,
                output_mlp=output_mlp, task_type_dict=task_type_dict,
                metanode=graph.metanode, metaadj=metaadj,
                metatask=task.metatask,
                layer_pooling=layer_pooling if args.head in ("llm_mlp",) else None,
                attention_pool=attention_pool if args.head in ("llm_mlp",) else None,
            )

            accelerator.backward(loss)

            if args.head != "default":
                all_params = [p for p in trainable_params if p.requires_grad]
                if projector is not None:
                    all_params.extend(projector.parameters())
                torch.nn.utils.clip_grad_norm_(all_params, 0.5)

            optimizer.step()

            if step % 10 == 0:
                tbtracker.log({"training_loss": loss.item()}, step=step)
                if accelerator.is_main_process and step % 50 == 0:
                    print(f"  step {step}: loss = {loss.item():.4f}")

        accelerator.wait_for_everyone()

        # ── Save checkpoint ──
        checkpoint_path = (
            osp.join(args.savepath, f"checkpoint-{epoch}-{step}")
            if args.savepath is not None else None
        )
        if args.savepath is not None and accelerator.is_main_process:
            accelerator.save_model(model, checkpoint_path)
            if projector is not None:
                torch.save(
                    accelerator.unwrap_model(projector).state_dict(),
                    osp.join(checkpoint_path, "projector.pt"),
                )
            if output_mlp is not None:
                torch.save(
                    accelerator.unwrap_model(output_mlp).state_dict(),
                    osp.join(checkpoint_path, "output_mlp.pt"),
                )

        # ── Validation ──
        if (epoch + 1) % args.eval_per_epoch == 0:
            eval_metric = {}
            for tn in eval_tasknames:
                if accelerator.is_main_process:
                    print(f"Validating {tn}...")
                eval_metric[tn] = eval_task(
                    model, dec, valid_dataset_dict[tn], args, accelerator,
                    metric_dict[tn], **head_kwargs,
                )
                if accelerator.is_main_process:
                    tbtracker.log(
                        {f"valid_metric/{tn}/{metric_dict[tn]}": eval_metric[tn]},
                        step=step,
                    )
                    print(
                        f"valid_metric/{tn}/{metric_dict[tn]}: {eval_metric[tn]}",
                        flush=True,
                    )
            avg_valid_metric = 0.0
            if accelerator.is_main_process:
                avg_valid_metric = sum(eval_metric.values()) / len(eval_metric)
                print(f"Average valid metric: {avg_valid_metric}", flush=True)

            # Broadcast main process metric to all ranks (avoid dilution from zeros)
            metric_tensor = torch.tensor(
                [avg_valid_metric], device=accelerator.device,
            )
            gathered = accelerator.gather(metric_tensor)
            # Use rank-0 value (main process), not mean across ranks
            avg_valid_metric = gathered[0].item()

            if avg_valid_metric > best_valid_metric:
                best_valid_metric = avg_valid_metric
                best_checkpoint_path = checkpoint_path
                best_epoch = epoch
                # Test on best validation
                eval_metric = {}
                for tn in eval_tasknames:
                    if accelerator.is_main_process:
                        print(f"test {tn}...")
                    eval_metric[tn] = eval_task(
                        model, dec, test_dataset_dict[tn], args, accelerator,
                        metric_dict[tn], **head_kwargs,
                    )
                    if accelerator.is_main_process:
                        tbtracker.log(
                            {f"test_metric/{tn}/{metric_dict[tn]}": eval_metric[tn]},
                            step=step,
                        )
                        print(
                            f"test_metric/{tn}/{metric_dict[tn]}: {eval_metric[tn]}",
                            flush=True,
                        )
                if accelerator.is_main_process:
                    avg_test_metric = sum(eval_metric.values()) / len(eval_metric)
                    print(f"Average test metric: {avg_test_metric}", flush=True)

            if args.patience > 0 and epoch - best_epoch > args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

            model.train()
            if projector is not None:
                projector.train()
            if output_mlp is not None:
                output_mlp.train()

    # ── Final test with best checkpoint ──
    test_metric = {}
    if best_checkpoint_path is not None:
        if accelerator.is_main_process:
            print(f"Loading best checkpoint from {best_checkpoint_path}")
        unwrap_model = accelerator.unwrap_model(model)
        accelerate.load_checkpoint_in_model(unwrap_model, best_checkpoint_path)
        model = accelerator.prepare(unwrap_model)

        # Load projector / output_mlp checkpoints
        if projector is not None:
            proj_path = osp.join(best_checkpoint_path, "projector.pt")
            if osp.exists(proj_path):
                accelerator.unwrap_model(projector).load_state_dict(
                    torch.load(proj_path, map_location=accelerator.device),
                )
        if output_mlp is not None:
            mlp_path = osp.join(best_checkpoint_path, "output_mlp.pt")
            if osp.exists(mlp_path):
                accelerator.unwrap_model(output_mlp).load_state_dict(
                    torch.load(mlp_path, map_location=accelerator.device),
                )

        if accelerator.is_main_process and args.savepath is not None:
            best_dir = osp.join(args.savepath, "best_checkpoint")
            print(f"Saving best checkpoint at {best_dir}")
            accelerator.save_model(model, best_dir)
            if projector is not None:
                torch.save(
                    accelerator.unwrap_model(projector).state_dict(),
                    osp.join(best_dir, "projector.pt"),
                )
            if output_mlp is not None:
                torch.save(
                    accelerator.unwrap_model(output_mlp).state_dict(),
                    osp.join(best_dir, "output_mlp.pt"),
                )

    for tn in eval_tasknames:
        if accelerator.is_main_process:
            print(f"Testing {tn}...")
        eval_metric = eval_task(
            model, dec, test_dataset_dict[tn], args, accelerator,
            metric_dict[tn], **head_kwargs,
        )
        if accelerator.is_main_process:
            tbtracker.log({f"test_metric/{tn}": eval_metric}, step=step)
            print(f"test_metric/{tn}: {eval_metric}")
            test_metric[tn] = eval_metric

    if accelerator.is_main_process:
        avg_metric = sum(test_metric.values()) / len(test_metric)
        print(f"Average test metric: {avg_metric}")

    accelerator.end_training()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(
        description="Griffin + LLM Unified Training Pipeline",
    )
    parser.add_argument("--mode", type=str, default="train",
                        help="train or test")
    parser.add_argument("dataset", type=str)
    parser.add_argument("logdir", type=str)
    parser.add_argument("logname", type=str)
    parser.add_argument("--tasks", type=str, nargs="+", default=["ALLTASK"],
                        help="ALLTASK | RETTASK | REGTASK | task names (used for training)")
    parser.add_argument("--eval_tasks", type=str, nargs="+", default=None,
                        help="Tasks for validation/test (if different from --tasks). "
                             "Supports same keywords as --tasks (ALLTASK, others-1, etc). "
                             "When set, training uses --tasks but validation/test/early-stopping "
                             "use --eval_tasks. Used for cross-dataset transfer experiments.")
    parser.add_argument("--savepath", type=str, default=None)
    parser.add_argument("--loadpath", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Enable debug diagnostics: print neighbor embedding "
                             "info, shape checks, and per-sample diversity assertions")

    # ── Head selection ──
    parser.add_argument("--head", type=str, default="default",
                        choices=["default", "llm", "llm_mlp", "tabpfn", "tabicl"],
                        help="Prediction head: 'default' (Griffin dec/@y.T), "
                             "'llm' (text generation), 'llm_mlp' (hidden→MLP), "
                             "'tabpfn' (TabPFN v2 ICL), 'tabicl' (TabICL v2 ICL)")

    # ── Training ──
    parser.add_argument("--batchsize", type=int, default=512)
    parser.add_argument("--eval_batchsize", type=int)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=4e-4)
    parser.add_argument("--maxepoch", type=int, default=10)
    parser.add_argument("--patience", type=int, default=-1)
    parser.add_argument("--eval_per_epoch", type=int, default=3)
    parser.add_argument("--downsample_num", type=int, default=0,
                        help="Fine-tune on this many samples per task per epoch "
                             "(0=use all). Griffin transfer uses 1024.")
    parser.add_argument("--downsample_seed", type=int, default=42)

    # ── Griffin MPNN ──
    parser.add_argument("--num_mp", type=int, default=4)
    parser.add_argument("--hiddim", type=int, default=512)
    parser.add_argument("--fanout", type=int, default=10)
    parser.add_argument("--fewshotfanout", type=int, default=3)
    parser.add_argument("--hop", type=int, default=2)
    parser.add_argument("--use_rev", type=str2bool, default=True)
    parser.add_argument("--use_gate", type=str2bool, default=True)

    # ── LLM-specific (only used when --head is llm or llm_mlp) ──
    parser.add_argument("--llm_model", type=str,
                        default="meta-llama/Llama-3.2-1B",
                        help="HuggingFace model ID for the LLM decoder. "
                             "Tested: meta-llama/Llama-3.2-1B (2048d), "
                             "google/gemma-3-270m (640d), "
                             "Qwen/Qwen3-0.6B (1024d)")
    parser.add_argument("--llm_frozen", action="store_true", default=True,
                        help="Freeze LLM weights")
    parser.add_argument("--use_lora", action="store_true", default=False,
                        help="Apply LoRA to LLM")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--freeze_griffin", action="store_true", default=False,
                        help="Freeze Griffin MPNN permanently")
    parser.add_argument("--max_new_tokens", type=int, default=16)

    # ── Projection / neighbor / ICL ──
    parser.add_argument("--projector_bottleneck", type=int, default=1024,
                        help="Bottleneck dim in projection layer")
    parser.add_argument("--projector_lr", type=float, default=1e-3,
                        help="LR for projector in phase-2 training")
    parser.add_argument("--neighbor_tokens", type=int, default=0,
                        help="Number of neighbor embeddings to inject (0=disabled)")
    parser.add_argument("--num_demo", type=int, default=0,
                        help="Number of ICL demo examples (0=disabled)")
    parser.add_argument("--warmup_epochs", type=int, default=0,
                        help="Epochs to train projector only before unfreezing Griffin")
    parser.add_argument("--entity_after_question", action="store_true", default=True,
                        help="Place entity/neighbor tokens after question text (default). "
                             "Use --no_entity_after_question for old order (entity before question).")
    parser.add_argument("--no_entity_after_question", dest="entity_after_question",
                        action="store_false",
                        help="Place entity/neighbor tokens before question (old prompt order)")

    # ── Output MLP (only for --head llm_mlp) ──
    parser.add_argument("--output_mlp_dim", type=int, default=1,
                        help="Output dimension of MLP head (1 for regression, "
                             "num_classes for classification)")
    parser.add_argument("--shared_head", action="store_true", default=False,
                        help="Use single shared OutputMLP instead of per-task heads. "
                             "out_channels = max(num_classes) across all tasks. "
                             "Essential for transfer to unseen tasks via --eval_tasks.")
    parser.add_argument("--task_type_heads", action="store_true", default=False,
                        help="Use 3 per-task-type heads (regression/binary/multi-class) "
                             "instead of per-task heads. Each head is shared across tasks "
                             "of the same output structure. Best for transfer with mixed "
                             "task types. Overrides --shared_head.")
    parser.add_argument("--unified_decoder", action="store_true", default=False,
                        help="Use Griffin-style unified decoder: back-project LLM hidden "
                             "states to Griffin embedding space, then use dec() for "
                             "regression and @y.T for classification. No task-specific "
                             "output heads — classification uses graph class embeddings. "
                             "Overrides --shared_head and --task_type_heads.")
    parser.add_argument("--output_mlp_hidden", type=int, default=0,
                        help="Hidden dim in OutputMLP (0=single linear, no hidden layer). "
                             "Use 0 for few-shot transfer to minimize params.")
    parser.add_argument("--pool_mode", type=str, default="entity",
                        choices=["last", "entity", "graph", "attention"],
                        help="How to pool LLM hidden states for --head llm_mlp: "
                             "'last' = last token only (1x llm_dim), "
                             "'entity' = entity + last token (2x llm_dim, default), "
                             "'graph' = mean(entity+neighbors) + last token (2x llm_dim), "
                             "'attention' = learned attention over all positions (1x llm_dim)")
    parser.add_argument("--pool_layers", type=int, default=1,
                        help="Number of LLM layers to pool (1=last layer only, "
                             "K>1=learned weighted sum of last K layers). "
                             "Auto-clamped to num_hidden_layers. Recommended: 1 or 4.")

    # ── TabPFN / TabICL (only for --head tabpfn or tabicl) ──
    parser.add_argument("--tabpfn_version", type=str, default="v2.5",
                        choices=["v2.5", "v2.6", "default"],
                        help="TabPFN model version: v2.5, v2.6, or default (package default)")
    parser.add_argument("--tabpfn_finetune", action="store_true", default=False,
                        help="Fine-tune TabPFN on Griffin embeddings "
                             "(uses FinetunedTabPFNClassifier/Regressor)")
    parser.add_argument("--tabpfn_finetune_epochs", type=int, default=30,
                        help="Epochs for TabPFN fine-tuning")
    parser.add_argument("--tabpfn_finetune_lr", type=float, default=2e-5,
                        help="Learning rate for TabPFN fine-tuning")
    parser.add_argument("--icl_n_estimators", type=int, default=8,
                        help="Number of ensemble members for TabPFN/TabICL")
    parser.add_argument("--icl_proj_dim", type=int, default=128,
                        help="Output dimension of ICL projection layer "
                             "(compresses Griffin hiddim → this dim for TabPFN/TabICL)")
    parser.add_argument("--icl_max_context", type=int, default=10000,
                        help="Max ICL context examples (subsampled if training "
                             "set exceeds this). TabPFN default: 10000")
    parser.add_argument("--probe_epochs", type=int, default=0,
                        help="Epochs to fine-tune Griffin + projection with "
                             "linear probe before ICL evaluation (0=skip)")
    parser.add_argument("--finetune_samples", type=int, default=0,
                        help="Fine-tune output heads on N samples per eval task "
                             "before testing. Used with --mode test to adapt "
                             "a transfer checkpoint to target tasks. "
                             "Only output heads are trained (Griffin/projector/LLM frozen).")
    parser.add_argument("--finetune_epochs", type=int, default=5,
                        help="Number of fine-tuning epochs when --finetune_samples > 0")
    parser.add_argument("--finetune_lr", type=float, default=1e-3,
                        help="Learning rate for fine-tuning output heads")
    parser.add_argument("--finetune_projector", action="store_true", default=False,
                        help="Also fine-tune the projector during --finetune_samples "
                             "(default: only output heads). Projector uses finetune_lr * 0.1.")
    parser.add_argument("--finetune_lora", action="store_true", default=False,
                        help="Also fine-tune LoRA adapters on the LLM during "
                             "--finetune_samples. Requires --use_lora to be set "
                             "so LoRA adapters exist on the LLM. LoRA uses "
                             "finetune_lr * 0.1.")
    parser.add_argument("--focal_loss", action="store_true", default=False,
                        help="Use focal loss for binary classification tasks "
                             "(num_class==2). Helps when class imbalance is severe. "
                             "Has no effect on regression or multiclass losses.")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Gamma for focal loss; gamma=0 reduces to cross-entropy.")
    parser.add_argument("--cot_prompt", action="store_true", default=False,
                        help="Insert per-task chain-of-thought reasoning hints "
                             "between the question and 'Answer:' position. The "
                             "LLM doesn't generate reasoning text (the head is "
                             "MLP, not autoregressive) — it gets extra prompt "
                             "positions to attend over before the MLP reads the "
                             "answer hidden state. See TASK_REASONING in "
                             "task_prompts.py for per-task hints.")

    args = parser.parse_args()
    args.eval_batchsize = (
        args.batchsize if args.eval_batchsize is None else args.eval_batchsize
    )

    # Auto-adjust defaults for LLM heads
    if args.head in ("llm", "llm_mlp"):
        if args.batchsize == 512:
            args.batchsize = 4
            print(f"[Auto] Reduced batchsize to {args.batchsize} for LLM head")
        if args.lr == 3e-4:
            args.lr = 1e-4
            print(f"[Auto] Reduced LR to {args.lr} for LLM head")
        if args.eval_per_epoch == 3:
            args.eval_per_epoch = 1

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.savepath:
        os.makedirs(args.savepath, exist_ok=True)

    main(args)
