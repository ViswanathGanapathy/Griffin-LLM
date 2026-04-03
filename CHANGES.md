# CHANGES — Griffin Unified Training Pipeline

## Summary

The Griffin codebase has been extended to support **five prediction heads** over the same Griffin MPNN backbone, enabling systematic comparison of different output strategies on RelBench tasks. The heads span gradient-based training (default, LLM, LLM+MLP) and in-context learning (TabPFN v2, TabICL v2).

---

## New Files

### `task_prompts.py`

Task-specific natural-language descriptions and questions for all 24 multi-table RelBench tasks (commerce-1, commerce-2, others-1, others-2). Used by LLM heads to construct structured prompts.

- `TASK_DESCRIPTION` dict — 1–2 sentence descriptions of each prediction task.
- `TASK_QUESTION` dict — targeted questions with output format instructions.
- `get_task_description(task_name)` / `get_task_question(task_name, task_type)` — auto-generate fallback prompts for **unseen tasks** (critical for generalization experiments).

### `tabular_heads.py`

Wrappers for TabPFN v2 and TabICL v2 as Griffin prediction heads.

- **`TabPFNHead`** — Wraps `tabpfn.TabPFNClassifier`/`TabPFNRegressor` (and their finetuned variants). Supports frozen ICL inference and optional TabPFN fine-tuning on extracted Griffin embeddings.
- **`TabICLHead`** — Wraps `tabicl.TabICLClassifier`/`TabICLRegressor`. Uses TabICL's 3-stage architecture (ColEmbed → RowInteract → ICL) for in-context prediction.
- **`LinearProbe`** — Simple `LayerNorm → Linear` head used as a proxy loss to fine-tune Griffin MPNN weights when the final prediction head (TabPFN/TabICL) is non-differentiable.
- **`extract_embeddings(model, dataset, accelerator)`** — Runs Griffin in eval mode, collects seed-node embeddings and labels across all data-parallel processes. Returns numpy arrays ready for ICL heads.
- **`eval_with_icl_head(icl_head, train_embs, ...)`** — Fits an ICL head on train embeddings and evaluates on test, returning the metric score.

---

## Modified Files

### `hloaderwrapper.py`

**Added `LoaderWrapperTaskLLM` class** (extends `LoaderWrapperTask`).

Returns three extra fields per batch beyond the standard 9-element tuple:
1. `taskname` (str) — which task this batch belongs to (fixes the multi-task TODO)
2. `rootnodetype` (str) — entity type of seed nodes (for LLM prompt context)
3. `neighbor_mask` (BoolTensor) — marks 1-hop neighbors of seed nodes in the unified graph (for neighbor embedding injection)

The original `LoaderWrapperTask` is **unchanged** — used when `--head default`, `--head tabpfn`, or `--head tabicl`.

### `hmaintask_combine_llm.py`

**Complete rewrite** of the LLM integration file into a unified training pipeline supporting all five heads.

#### Architecture Components

| Component | Class | Purpose |
|-----------|-------|---------|
| Projection | `GriffinToLLMProjector` | Maps Griffin 512-d → LLM embedding space via bottleneck (512→1024→llm_dim) with Sigmoid activation |
| LLM Wrapper | `LLMDecoder` | HuggingFace causal LM with frozen/LoRA modes, Yes/No token caching |
| Output MLP | `OutputMLP` | Maps LLM hidden states → prediction (regression scalar or classification logits) |
| Linear Probe | `LinearProbe` | Proxy head for Griffin fine-tuning with ICL heads |
| Prompt Builder | `build_llm_inputs()` | Constructs structured prompts with task description, entity embeddings, neighbor tokens, ICL demos |

#### Head Modes

| `--head` | Training | Inference | Gradient flows to Griffin? |
|----------|----------|-----------|---------------------------|
| `default` | MSE/CE with `dec`/@`y.T` | Same | Yes |
| `llm` | LM loss on answer tokens | Text generation or token-logit extraction | Yes (through projector) |
| `llm_mlp` | MSE/CE on MLP output | LLM hidden state → MLP | Yes (through projector + LLM) |
| `tabpfn` | Optional linear probe | Extract embeddings → TabPFN ICL | Only during probe phase |
| `tabicl` | Optional linear probe | Extract embeddings → TabICL ICL | Only during probe phase |

#### Key Features Added

1. **Neighbor embedding injection** (`--neighbor_tokens K`): Projects 1-hop neighbor embeddings and prepends them to the LLM prompt alongside the seed entity embedding. Gives the LLM richer relational context.

2. **In-context learning demos** (`--num_demo D`): For LLM heads, prepends D labeled examples in `[embed] → [label]` format before the question, enabling few-shot prompting.

3. **Token-logit extraction**: For binary classification tasks, extracts logits at the Yes/No token positions directly instead of generating and parsing text. Faster and more robust.

4. **2-phase training** (`--warmup_epochs N`): Phase 1 freezes Griffin and trains only the projector (alignment). Phase 2 unfreezes Griffin with differential learning rates.

5. **TabPFN fine-tuning** (`--tabpfn_finetune`): Uses `FinetunedTabPFNClassifier`/`Regressor` to fine-tune TabPFN on the extracted Griffin embeddings (not just frozen ICL).

6. **Linear probe fine-tuning** (`--probe_epochs N`): For ICL heads, optionally fine-tunes Griffin MPNN weights using a simple linear probe loss before extracting embeddings for the ICL head.

7. **Multi-task batch routing**: `LoaderWrapperTaskLLM` returns `taskname` per batch, so the correct prompt template and task type are used (fixes the original TODO at line 628).

8. **Rich task prompts**: System prompt includes task description, entity type, and structured question from `task_prompts.py`. Unseen tasks get auto-generated prompts for zero-shot generalization.

#### Training Flow by Head Type

**Gradient-based heads** (`default`, `llm`, `llm_mlp`):
```
for epoch in epochs:
    for batch in dataloader:
        loss = compute_loss(model, head, batch)
        loss.backward()
        optimizer.step()
    validate()
    test_on_best_valid()
```

**ICL-based heads** (`tabpfn`, `tabicl`):
```
# Phase 1 (optional): Fine-tune Griffin with linear probe
for epoch in probe_epochs:
    for batch in dataloader:
        loss = linear_probe(model(batch))
        loss.backward()  # gradient flows to Griffin
        optimizer.step()

# Phase 2: Extract & predict
for task in tasks:
    train_embs = extract_embeddings(model, train_data)
    icl_head.fit(train_embs, train_labels)
    valid_score = icl_head.predict(valid_embs)
    test_score = icl_head.predict(test_embs)
```

---

## Unchanged Files

| File | Purpose |
|------|---------|
| `hmaintask_combine.py` | Original Griffin training script — fully preserved, not modified |
| `hmaintask_completion.py` | Stage 1 pretraining (feature completion) — unchanged |
| `hmodel.py` | `GriffinMod` MPNN architecture — unchanged |
| `hdataset.py` | `Graph`, `Task`, `Node` data structures — unchanged |
| `hFloatEmb.py` | `SimpleRepeater`, `getfloatenc`, `getfloatdec` — unchanged |
| `metric.py` | `compute_metric` — unchanged |
| `task_names.yaml` | Task split definitions — unchanged |
| `hconfig.yaml` | Accelerate config — unchanged |

---

## Dependency Requirements

```
# Core (already required by Griffin)
torch, torch_geometric, accelerate, datasets, pyyaml, torchmetrics

# For --head llm or llm_mlp
transformers
peft  # only if --use_lora

# For --head tabpfn
tabpfn>=7.0.0  # pip install tabpfn

# For --head tabicl
tabicl>=2.0.0  # pip install tabicl
```

---

## Sprint v1 Bug Fixes

### Fix 1: Per-sample neighbor embedding assignment (HIGH)
**Before**: `_extract_llm_components` took the first K neighbors from the entire batch's flat `neighbor_mask` and replicated them to every sample. All samples received identical neighbor context.
**After**: Uses `edge_index` and `mapping` to build per-seed neighbor lists. Each sample receives its own 1-hop neighbors, projected and zero-padded to K tokens.

### Fix 2: Multi-GPU validation metric aggregation (MEDIUM)
**Before**: `avg_valid_metric` was only assigned inside `if accelerator.is_main_process` but referenced by all ranks. The `gather().mean()` diluted the real metric with zeros from non-main processes.
**After**: Initialized `avg_valid_metric = 0.0` before the conditional block. Uses `gathered[0]` (rank-0 value) instead of `.mean()` to avoid dilution.

### Fix 3: Projector activation (LOW)
**Before**: `GriffinToLLMProjector` used `nn.Sigmoid()` which squashes output to [0,1], limiting expressiveness when projecting to LLM embedding space.
**After**: Replaced with `nn.GELU()` to match Rel-LLM's standard MLP design and allow a wider output range.

### Fix 4: Auto-detect OutputMLP dimensions per task (MEDIUM)
**Before**: `--output_mlp_dim` defaulted to 1, which broke classification tasks and made multi-task training with mixed regression/classification impossible without manual override.
**After**: `OutputMLP` is now created as an `nn.ModuleDict` keyed by task name. Regression tasks get `out_channels=1`, classification tasks get `out_channels=num_class` from task metadata. `compute_loss` and `compute_output` select the correct per-task MLP.

### Fix 5: Reuse graph/task objects in ICL evaluation (LOW)
**Before**: `_run_icl_evaluation` called `Graph(args.dataset)` and `Task(args.dataset)` inside a loop, reloading the full graph from disk per task.
**After**: Accepts `graph` and `task_obj` parameters; callers pass the already-loaded objects from `main()`.

---

## Backward Compatibility

- `hmaintask_combine.py` is **untouched** and continues to work exactly as before.
- `hmaintask_combine_llm.py --head default` uses `LoaderWrapperTask` and the exact same `dec(model(*data)[mapping])` / `@y.T` logic — functionally identical to `hmaintask_combine.py`.
- All existing checkpoints load without changes.
- No modifications to `GriffinMod`, `Graph`, `Task`, or any preprocessing code.
