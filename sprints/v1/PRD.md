# Sprint v1 — PRD: Griffin+LLM Bug Fixes & Experiment Readiness

## Overview
Fix 5 correctness and robustness issues in the Griffin+LLM integration (`hmaintask_combine_llm.py`) discovered during code review. These bugs block reliable training and evaluation of the LLM heads — especially the `llm_mlp` head that implements the Rel-LLM-style flow: **Griffin MPNN → Projection → Frozen LLM → Output MLP**. After fixes, the codebase will be experiment-ready for ablation studies across all five head types.

## Goals
- Per-sample neighbor embeddings work correctly (each sample gets its own 1-hop neighbors, not a shared pool)
- Multi-GPU validation metric aggregation is correct (no uninitialized variable on non-main processes)
- Projector activation matches Rel-LLM best practices (GELU instead of Sigmoid)
- Classification tasks auto-detect output dimension from task metadata (no manual `--output_mlp_dim`)
- ICL evaluation reuses in-memory graph/task objects (no redundant disk I/O)

## User Stories
- As a researcher, I want neighbor tokens to represent each sample's actual neighbors, so that the LLM receives correct relational context per entity
- As a researcher, I want multi-GPU training to produce correct validation metrics, so I can trust early stopping and checkpoint selection
- As a researcher, I want to run multi-task experiments mixing regression and classification without manually setting output dimensions per task

## Technical Architecture
- **Graph Encoder**: GriffinMod MPNN (unchanged, `hmodel.py`)
- **Projection**: `GriffinToLLMProjector` — MLP bottleneck (512 → 1024 → llm_dim)
- **LLM**: Frozen HuggingFace causal LM (Llama-3.2-1B default) via `LLMDecoder`
- **Output**: `OutputMLP` for `llm_mlp` head, text generation for `llm` head
- **Data**: `LoaderWrapperTaskLLM` provides per-batch metadata (task name, neighbor mask, feature names)

```
Data Flow (llm_mlp head):
  Relational DB → Heterogeneous Subgraph → GriffinMod MPNN
       → [B, 512] seed embeddings
       → GriffinToLLMProjector → [B, 1, llm_dim] soft tokens
       → (+ neighbor tokens + text prompt)
       → Frozen LLM → hidden states
       → Pool (last / entity / graph) → OutputMLP → prediction
```

## Out of Scope (v2+)
- New head types or model architectures
- Hyperparameter tuning (Optuna experiments)
- Cross-dataset transfer experiments (`hmaintask_combine_llm_transfer.py`)
- Training script execution or experiment runs (this sprint is fixes only)
- Changes to `hmodel.py`, `hdataset.py`, or the original `hmaintask_combine.py`

## Dependencies
- Pretrained Griffin checkpoint (existing)
- HuggingFace model access for LLM (Llama-3.2-1B or similar)
- Existing codebase at `/home/pviswanath/Griffin`
