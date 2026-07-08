# Griffin + TabPFN/TabICL: Data, Training, Evaluation Handoff

Quickstart for a collaborator who has their own multi-table (relational)
dataset and wants to:

1. Convert it to Griffin's on-disk format
2. Train a Griffin encoder on it
3. Evaluate the trained encoder with TabPFN or TabICL as the ICL head

This doc is self-contained. All commands assume you are in the Griffin
repo root after cloning `smpnn-ablations`.

---

## 0. TL;DR — what you'll produce

You'll end up with:

- A `datasets/<your-dataset>/` directory in Griffin's canonical format
- A trained encoder at `checkpoints/<run-name>/best_checkpoint/`
- Test metrics logged to stdout and TensorBoard, showing AUROC / RMSE per
  task under either TabPFN or TabICL as the ICL head

Total wall time on a single A100 80GB:
- Data conversion: 30 min – 4 h (depends on # of tables and rows)
- Training: 3–6 h per 20-epoch run
- ICL evaluation: 5–15 min per (task × head) cell

---

## 1. Environment setup

```bash
# Conda env named "griffin"
conda create -n griffin python=3.11 -y
conda activate griffin

# Core deps
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric torch-scatter accelerate transformers
pip install datasets pandas pyyaml sentence-transformers

# ICL heads (either or both)
pip install tabpfn==8.0.3      # TabPFN v3 backend
pip install tabicl              # TabICL v2 backend
# Optional: pip install tabicl[finetune]  # only if fine-tuning TabICL

# Misc
pip install safetensors seaborn matplotlib tensorboard
```

TabPFN v3 requires a free license token on first run — follow the prompt.
Register at https://priorlabs.ai.

Verify GPU + CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 2. Get the code

```bash
git clone https://github.com/ViswanathGanapathy/Griffin-LLM.git Griffin
cd Griffin
git checkout smpnn-ablations
```

---

## 3. What Griffin expects on disk

Every dataset that Griffin trains on has this layout at
`datasets/<name>/`:

```
datasets/<name>/
├── metanode.yaml          # per-nodetype: num rows + feature column names + is_target flag
├── metaadj.yaml           # per-nodetype: incoming / outgoing edge type names
├── metatask.yaml          # per-task: target_type, task_type, num_class, split sizes, metric, hastimestamp
├── edgenameemb.pt         # dict[edge_type_name → 512-d Nomic embedding]
├── featnameemb.pt         # dict[feature_column_name → 512-d Nomic embedding]
├── tasknameemb.pt         # dict[task_name → 512-d Nomic embedding]
├── node/<nodetype>/feat/  # HuggingFace Arrow dataset (torch format) with row features
├── node/<nodetype>/textemb/  # (optional) text-column embeddings
├── edge/<nodetype>/adj/   # adjacency arrays per source node; supports temporal edges
└── task/<taskname>/       # HuggingFace Arrow dataset with (nodeidx, label, timestamp) per split
```

Loaded by `Graph` and `Task` in [hdataset.py](../../hdataset.py). Files at
[hdataset.py:140-152](../../hdataset.py#L140-L152) load `metanode.yaml`,
`metaadj.yaml`, `edgenameemb.pt`, `featnameemb.pt`. Tasks load from
[hdataset.py:310-311](../../hdataset.py#L310-L311).

### Schema of the meta files

**metanode.yaml** — one entry per node type:
```yaml
<NodeType>:
  num: 12345                       # row count
  is_target: false                 # true only for task-target node types
  feat:                            # feature column names (must match keys in featnameemb.pt)
    - <NodeType>___<col1>
    - <NodeType>___<col2>___Griffin_text_<text_col>   # text columns get a Griffin_text_ prefix
```

**metaadj.yaml** — one entry per node type, listing named edge types:
```yaml
<NodeType>:
  in:                              # incoming edges (tail role)
    - "tail of <SrcType>:<SrcType>-<edge_name>:<NodeType>"
  out:                             # outgoing edges (head role)
    - "head of <NodeType>:<NodeType>-<edge_name>:<TgtType>"
```

Edge names must appear as keys in `edgenameemb.pt`.

**metatask.yaml** — one entry per task:
```yaml
<task-name>:
  target_type: <NodeType>          # what node type the task predicts on
  task_type: retrieval | regression # retrieval = classification (any # classes); regression = MSE
  num_class: 12                    # classes (1 for regression)
  seed_type: <NodeType>_<label_col># the label column embedded as feat
  metric: retrieval_auroc | rmse
  masked_feat: []                  # columns to mask out at inference (e.g. the label column itself)
  hastimestamp: true               # if true, split-by-time is enforced
  split:                           # [train_count, valid_count, test_count]
    - 192106
    - 10672
    - 10673
  extra_feat: []                   # rarely used
```

See [datasets/joint-v65/metatask.yaml](../../datasets/joint-v65/metatask.yaml)
for 32 real examples.

---

## 4. Converting your multi-table data → Griffin format

The end-to-end ETL is 4 scripts run in sequence. They all read from a
"raw" src directory and write to the "processed" dst directory. All 4
converters live in the repo root.

### Step 4.1 — Prepare a raw `srcpath/` with a schema-first `metadata.yaml`

Griffin's converters ingest a schema-first source layout, not raw CSVs.
Your `srcpath/` should look like:

```
<your-raw-data>/
├── metadata.yaml       # schema descriptor (see below)
├── <col1>.npy          # column arrays (int64, float32, or bytes for text)
├── <col2>.npy
├── ...
└── <task_split>.npy    # per-task train/valid/test index arrays + labels
```

`metadata.yaml` describes both the **graph** (nodes, features, edges) and
the **tasks** (with node-pair references + labels). A minimal example:

```yaml
graph:
  nodes:
    - type: User
      num: 100000
    - type: Product
      num: 5000
  feature_data:
    - domain: node
      node_type: User
      name: User___signup_year
      path: user_signup_year.npy       # int64 array, shape (100000,)
      dtype: int64
    - domain: node
      node_type: Product
      name: Product___price
      path: product_price.npy          # float32 array
      dtype: float32
    - domain: node
      node_type: Product
      name: Product___Griffin_text_title       # text columns get Griffin_text_ prefix
      path: product_title_embeddings.npy       # pre-computed sentence embeddings (N, D)
      dtype: float32
      is_text: true
  edges:
    - name: User-purchase:Product
      head_type: User
      tail_type: Product
      head_path: purchase_user_ids.npy         # int64, length E
      tail_path: purchase_product_ids.npy      # int64, length E
      timestamp_path: purchase_timestamp.npy   # optional, int64 unix seconds
tasks:
  - name: my-churn-task
    target_type: User
    task_type: retrieval                       # or regression
    num_class: 2
    metric: retrieval_auroc                    # or rmse
    hastimestamp: true
    node_pairs:                                # (nodeidx, label) triples per split
      - path: churn_train.npy                  # shape (N_train, 2 or 3): [nodeidx, label, timestamp?]
      - path: churn_valid.npy
      - path: churn_test.npy
```

**Precondition:** if `hastimestamp: true`, your node feature arrays must
be **sorted by timestamp**. Griffin's fewshot sampler modes past-only
selection off row-index (see
[hdataset.py:291-303](../../hdataset.py#L291-L303)), so a wrong order
silently leaks labels.

### Step 4.2 — Run the four converters, in order

```bash
export SRC=path/to/your-raw-data
export DST=datasets/<your-dataset-name>

# 1) Convert node features (metanode.yaml + node/<type>/feat/)
python dataconverter.py "$SRC" "$DST" --ncpu 16

# 2) Convert edges (metaadj.yaml + edge/<type>/adj/)
python dataconverteredge.py "$SRC" "$DST" --ncpu 4

# 3) Convert tasks (metatask.yaml + task/<taskname>/)
python dataconvertertask.py "$SRC" "$DST" --ncpu 16

# 4) Post-process: embed edge/feat/task names with Nomic, write *nameemb.pt
python dataconverterpost.py "$DST" --ncpu 1
```

Step 4 requires GPU (loads `nomic-embed-text-v1.5` from HuggingFace on
`cuda:0`). See
[dataconverterpost.py:13-42](../../dataconverterpost.py#L13-L42).

### Step 4.3 — Sanity check the output

```bash
ls "$DST"
# should show: metanode.yaml metaadj.yaml metatask.yaml edgenameemb.pt
# featnameemb.pt tasknameemb.pt node/ edge/ task/

python -c "
from hdataset import Graph, Task
g = Graph('$DST')
t = Task('$DST')
print('nodes:', list(g.metanode.keys())[:5])
print('tasks:', list(t.metatask.keys()))
"
```

If both classes construct without error, the data format is valid.

---

## 5. Register your task family (optional but recommended)

Griffin uses `task_names.yaml` at the repo root to define task groups
(e.g. `others-1`, `commerce-1`). Add yours:

```yaml
# Edit task_names.yaml, append:
my-family:
  - my-churn-task
  - my-other-task
```

Then you can pass `--tasks my-family` to training. Alternatively, pass
individual task names directly: `--tasks my-churn-task`.

Special aliases (defined in
[hmaintask_combine.py:175-178](../../hmaintask_combine.py#L175-L178)):

- `ALLTASK` — every task in `metatask.yaml`
- `RETTASK` — all retrieval (classification) tasks
- `REGTASK` — all regression tasks

---

## 6. Training

### 6.1 — Baseline Griffin (vanilla)

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch hmaintask_combine.py \
    datasets/<your-dataset-name> \
    logs/griffin-vanilla-run \
    griffin-vanilla-run \
    --tasks my-family \
    --seed 42 \
    --num_mp 4 \
    --hiddim 512 --use_rev True --use_gate True \
    --maxepoch 20 --batchsize 256 \
    --lr 3e-4 --wd 4e-4 \
    --hop 2 --fanout 20 --fewshotfanout 3 \
    --eval_per_epoch 1 \
    --savepath checkpoints/griffin-vanilla-run \
    2>&1 | tee logs/griffin-vanilla-run.log
```

Positional args are `(dataset_path, log_dir, log_name)`. Best checkpoint
lands at `checkpoints/griffin-vanilla-run/best_checkpoint/`.

### 6.2 — SMPNN-Griffin (recommended: our paper variant)

Two flags flip vanilla → SMPNN: `--use_smpnn` and `--alpha_init`.

```bash
accelerate launch hmaintask_combine.py \
    datasets/<your-dataset-name> \
    logs/griffin-smpnn-run \
    griffin-smpnn-run \
    --tasks my-family \
    --seed 42 \
    --num_mp 6 --use_smpnn --alpha_init 1e-2 \
    --hiddim 512 --use_rev True --use_gate True \
    --maxepoch 20 --batchsize 256 \
    --lr 3e-4 --wd 4e-4 \
    --hop 2 --fanout 20 --fewshotfanout 3 \
    --eval_per_epoch 1 \
    --savepath checkpoints/griffin-smpnn-run \
    --log_alpha_every 2 \
    2>&1 | tee logs/griffin-smpnn-run.log
```

**Recommended `alpha_init` values:**
- `1e-6` — paper-default (D1). Safe start but slow to activate.
- `1e-2` — our winner on cross-task transfer (D3). Recommended for new datasets.
- `1e-4` — middle (D2). Good if `1e-2` is unstable.

See [sprints/v2/ARCHITECTURE_VARIANTS.md](ARCHITECTURE_VARIANTS.md) for
the full variant catalog and their trade-offs.

### 6.3 — Training-time OOM

If OOM: drop `--batchsize` (256 → 128 → 64) then `--fanout` (20 → 10) then
`--num_mp` (6 → 4). Keep `--hop 2`.

---

## 7. Evaluation with TabPFN as the ICL head

TabPFN v3 is our default. Runs the trained Griffin encoder over the test
set, then feeds the (test embedding, past training embeddings + labels)
into TabPFN for in-context prediction.

```bash
python hmaintask_combine_llm.py \
    datasets/<your-dataset-name> \
    logs/griffin-smpnn-eval-tabpfn \
    griffin-smpnn-eval-tabpfn \
    --head tabpfn \
    --tasks rel-f1-driver-position \
    --eval_tasks my-family \
    --loadpath checkpoints/griffin-smpnn-run/best_checkpoint \
    --seed 42 \
    --num_mp 6 --use_smpnn --alpha_init 1e-2 \
    --hiddim 512 --use_rev True --use_gate True \
    --hop 2 --fanout 20 --fewshotfanout 3 \
    --batchsize 256 \
    --output_mlp_dim 1 --no_target_normalize \
    --no_icl_projection --probe_epochs 0 \
    --icl_n_estimators 8 --icl_max_context 30000 \
    --tabpfn_version v3 --tabpfn_fit_mode fit_with_cache \
    --tabpfn_inference_precision autocast \
    --tabpfn_inference_config {\"MAX_NUMBER_OF_SAMPLES\":50000,\"MAX_NUMBER_OF_FEATURES\":600} \
    --savepath checkpoints/griffin-smpnn-eval-tabpfn \
    2>&1 | tee logs/griffin-smpnn-eval-tabpfn.log
```

**Key eval flags** (see
[hmaintask_combine_llm.py:2716-2823](../../hmaintask_combine_llm.py#L2716-L2823)):
- `--head tabpfn` — pick TabPFN
- `--eval_tasks` — which task family to evaluate (can differ from `--tasks`)
- `--loadpath` — path to Griffin best_checkpoint from training
- `--no_icl_projection` — feed 512-d Griffin embeddings raw to TabPFN
  (skips the learned projection layer). Recommended for new datasets — the
  projection adds a probe-training step (`--probe_epochs`) that
  underperforms raw features in most cases.
- `--probe_epochs 0` — do not fine-tune Griffin during eval
- `--icl_max_context 30000` — TabPFN sees at most 30K past examples per
  test query (drop to 20000 if OOM)
- `--icl_n_estimators 8` — ensemble size
- `--tabpfn_inference_config` — cap TabPFN feature dim to 600 and sample
  dim to 50K (matches TabPFN's own limits; JSON must have no whitespace)

**Test metrics** print as `test_metric/<task_name>` lines in the log — 6
lines per eval (one per task in the family × head).

### 7.1 — TabPFN fine-tuning (optional)

If you want to fine-tune TabPFN on your data instead of pure in-context:

```bash
    --tabpfn_finetune --tabpfn_finetune_epochs 30 --tabpfn_finetune_lr 2e-5
```

---

## 8. Evaluation with TabICL as the ICL head

Same script, different head. TabICL is faster and lighter-weight but
supports smaller context (10K vs TabPFN's 30K).

```bash
python hmaintask_combine_llm.py \
    datasets/<your-dataset-name> \
    logs/griffin-smpnn-eval-tabicl \
    griffin-smpnn-eval-tabicl \
    --head tabicl \
    --tasks rel-f1-driver-position \
    --eval_tasks my-family \
    --loadpath checkpoints/griffin-smpnn-run/best_checkpoint \
    --seed 42 \
    --num_mp 6 --use_smpnn --alpha_init 1e-2 \
    --hiddim 512 --use_rev True --use_gate True \
    --hop 2 --fanout 20 --fewshotfanout 3 \
    --batchsize 256 \
    --output_mlp_dim 1 --no_target_normalize \
    --no_icl_projection --probe_epochs 0 \
    --icl_n_estimators 8 --icl_max_context 10000 \
    --tabicl_checkpoint_version v2 \
    --savepath checkpoints/griffin-smpnn-eval-tabicl \
    2>&1 | tee logs/griffin-smpnn-eval-tabicl.log
```

**TabICL-specific flags** (see
[hmaintask_combine_llm.py:2751-2805](../../hmaintask_combine_llm.py#L2751-L2805)):
- `--tabicl_checkpoint_version v2` — latest weights (v1 also available)
- `--tabicl_model_path <path.ckpt>` — override with a local checkpoint
- `--tabicl_finetune` (optional) — fine-tune with epoch/lr/patience flags
- `--tabicl_finetune_freeze_col / freeze_row / freeze_icl` — parameter-efficient FT

---

## 9. Reading results

Every eval prints lines like:

```
test_metric/my-churn-task 0.842
test_metric/my-other-task 0.611
```

Aggregate multi-task or multi-seed runs with:

```bash
python collect_smpnn_results.py \
    --logs logs/griffin-smpnn-eval-*.log \
    --out griffin_results.csv
```

The CSV has one row per (task, head, run) with per-task metrics.

---

## 10. Common pitfalls

| Symptom | Cause / Fix |
|---|---|
| `KeyError: <edge_type>` during training | edge_type in `metaadj.yaml` not in `edgenameemb.pt`. Rerun `dataconverterpost.py` after adding it to `edgenameemb.pt`. |
| Loss NaN in first epoch | Feature array has NaN. Fix in source `.npy`; there is no NaN mask. |
| `TabPFNValidationError: All features are constant` | Griffin encoder collapsed to constant output — check that training loss decreased. Also check `--no_target_normalize` is set when target has natural scale. |
| Wildly wrong metrics on eval | The `metatask.yaml` `metric:` field must match task type (retrieval → `retrieval_auroc`; regression → `rmse`). |
| Label leakage suspected | Verify node arrays are sorted by timestamp when `hastimestamp: true`. Fewshot sampler assumes this ([hdataset.py:288](../../hdataset.py#L288)). |
| OOM during eval | Drop `--icl_max_context` (30000 → 20000 for TabPFN, 10000 → 5000 for TabICL) or `--batchsize`. |
| Training much slower than expected | Bump `--num_workers` in `hmaintask_combine.py` DataLoader (default 16). |
| TabPFN JSON arg parsed wrong | The `--tabpfn_inference_config` JSON MUST have no whitespace and outer quotes preserved. |

---

## 11. Reference: which files matter

| File | Role |
|---|---|
| [hdataset.py](../../hdataset.py) | `Graph` and `Task` classes; on-disk format contract. |
| [hloaderwrapper.py](../../hloaderwrapper.py) | Batch construction, fewshot sampling, subgraph merging. |
| [hmodel_smpnn.py](../../hmodel_smpnn.py) | SMPNN-Griffin architecture (`GriffinMod`, `RMPNN`, `LinearGlobalAttention`). |
| [hmaintask_combine.py](../../hmaintask_combine.py) | Training entry point. |
| [hmaintask_combine_llm.py](../../hmaintask_combine_llm.py) | Eval entry point with ICL heads. |
| [tabular_heads.py](../../tabular_heads.py) | TabPFN and TabICL head implementations. |
| [dataconverter*.py](../../) | Four-stage ETL (node → edge → task → nameemb). |
| [task_names.yaml](../../task_names.yaml) | Task family definitions. |
| [sprints/v2/ARCHITECTURE_VARIANTS.md](ARCHITECTURE_VARIANTS.md) | 11 architecture variants + their trade-offs. |
| [sprints/v2/SMPNN_GRIFFIN_DESIGN.md](SMPNN_GRIFFIN_DESIGN.md) | Paper design doc + full results matrix. |

---

## 12. When you're done — sending back

Push results to the branch (if you have push access):

```bash
git add griffin_results.csv logs/griffin-*
git commit -m "Multi-table dataset results (SMPNN vs vanilla, TabPFN + TabICL)"
git push github smpnn-ablations
```

Or email the CSV + selected log files to the project lead.
