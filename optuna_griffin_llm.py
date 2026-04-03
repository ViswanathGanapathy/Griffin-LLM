"""
optuna_griffin_llm.py — Hyperparameter Search for Griffin + LLM Transfer

Wraps hmaintask_combine_llm_transfer.py with Optuna for automated
hyperparameter optimization.

Usage:
  python optuna_griffin_llm.py \
      datasets/joint-v65 logs/optuna log \
      --loadpath checkpoints/single-sft \
      --savepath checkpoints/optuna \
      --train_tasks rel-amazon-user-churn rel-amazon-item-churn \
      --test_tasks  rel-hm-user-churn \
      --n_trials 50 \
      --study_name griffin_llm_transfer \
      --hop 2 --fanout 20 --hiddim 512 --num_mp 4

  # Resume a previous study:
  python optuna_griffin_llm.py \
      ... \
      --study_name griffin_llm_transfer \
      --storage sqlite:///optuna_griffin.db \
      --n_trials 50

Searchable hyperparameters:
  - Learning rate
  - Weight decay
  - Batch size
  - Projector architecture (depth, bottleneck dim, dropout)
  - Number of MPNN layers
  - Fanout / hop
  - LoRA rank and alpha (if --search_lora)
  - Whether to freeze Griffin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from hdataset import Graph, Task
from hloaderwrapper import LoaderWrapperTask
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
from typing import Union
from metric import compute_metric
import yaml
import optuna
from optuna.trial import TrialState
import json
import time

from transformers import AutoModelForCausalLM, AutoTokenizer


# ═══════════════════════════════════════════════════════════════════════════
# LLM Components (identical to hmaintask_combine_llm_transfer.py)
# ═══════════════════════════════════════════════════════════════════════════

class GriffinToLLMProjector(nn.Module):
    """Configurable projector — Optuna searches over architecture."""
    def __init__(self, griffin_dim=512, llm_dim=2048,
                 bottleneck_dim=2048, num_layers=2, dropout=0.1):
        super().__init__()
        layers = []
        in_dim = griffin_dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, bottleneck_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = bottleneck_dim
        layers.extend([
            nn.Linear(in_dim, llm_dim),
            nn.Dropout(dropout),
        ])
        self.projector = nn.Sequential(*layers)

    def forward(self, x):
        return self.projector(x)


class LLMDecoder:
    """Cached LLM — loaded once, reused across trials."""
    _cache = {}  # class-level cache to avoid reloading LLM each trial

    @classmethod
    def get_or_create(cls, model_name, frozen=True, use_lora=False,
                      lora_r=8, lora_alpha=16, device="cuda"):
        cache_key = (model_name, frozen, use_lora, lora_r, lora_alpha)
        if cache_key not in cls._cache:
            cls._cache[cache_key] = cls(
                model_name, frozen, use_lora, lora_r, lora_alpha, device
            )
        return cls._cache[cache_key]

    def __init__(self, model_name, frozen=True, use_lora=False,
                 lora_r=8, lora_alpha=16, device="cuda"):
        print(f"[LLM] Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=False, padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True,
        ).to(device)

        self.llm_dim = self.model.config.hidden_size
        self.word_embedding = self.model.get_input_embeddings()
        self.device = device

        if frozen and not use_lora:
            for p in self.model.parameters():
                p.requires_grad = False
        elif use_lora:
            from peft import LoraConfig, get_peft_model
            config = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
                target_modules=["q_proj", "v_proj"],
                bias="none", task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, config)

        with torch.no_grad():
            pad_id = torch.tensor([self.tokenizer.pad_token_id], device=device)
            self.pad_embed = self.word_embedding(pad_id)

    def tokenize(self, text):
        return self.tokenizer(
            text, add_special_tokens=False, return_tensors="pt"
        ).input_ids[0].to(self.device)

    def embed_tokens(self, token_ids):
        return self.word_embedding(token_ids)


# ═══════════════════════════════════════════════════════════════════════════
# Training/eval functions (same core logic)
# ═══════════════════════════════════════════════════════════════════════════

def build_llm_inputs(graph_embeds, llm_decoder, task_name, labels=None,
                     task_type="regression"):
    ld = llm_decoder
    device = graph_embeds.device
    batch_size = graph_embeds.shape[0]
    is_training = labels is not None

    task_clean = task_name.replace("-", " ").replace("_", " ")
    if task_type == "regression":
        prompt = f"Predict the numerical value for '{task_clean}' based on the following database record: "
        instruction = " Output a single number."
    else:
        prompt = f"Predict the class for '{task_clean}' based on the following database record: "
        instruction = " Output the class index."

    prompt_ids = ld.tokenize(prompt)
    prompt_emb = ld.embed_tokens(prompt_ids)
    instr_ids = ld.tokenize(instruction)
    instr_emb = ld.embed_tokens(instr_ids)

    all_embeds, all_labels = [], []
    max_len = 0

    for i in range(batch_size):
        parts = [prompt_emb, graph_embeds[i], instr_emb]
        label_parts = [
            torch.full((prompt_emb.size(0),), -100, dtype=torch.long, device=device),
            torch.full((1,), -100, dtype=torch.long, device=device),
            torch.full((instr_emb.size(0),), -100, dtype=torch.long, device=device),
        ]
        if is_training:
            ans_text = f" {labels[i].item():.4f}" if task_type == "regression" else f" {labels[i].item()}"
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

    padded_embeds, padded_labels, attention_masks = [], [], []
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
            attention_masks.append(torch.ones(real_len, dtype=torch.long, device=device))

    return (
        torch.stack(padded_embeds),
        torch.stack(attention_masks),
        torch.stack(padded_labels),
    )


def compute_loss(model, projector, llm_decoder, data, task_name, task_type):
    label, y, mapping = data[-3:]
    data = data[:-3]
    seed_embeddings = model(*data)[mapping]
    graph_embeds = projector(seed_embeddings.float()).unsqueeze(1)
    inputs_embeds, attention_mask, labels = build_llm_inputs(
        graph_embeds, llm_decoder, task_name, labels=label, task_type=task_type,
    )
    with torch.cuda.amp.autocast(dtype=torch.float16):
        outputs = llm_decoder.model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            labels=labels, return_dict=True,
        )
    return outputs.loss


@torch.no_grad()
def compute_output(model, projector, llm_decoder, data, task_name,
                   task_type, max_new_tokens=16):
    label, y, mapping = data[-3:]
    data = data[:-3]
    seed_embeddings = model(*data)[mapping]
    graph_embeds = projector(seed_embeddings.float()).unsqueeze(1)
    inputs_embeds, attention_mask, _ = build_llm_inputs(
        graph_embeds, llm_decoder, task_name, labels=None, task_type=task_type,
    )
    batch_size = seed_embeddings.shape[0]
    device = seed_embeddings.device
    with torch.cuda.amp.autocast(dtype=torch.float16):
        generated_ids = llm_decoder.model.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=llm_decoder.tokenizer.pad_token_id,
        )
    if task_type == "regression" or y is None:
        predictions = []
        for i in range(batch_size):
            text = llm_decoder.tokenizer.decode(generated_ids[i], skip_special_tokens=True)
            try:
                val = float(text.strip().split()[-1])
            except:
                val = 0.0
            predictions.append(val)
        output = torch.tensor(predictions, device=device, dtype=torch.float).unsqueeze(1)
    else:
        num_classes = y.shape[0]
        predictions = []
        for i in range(batch_size):
            text = llm_decoder.tokenizer.decode(generated_ids[i], skip_special_tokens=True)
            try:
                val = int(text.strip().split()[-1])
            except:
                val = 0
            predictions.append(val)
        output = torch.full((batch_size, num_classes), -10.0, device=device)
        for i, p in enumerate(predictions):
            if 0 <= p < num_classes:
                output[i, p] = 10.0
    return output, label


def eval_task(model, projector, llm_decoder, dataset, batchsize, accelerator,
              metric, task_name, task_type):
    model.eval()
    projector.eval()
    dataset.rebuild_indice(accelerator)
    loader = DataLoader(
        dataset, shuffle=False, batch_size=1,
        collate_fn=lambda xlist: xlist[0],
        num_workers=4, persistent_workers=False,
    )
    loader = accelerator.prepare(loader)
    outputs, labels = [], []
    with torch.no_grad():
        for data in loader:
            output, label = compute_output(
                model, projector, llm_decoder, data, task_name, task_type,
            )
            if output.shape[0] < batchsize:
                padnum = batchsize - output.shape[0]
                if torch.is_floating_point(label):
                    label = torch.concat((label, torch.empty_like(label[[0]].expand(padnum)).fill_(torch.nan)), dim=0)
                else:
                    label = torch.concat((label, torch.empty_like(label[[0]].expand(padnum)).fill_(-1)), dim=0)
                output = torch.concat((output, torch.empty_like(output[[0]].expand(padnum, -1)).fill_(torch.nan)), dim=0)
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
    return None


def construct_dataset(graph, task, tasknames, split, batchsize, fanout, hop,
                      fewshotfanout, floatembmodel):
    return LoaderWrapperTask(
        graph,
        batch_size=batchsize,
        subgraphargs={
            "floatemb": floatembmodel,
            "fanout": fanout,
            "hop": hop,
        },
        shuffle=True if split == "train" else False,
        task=task,
        tasknames=tasknames,
        split=split,
        fewshotfanout=fewshotfanout,
    )


def resolve_task_names(task_list, task_metatask):
    if len(task_list) == 1:
        name = task_list[0]
        if name == "ALLTASK":
            return [t for t in task_metatask]
        elif name == "REGTASK":
            return [t for t in task_metatask if task_metatask[t]["task_type"] == "regression"]
        elif name == "RETTASK":
            return [t for t in task_metatask if task_metatask[t]["task_type"] == "retrieval"]
        elif name.startswith("EXCEPT__"):
            return [t for t in task_metatask if t != name[len("EXCEPT__"):]]
        elif name in ["commerce-1", "commerce-2", "others-1", "others-2"]:
            with open("task_names.yaml", "r") as f:
                return yaml.load(f, Loader=yaml.FullLoader)[name]
    return task_list


# ═══════════════════════════════════════════════════════════════════════════
# OPTUNA OBJECTIVE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def create_objective(args, graph, task, train_tasks, test_tasks,
                     task_type_dict, metric_dict, floatembmodel):
    """
    Returns an Optuna objective function that:
      1. Samples hyperparameters from the trial
      2. Builds model + projector with those HPs
      3. Trains for args.maxepoch epochs
      4. Returns the best cross-dataset validation metric
    """

    def objective(trial: optuna.Trial) -> float:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── SAMPLE HYPERPARAMETERS ──
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        wd = trial.suggest_float("wd", 1e-6, 1e-2, log=True)
        batchsize = trial.suggest_categorical("batchsize", [2, 4, 8])
        proj_dropout = trial.suggest_float("proj_dropout", 0.0, 0.3, step=0.05)
        proj_bottleneck = trial.suggest_categorical("proj_bottleneck", [512, 1024, 2048])
        proj_layers = trial.suggest_int("proj_layers", 1, 3)
        freeze_griffin = trial.suggest_categorical("freeze_griffin", [True, False])

        # Optionally search graph sampling params
        if args.search_graph:
            num_mp = trial.suggest_int("num_mp", 2, 6)
            fanout = trial.suggest_categorical("fanout", [10, 20, 30])
            hop = trial.suggest_int("hop", 1, 3)
        else:
            num_mp = args.num_mp
            fanout = args.fanout
            hop = args.hop

        # Optionally search LoRA params
        if args.search_lora:
            use_lora = trial.suggest_categorical("use_lora", [True, False])
            lora_r = trial.suggest_categorical("lora_r", [4, 8, 16]) if use_lora else 8
            lora_alpha = trial.suggest_categorical("lora_alpha", [8, 16, 32]) if use_lora else 16
        else:
            use_lora = args.use_lora
            lora_r = args.lora_r
            lora_alpha = args.lora_alpha

        trial_name = f"trial_{trial.number}"
        print(f"\n{'='*60}")
        print(f"Trial {trial.number}: lr={lr:.6f}, wd={wd:.6f}, bs={batchsize}")
        print(f"  proj: layers={proj_layers}, bottleneck={proj_bottleneck}, dropout={proj_dropout}")
        print(f"  freeze_griffin={freeze_griffin}, num_mp={num_mp}, fanout={fanout}, hop={hop}")
        if args.search_lora:
            print(f"  use_lora={use_lora}, lora_r={lora_r}, lora_alpha={lora_alpha}")
        print(f"{'='*60}")

        # ── Setup accelerator for this trial ──
        trial_logdir = osp.join(args.logdir, trial_name)
        os.makedirs(trial_logdir, exist_ok=True)
        tbconfig = ProjectConfiguration(project_dir=trial_logdir, logging_dir=trial_logdir)
        accelerator = Accelerator(log_with="tensorboard", project_config=tbconfig)
        accelerator.init_trackers(trial_name)

        try:
            # ── Build Griffin model ──
            model = GriffinMod(
                hiddim=args.hiddim, num_mp=num_mp,
                use_rev=args.use_rev, use_gate=args.use_gate
            )
            if args.loadpath is not None:
                accelerate.load_checkpoint_in_model(model, args.loadpath)

            # ── LLM (cached — loaded once across all trials) ──
            llm_decoder = LLMDecoder.get_or_create(
                model_name=args.llm_model,
                frozen=args.llm_frozen if not use_lora else False,
                use_lora=use_lora,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                device=accelerator.device,
            )

            # ── Projector with trial's architecture ──
            projector = GriffinToLLMProjector(
                griffin_dim=args.hiddim,
                llm_dim=llm_decoder.llm_dim,
                bottleneck_dim=proj_bottleneck,
                num_layers=proj_layers,
                dropout=proj_dropout,
            )

            # ── Optimizer ──
            trainable_params = []
            if freeze_griffin:
                for p in model.parameters():
                    p.requires_grad = False
            else:
                trainable_params.extend(model.parameters())
            trainable_params.extend(projector.parameters())
            if use_lora:
                trainable_params.extend(
                    p for p in llm_decoder.model.parameters() if p.requires_grad
                )

            optimizer = torch.optim.AdamW(
                [p for p in trainable_params if p.requires_grad],
                lr=lr, weight_decay=wd,
            )

            # ── Datasets ──
            train_dataset = construct_dataset(
                graph, task, train_tasks, "train", batchsize,
                fanout, hop, args.fewshotfanout, floatembmodel,
            )
            test_valid_dict = {
                tn: construct_dataset(
                    graph, task, [tn], "valid", batchsize,
                    fanout, hop, args.fewshotfanout, floatembmodel,
                )
                for tn in test_tasks
            }

            model, projector, optimizer = accelerator.prepare(model, projector, optimizer)

            # ── Training loop ──
            best_metric = -torch.inf
            step = 0

            for epoch in range(args.maxepoch):
                model.train()
                projector.train()
                train_dataset.rebuild_indice(accelerator)
                loader = DataLoader(
                    train_dataset, shuffle=True, batch_size=1,
                    collate_fn=lambda xlist: xlist[0],
                    num_workers=4, persistent_workers=False, pin_memory=True,
                )
                loader = accelerator.prepare(loader)

                epoch_loss = 0
                n_batches = 0
                for data in loader:
                    step += 1
                    optimizer.zero_grad()
                    current_task = train_tasks[0]
                    current_type = task_type_dict[current_task]
                    loss = compute_loss(
                        model, projector, llm_decoder, data,
                        task_name=current_task, task_type=current_type,
                    )
                    accelerator.backward(loss)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in trainable_params if p.requires_grad], 1.0
                    )
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_batches += 1

                avg_loss = epoch_loss / max(n_batches, 1)

                # ── Evaluate on test tasks (cross-dataset) ──
                if (epoch + 1) % args.eval_per_epoch == 0:
                    metrics = {}
                    for tn in test_tasks:
                        metrics[tn] = eval_task(
                            model, projector, llm_decoder,
                            test_valid_dict[tn], batchsize,
                            accelerator, metric_dict[tn],
                            tn, task_type_dict[tn],
                        )

                    if accelerator.is_main_process:
                        avg_metric = sum(
                            v for v in metrics.values() if v is not None
                        ) / len(metrics)
                        print(f"  Epoch {epoch}: loss={avg_loss:.4f}, "
                              f"transfer_metric={avg_metric:.4f}")

                        if avg_metric > best_metric:
                            best_metric = avg_metric

                        # Report to Optuna for pruning
                        trial.report(avg_metric, epoch)
                        if trial.should_prune():
                            print(f"  Trial {trial.number} pruned at epoch {epoch}")
                            raise optuna.TrialPruned()

        except optuna.TrialPruned:
            accelerator.end_training()
            raise
        except Exception as e:
            print(f"  Trial {trial.number} failed: {e}")
            accelerator.end_training()
            return float("-inf")

        accelerator.end_training()
        print(f"  Trial {trial.number} finished: best_metric={best_metric:.4f}")
        return best_metric

    return objective


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main(args):
    # ── Load data (shared across all trials) ──
    graph = Graph(args.dataset)
    task = Task(args.dataset)

    train_tasks = resolve_task_names(args.train_tasks, task.metatask)
    test_tasks = resolve_task_names(args.test_tasks, task.metatask)

    all_tasks = list(set(train_tasks + test_tasks))
    task_type_dict = {tn: task.metatask[tn].get("task_type", "regression") for tn in all_tasks}
    metric_dict = {tn: task.metatask[tn]["metric"] for tn in all_tasks}
    floatembmodel = SimpleRepeater(args.hiddim)

    print(f"\n{'='*60}")
    print(f"OPTUNA HYPERPARAMETER SEARCH")
    print(f"{'='*60}")
    print(f"Train tasks: {train_tasks}")
    print(f"Test tasks:  {test_tasks}")
    print(f"N trials:    {args.n_trials}")
    print(f"Study:       {args.study_name}")
    print(f"{'='*60}\n")

    # ── Create/load Optuna study ──
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,        # Don't prune first 5 trials
        n_warmup_steps=2,          # Don't prune before epoch 2
        interval_steps=1,
    )

    if args.storage:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            load_if_exists=True,
            direction="maximize",
            pruner=pruner,
        )
    else:
        study = optuna.create_study(
            study_name=args.study_name,
            direction="maximize",
            pruner=pruner,
        )

    # ── Run optimization ──
    objective = create_objective(
        args, graph, task, train_tasks, test_tasks,
        task_type_dict, metric_dict, floatembmodel,
    )

    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        gc_after_trial=True,       # Free GPU memory between trials
    )

    # ── Print results ──
    print(f"\n{'='*60}")
    print(f"OPTUNA SEARCH COMPLETE")
    print(f"{'='*60}")

    print(f"\nBest trial:")
    best = study.best_trial
    print(f"  Value (metric): {best.value:.4f}")
    print(f"  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    # Save results
    results_path = osp.join(args.logdir, "optuna_results.json")
    results = {
        "best_value": best.value,
        "best_params": best.params,
        "n_trials": len(study.trials),
        "study_name": args.study_name,
        "train_tasks": train_tasks,
        "test_tasks": test_tasks,
        "all_trials": [
            {
                "number": t.number,
                "value": t.value if t.value is not None else None,
                "params": t.params,
                "state": str(t.state),
            }
            for t in study.trials
        ],
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Print top 5 trials
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value or float("-inf"), reverse=True)
    print(f"\nTop 5 trials:")
    for t in completed[:5]:
        print(f"  Trial {t.number}: {t.value:.4f}")
        for k, v in t.params.items():
            print(f"    {k}: {v}")

    # ── Generate the best run command ──
    bp = best.params
    print(f"\n{'='*60}")
    print("To train with best hyperparameters:")
    print(f"{'='*60}")
    cmd = (
        f"python hmaintask_combine_llm_transfer.py \\\n"
        f"    {args.dataset} {args.logdir}/best_run log \\\n"
        f"    --loadpath {args.loadpath} \\\n"
        f"    --savepath {args.savepath}/best \\\n"
        f"    --train_tasks {' '.join(train_tasks)} \\\n"
        f"    --test_tasks {' '.join(test_tasks)} \\\n"
        f"    --lr {bp['lr']:.6f} --wd {bp['wd']:.6f} \\\n"
        f"    --batchsize {bp['batchsize']} \\\n"
        f"    --hiddim {args.hiddim} \\\n"
        f"    --num_mp {bp.get('num_mp', args.num_mp)} \\\n"
        f"    --fanout {bp.get('fanout', args.fanout)} \\\n"
        f"    --hop {bp.get('hop', args.hop)} \\\n"
        f"    --maxepoch {args.maxepoch * 2}"  # train longer with best HPs
    )
    if bp.get("freeze_griffin", False):
        cmd += " \\\n    --freeze_griffin"
    print(cmd)


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

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("logdir", type=str)
    parser.add_argument("logname", type=str)

    # Tasks
    parser.add_argument("--train_tasks", type=str, nargs="+", required=True)
    parser.add_argument("--test_tasks", type=str, nargs="+", required=True)

    # Fixed args (not searched)
    parser.add_argument("--loadpath", type=str, default=None)
    parser.add_argument("--savepath", type=str, default="checkpoints/optuna")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hiddim", type=int, default=512)
    parser.add_argument("--use_rev", type=str2bool, default=True)
    parser.add_argument("--use_gate", type=str2bool, default=False)
    parser.add_argument("--fewshotfanout", type=int, default=3)
    parser.add_argument("--maxepoch", type=int, default=5,
                        help="Epochs per trial (keep low for search speed)")
    parser.add_argument("--eval_per_epoch", type=int, default=1)
    parser.add_argument("--llm_model", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--llm_frozen", action="store_true", default=True)

    # Default search ranges (can be overridden)
    parser.add_argument("--num_mp", type=int, default=4)
    parser.add_argument("--fanout", type=int, default=20)
    parser.add_argument("--hop", type=int, default=2)
    parser.add_argument("--use_lora", action="store_true", default=False)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)

    # Optuna settings
    parser.add_argument("--n_trials", type=int, default=50,
                        help="Number of Optuna trials")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Max time in seconds for all trials")
    parser.add_argument("--study_name", type=str, default="griffin_llm_transfer",
                        help="Optuna study name (for persistence)")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URL (e.g., sqlite:///optuna.db)")

    # What to search
    parser.add_argument("--search_lora", action="store_true", default=False,
                        help="Include LoRA params in search space")
    parser.add_argument("--search_graph", action="store_true", default=False,
                        help="Include num_mp/fanout/hop in search space")

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.savepath, exist_ok=True)
    main(args)
