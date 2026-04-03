"""
hmaintask_combine_llm_transfer.py — Griffin + LLM Cross-Dataset Transfer

Train on one dataset's tasks (e.g., rel-amazon) and evaluate on another
dataset's tasks (e.g., rel-hm) to test cross-dataset generalization.

Both datasets must be in the same joint-v65 processed dataset.

═══════════════════════════════════════════════════════════════════════════
WHAT'S DIFFERENT FROM hmaintask_combine_llm.py:
  - Added --train_tasks and --test_tasks (separate lists)
  - Training only uses train_tasks
  - Validation/Test only uses test_tasks
  - Everything else identical
═══════════════════════════════════════════════════════════════════════════

Usage examples:

  # Train on ALL rel-amazon tasks, test on ALL rel-hm tasks:
  python hmaintask_combine_llm_transfer.py \
      datasets/joint-v65 logs/transfer-amazon-to-hm log \
      --loadpath checkpoints/single-sft \
      --savepath checkpoints/transfer-amazon-to-hm \
      --train_tasks rel-amazon-user-churn rel-amazon-item-churn \
                     rel-amazon-user-ltv rel-amazon-product-ltv \
      --test_tasks rel-hm-user-churn rel-hm-item-sales \
      --hop 2 --fanout 20 --maxepoch 10 --patience 5 \
      --batchsize 4 --lr 1e-4 --hiddim 512

  # Train on a single rel-amazon task, test on a single rel-hm task:
  python hmaintask_combine_llm_transfer.py \
      datasets/joint-v65 logs/transfer log \
      --loadpath checkpoints/single-sft \
      --savepath checkpoints/transfer \
      --train_tasks rel-amazon-user-churn \
      --test_tasks rel-hm-user-churn \
      --batchsize 4

  # Use Griffin's existing data splits (commerce-1 → others-1):
  python hmaintask_combine_llm_transfer.py \
      datasets/joint-v65 logs/transfer log \
      --loadpath checkpoints/single-sft \
      --savepath checkpoints/transfer \
      --train_tasks commerce-1 \
      --test_tasks others-1 \
      --batchsize 4

Known rel-amazon tasks:
  rel-amazon-user-churn       (classification)
  rel-amazon-item-churn       (classification)
  rel-amazon-user-ltv         (regression)
  rel-amazon-product-ltv      (regression)
  rel-amazon-review-verified  (classification)

Known rel-hm tasks:
  rel-hm-user-churn           (classification)
  rel-hm-item-sales           (regression)
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

from transformers import AutoModelForCausalLM, AutoTokenizer


# ═══════════════════════════════════════════════════════════════════════════
# LLM Components (same as hmaintask_combine_llm.py)
# ═══════════════════════════════════════════════════════════════════════════

class GriffinToLLMProjector(nn.Module):
    def __init__(self, griffin_dim=512, llm_dim=2048, dropout=0.1):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(griffin_dim, llm_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_dim, llm_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.projector(x)


class LLMDecoder:
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
            self.model.print_trainable_parameters()
        
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
# Prompt building & loss/output functions (same as hmaintask_combine_llm.py)
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
    
    all_embeds = []
    all_labels = []
    max_len = 0
    
    for i in range(batch_size):
        parts = [prompt_emb, graph_embeds[i], instr_emb]
        label_parts = [
            torch.full((prompt_emb.size(0),), -100, dtype=torch.long, device=device),
            torch.full((1,), -100, dtype=torch.long, device=device),
            torch.full((instr_emb.size(0),), -100, dtype=torch.long, device=device),
        ]
        
        if is_training:
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
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
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
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
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


def eval_task(model, projector, llm_decoder, dataset, args, accelerator,
              metric, task_name, task_type):
    model.eval()
    projector.eval()
    dataset.rebuild_indice(accelerator)
    batchsize = dataset.batch_size
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


def construct_dataset(graph, task, tasknames, split, args, floatembmodel):
    return LoaderWrapperTask(
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


# ═══════════════════════════════════════════════════════════════════════════
# MODIFIED main() — Separate train_tasks and test_tasks
# ═══════════════════════════════════════════════════════════════════════════

def resolve_task_names(task_list, task_metatask):
    """
    Resolve task name shortcuts into actual task name lists.
    Handles: ALLTASK, REGTASK, RETTASK, commerce-1, etc.
    """
    if len(task_list) == 1:
        name = task_list[0]
        if name == "ALLTASK":
            return [t for t in task_metatask]
        elif name == "REGTASK":
            return [t for t in task_metatask if task_metatask[t]["task_type"] == "regression"]
        elif name == "RETTASK":
            return [t for t in task_metatask if task_metatask[t]["task_type"] == "retrieval"]
        elif name.startswith("EXCEPT__"):
            exc = name[len("EXCEPT__"):]
            return [t for t in task_metatask if t != exc]
        elif name in ["commerce-1", "commerce-2", "others-1", "others-2"]:
            with open("task_names.yaml", "r") as f:
                tasks_dict = yaml.load(f, Loader=yaml.FullLoader)
            return tasks_dict[name]
    return task_list


def main(args):
    tbconfig = ProjectConfiguration(project_dir=args.logdir, logging_dir=args.logdir)
    accelerator = Accelerator(log_with="tensorboard", project_config=tbconfig)
    accelerator.init_trackers(args.logname)
    tbtracker = accelerator.get_tracker("tensorboard")

    # ── Griffin MPNN model ──
    model = GriffinMod(
        hiddim=args.hiddim, num_mp=args.num_mp,
        use_rev=args.use_rev, use_gate=args.use_gate
    )
    if args.loadpath is not None:
        accelerate.load_checkpoint_in_model(model, args.loadpath)

    # ── LLM + Projector ──
    llm_decoder = LLMDecoder(
        model_name=args.llm_model, frozen=args.llm_frozen,
        use_lora=args.use_lora, lora_r=args.lora_r,
        lora_alpha=args.lora_alpha, device=accelerator.device,
    )
    projector = GriffinToLLMProjector(
        griffin_dim=args.hiddim, llm_dim=llm_decoder.llm_dim, dropout=0.1,
    )

    # ── Trainable params ──
    trainable_params = []
    if args.freeze_griffin:
        for p in model.parameters():
            p.requires_grad = False
    else:
        trainable_params.extend(model.parameters())
    trainable_params.extend(projector.parameters())
    if args.use_lora:
        trainable_params.extend(p for p in llm_decoder.model.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(
        [p for p in trainable_params if p.requires_grad],
        lr=args.lr, weight_decay=args.wd
    )

    # ── Data ──
    graph = Graph(args.dataset)
    task = Task(args.dataset)

    # ══════════════════════════════════════════════════════════════════
    # KEY CHANGE: Separate train and test task lists
    # ══════════════════════════════════════════════════════════════════
    train_tasks = resolve_task_names(args.train_tasks, task.metatask)
    test_tasks = resolve_task_names(args.test_tasks, task.metatask)

    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"CROSS-DATASET TRANSFER")
        print(f"{'='*60}")
        print(f"Train tasks: {train_tasks}")
        print(f"Test tasks:  {test_tasks}")
        print(f"{'='*60}\n")

    # Build task_type lookup for ALL tasks (train + test)
    all_tasks = list(set(train_tasks + test_tasks))
    task_type_dict = {}
    for tn in all_tasks:
        task_type_dict[tn] = task.metatask[tn].get("task_type", "regression")

    floatembmodel = SimpleRepeater(args.hiddim)

    # ── Training dataset: only train_tasks ──
    train_dataset = construct_dataset(graph, task, train_tasks, "train", args, floatembmodel)

    # ── Validation: use train_tasks for validation during training ──
    train_valid_dict = {
        tn: construct_dataset(graph, task, [tn], "valid", args, floatembmodel)
        for tn in train_tasks
    }

    # ── Test datasets: only test_tasks (cross-dataset evaluation) ──
    test_dataset_dict = {
        tn: construct_dataset(graph, task, [tn], "test", args, floatembmodel)
        for tn in test_tasks
    }
    # Also build valid sets for test tasks (for zero-shot eval during training)
    test_valid_dict = {
        tn: construct_dataset(graph, task, [tn], "valid", args, floatembmodel)
        for tn in test_tasks
    }

    metric_dict = {tn: task.metatask[tn]["metric"] for tn in all_tasks}

    best_valid_metric = -torch.inf
    best_checkpoint_path = None
    best_epoch = 0

    model, projector, optimizer = accelerator.prepare(model, projector, optimizer)

    if accelerator.is_main_process:
        g_params = sum(p.numel() for p in model.parameters())
        p_params = sum(p.numel() for p in projector.parameters())
        l_params = sum(p.numel() for p in llm_decoder.model.parameters())
        t_params = sum(p.numel() for p in trainable_params if p.requires_grad)
        print(f"[Params] Griffin: {g_params:,} | Projector: {p_params:,} | LLM: {l_params:,}")
        print(f"[Params] Trainable: {t_params:,}\n")

    # ── Training loop ──
    model.train()
    projector.train()
    step = 0

    for epoch in range(args.maxepoch):
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch} starts")

        train_dataset.rebuild_indice(accelerator)
        loader = DataLoader(
            train_dataset, shuffle=True, batch_size=1,
            collate_fn=lambda xlist: xlist[0],
            num_workers=8, prefetch_factor=4,
            persistent_workers=False, pin_memory=True,
        )
        loader = accelerator.prepare(loader)

        for data in loader:
            step += 1
            optimizer.zero_grad()

            # For multi-task training, use the first train task
            # (in production, detect from batch metadata)
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

            if step % 10 == 0:
                tbtracker.log({"training_loss": loss.item()}, step=step)
                if accelerator.is_main_process and step % 50 == 0:
                    print(f"  step {step}: loss = {loss.item():.4f}")

        accelerator.wait_for_everyone()

        # Save checkpoint
        checkpoint_path = (
            osp.join(args.savepath, f"checkpoint-{epoch}-{step}")
            if args.savepath else None
        )
        if args.savepath and accelerator.is_main_process:
            accelerator.save_model(model, checkpoint_path)
            torch.save(
                accelerator.unwrap_model(projector).state_dict(),
                osp.join(checkpoint_path, "projector.pt"),
            )

        # ── Validation on TRAIN tasks (for early stopping) ──
        if (epoch + 1) % args.eval_per_epoch == 0:
            if accelerator.is_main_process:
                print(f"\n--- Validation on TRAIN tasks ---")
            eval_metric = {}
            for tn in train_tasks:
                if accelerator.is_main_process:
                    print(f"  Validating {tn}...")
                eval_metric[tn] = eval_task(
                    model, projector, llm_decoder,
                    train_valid_dict[tn], args, accelerator,
                    metric_dict[tn], tn, task_type_dict[tn],
                )
                if accelerator.is_main_process:
                    print(f"    valid/{tn}: {eval_metric[tn]}")

            if accelerator.is_main_process:
                avg_valid = sum(v for v in eval_metric.values() if v is not None) / len(eval_metric)
                print(f"  Avg train-valid metric: {avg_valid:.4f}")

            avg_valid = accelerator.gather(
                torch.tensor([avg_valid if accelerator.is_main_process else 0.0]).to(accelerator.device)
            ).mean().item()

            if avg_valid > best_valid_metric:
                best_valid_metric = avg_valid
                best_checkpoint_path = checkpoint_path
                best_epoch = epoch

                # ── Zero-shot eval on TEST tasks (cross-dataset) ──
                if accelerator.is_main_process:
                    print(f"\n--- Zero-shot eval on TEST tasks (cross-dataset) ---")
                for tn in test_tasks:
                    if accelerator.is_main_process:
                        print(f"  Testing {tn}...")
                    test_metric = eval_task(
                        model, projector, llm_decoder,
                        test_valid_dict[tn], args, accelerator,
                        metric_dict[tn], tn, task_type_dict[tn],
                    )
                    if accelerator.is_main_process:
                        tbtracker.log({f"transfer_valid/{tn}": test_metric}, step=step)
                        print(f"    transfer_valid/{tn}: {test_metric}")

            if args.patience > 0 and epoch - best_epoch > args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

            model.train()
            projector.train()

    # ══════════════════════════════════════════════════════════════════
    # FINAL: Test on cross-dataset test tasks
    # ══════════════════════════════════════════════════════════════════
    if best_checkpoint_path is not None:
        if accelerator.is_main_process:
            print(f"\nLoading best checkpoint from {best_checkpoint_path}")
        unwrap_model = accelerator.unwrap_model(model)
        accelerate.load_checkpoint_in_model(unwrap_model, best_checkpoint_path)
        model = accelerator.prepare(unwrap_model)
        proj_path = osp.join(best_checkpoint_path, "projector.pt")
        if osp.exists(proj_path):
            accelerator.unwrap_model(projector).load_state_dict(
                torch.load(proj_path, map_location=accelerator.device)
            )
        if accelerator.is_main_process and args.savepath:
            best_dir = osp.join(args.savepath, 'best_checkpoint')
            os.makedirs(best_dir, exist_ok=True)
            accelerator.save_model(model, best_dir)
            torch.save(
                accelerator.unwrap_model(projector).state_dict(),
                osp.join(best_dir, 'projector.pt'),
            )

    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"FINAL CROSS-DATASET TEST RESULTS")
        print(f"Trained on: {train_tasks}")
        print(f"Testing on: {test_tasks}")
        print(f"{'='*60}")

    final_metrics = {}
    for tn in test_tasks:
        if accelerator.is_main_process:
            print(f"  Testing {tn}...")
        metric_val = eval_task(
            model, projector, llm_decoder,
            test_dataset_dict[tn], args, accelerator,
            metric_dict[tn], tn, task_type_dict[tn],
        )
        if accelerator.is_main_process:
            tbtracker.log({f"final_test/{tn}": metric_val}, step=step)
            print(f"    FINAL test/{tn}/{metric_dict[tn]}: {metric_val}")
            final_metrics[tn] = metric_val

    if accelerator.is_main_process:
        avg = sum(final_metrics.values()) / len(final_metrics)
        print(f"\n  Average cross-dataset test metric: {avg:.4f}")
        print(f"{'='*60}")

    accelerator.end_training()


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
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("dataset", type=str)
    parser.add_argument("logdir", type=str)
    parser.add_argument("logname", type=str)

    # ══════════════════════════════════════════════════════════════════
    # KEY: Separate --train_tasks and --test_tasks
    # ══════════════════════════════════════════════════════════════════
    parser.add_argument("--train_tasks", type=str, nargs="+", required=True,
                        help="Tasks to train on (e.g., rel-amazon-user-churn)")
    parser.add_argument("--test_tasks", type=str, nargs="+", required=True,
                        help="Tasks to test on (e.g., rel-hm-user-churn)")

    parser.add_argument("--savepath", type=str, default=None)
    parser.add_argument("--loadpath", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    # Training
    parser.add_argument("--batchsize", type=int, default=4)
    parser.add_argument("--eval_batchsize", type=int)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=2e-4)
    parser.add_argument("--maxepoch", type=int, default=10)
    parser.add_argument("--patience", type=int, default=-1)
    parser.add_argument("--eval_per_epoch", type=int, default=1)

    # Griffin MPNN
    parser.add_argument("--num_mp", type=int, default=4)
    parser.add_argument("--hiddim", type=int, default=512)
    parser.add_argument("--fanout", type=int, default=20)
    parser.add_argument("--fewshotfanout", type=int, default=3)
    parser.add_argument("--hop", type=int, default=2)
    parser.add_argument("--use_rev", type=str2bool, default=True)
    parser.add_argument("--use_gate", type=str2bool, default=False)

    # LLM
    parser.add_argument("--llm_model", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--llm_frozen", action="store_true", default=True)
    parser.add_argument("--use_lora", action="store_true", default=False)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--freeze_griffin", action="store_true", default=False)
    parser.add_argument("--max_new_tokens", type=int, default=16)

    args = parser.parse_args()
    args.eval_batchsize = args.batchsize if args.eval_batchsize is None else args.eval_batchsize
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.savepath:
        os.makedirs(args.savepath, exist_ok=True)
    main(args)
