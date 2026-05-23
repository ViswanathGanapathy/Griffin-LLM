"""
hmodel_smpnn.py — SMPNN variant of Griffin's GNN module.

Drop-in replacement for hmodel.py (same GriffinMod class name, same forward
signature). Differences from vanilla Griffin:

  - Per-layer affine LayerNorms (ln_gnn, ln_ff) for each sub-block
  - Two sub-blocks per MP layer: GNN sub-block + FF sub-block
  - Separate residuals on each sub-block (Pre-LN Transformer pattern)
  - Learnable alpha scaling on both sub-blocks (init 1e-6 = near-identity)

State-dict is a superset of vanilla — vanilla checkpoints load cleanly
(missing keys for ln_gnn/ln_ff/alpha_gnn/alpha_ff default-init at
near-identity, so the warm-start is behavior-preserving).

Reference: Saez de Ocariz Borde et al. (2024), "Scalable Message Passing
Neural Networks: No Need for Attention in Large Graph Representation
Learning" (arXiv:2411.00835).

Usage:
    # In hmaintask_combine_llm.py
    from hmodel_smpnn import GriffinMod   # via --use_smpnn flag
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Union
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.aggr import MeanAggregation, MaxAggregation
import numpy as np
import os


class SelfAverageAggregator(nn.Module):
    def __init__(self, hiddim: int, num_heads: int = 8, num_layer: int = 1,
                 dim_feedforward: int = None, dropout: float = 0.1):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = hiddim
        self.linq = nn.Linear(hiddim, hiddim, bias=False)
        self.crossattention = nn.MultiheadAttention(
            hiddim, num_heads, dropout, bias=False, batch_first=True,
        )

    def forward(self, column_name_emb: Tensor, x, mask=None):
        column_name_emb = column_name_emb.unsqueeze(0).expand(x.shape[0], -1, -1)
        ret = self.crossattention(
            column_name_emb, column_name_emb, x,
            key_padding_mask=mask, need_weights=False,
        )[0]
        return self.linq(ret)


class SelfAttentionAggregator(nn.Module):
    def __init__(self, hiddim: int, num_heads=8, num_layer=1,
                 dim_feedforward: int = None, dropout=0.1):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = hiddim
        self.linq = nn.Linear(hiddim, hiddim, bias=False)
        self.crossattention = nn.MultiheadAttention(
            hiddim, num_heads, dropout, bias=False, batch_first=True,
        )

    def forward(self, tar: Union[Tensor, None], column_name_emb: Tensor, x, mask=None):
        column_name_emb = column_name_emb.unsqueeze(0)
        q = tar
        ret = self.crossattention(
            q, column_name_emb.expand(x.shape[0], -1, -1), x,
            key_padding_mask=mask, need_weights=False,
        )[0]
        return ret * self.linq(tar)


def checktaskfeat(taskfeat: List[Tensor], node: List[Tuple[Tensor, Tensor]]):
    for i in range(len(taskfeat)):
        if taskfeat[i] is None:
            taskfeat[i] = node[i][0].mean(dim=0)
        if taskfeat[i].ndim == 1:
            taskfeat[i] = taskfeat[i].unsqueeze(0).expand(node[i][1].shape[0], -1)
        assert taskfeat[i].ndim == 2
    return taskfeat


def _attention_merge_feat_self_attention(attention_model, feat, colfeat, mask):
    return torch.mean(attention_model(colfeat, feat, mask), dim=1)


def _attention_merge_feat_with_relation(attention_model, feat, colfeat, taskfeat, mask):
    return attention_model(taskfeat.unsqueeze(1), colfeat, feat, mask).squeeze(1)


class RMPNN(nn.Module):
    """Relational MPNN with mean+max aggregation per (target, edge_type)."""

    def __init__(self, hiddim) -> None:
        super().__init__()
        self.rellin = nn.Sequential(nn.Linear(hiddim, hiddim))
        self.aggr1 = MeanAggregation()
        self.aggr2 = MaxAggregation()

    def reset_parameters(self) -> None:
        return self.rellin[0].reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor,
                edge_attr_type: Tensor, edge_attr: Tensor) -> Tensor:
        if edge_index is None or edge_index.shape[1] == 0:
            return 0 * self.rellin(x[0])
        edge_attr = self.rellin(edge_attr)
        center, inv = torch.unique(
            torch.stack((edge_index[0], edge_attr_type), dim=0),
            dim=1, return_inverse=True,
        )
        out1 = self.aggr1(x[edge_index[1]], inv, dim_size=center.shape[1])
        out2 = self.aggr2(out1 * edge_attr[center[1]], center[0], dim_size=x.shape[0])
        return out2


class GriffinMod(nn.Module):
    """SMPNN-style Griffin GNN.

    Per-layer structure (compare to vanilla Griffin which has a single
    block per layer):

        # Sub-block 1: GNN
        h1 = ln_gnn[i](x)
        gnn = (gate * mpnn(mlp(h1))) + (revgate * revmpnn(mlp(h1)))   # if use_rev
        x   = x + alpha_gnn[i] * gnn

        # Sub-block 2: FF
        h2 = ln_ff[i](x)
        x  = x + alpha_ff[i] * mlp2(h2)
    """

    def __init__(self, hiddim: int = 256, num_tf: int = 1, num_mp: int = 2,
                 use_rev: bool = True, use_gate: bool = True,
                 alpha_init: float = 1e-6):
        super().__init__()
        self.nodefeataggr = nn.ModuleList(
            [SelfAverageAggregator(hiddim, num_layer=num_tf)]
            + [SelfAttentionAggregator(hiddim, num_layer=num_tf) for _ in range(num_mp - 1)]
        )
        self.mpnn = nn.ModuleList([RMPNN(hiddim) for _ in range(num_mp)])
        self.lintask = nn.ModuleList(
            [nn.Linear(hiddim, hiddim, bias=False) for _ in range(num_mp - 1)]
        )
        self.mlp = nn.ModuleList([
            nn.Sequential(nn.Linear(hiddim, hiddim, bias=False),
                          nn.SiLU(inplace=True))
            for _ in range(num_mp)
        ])
        self.mlp2 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hiddim, hiddim, bias=False),
                nn.SiLU(inplace=True),
                nn.Linear(hiddim, hiddim, bias=False),
            )
            for _ in range(num_mp)
        ])
        self.num_mp = num_mp
        self.use_rev = use_rev
        self.use_gate = use_gate

        # SMPNN-only params (vanilla checkpoints will be missing these):
        self.ln_gnn = nn.ModuleList([nn.LayerNorm(hiddim) for _ in range(num_mp)])
        self.ln_ff  = nn.ModuleList([nn.LayerNorm(hiddim) for _ in range(num_mp)])
        self.alpha_gnn = nn.ParameterList(
            [nn.Parameter(torch.tensor(alpha_init)) for _ in range(num_mp)]
        )
        self.alpha_ff = nn.ParameterList(
            [nn.Parameter(torch.tensor(alpha_init)) for _ in range(num_mp)]
        )

        # Shared LN (kept for parity with vanilla — used in taskfeat
        # projection and final output).
        self.ln = nn.LayerNorm(hiddim)

        self.gatelin = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hiddim, int(hiddim**0.5)),
                nn.SiLU(inplace=True),
                nn.Linear(int(hiddim**0.5), 1, bias=False),
            )
            for _ in range(num_mp)
        ]) if use_gate else None
        self.revmpnn = nn.ModuleList([RMPNN(hiddim) for _ in range(num_mp)]) if use_rev else None
        self.revgatelin = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hiddim, int(hiddim**0.5)),
                nn.SiLU(inplace=True),
                nn.Linear(int(hiddim**0.5), 1, bias=False),
            )
            for _ in range(num_mp)
        ]) if (use_gate and use_rev) else None

        with torch.no_grad():
            for _ in range(num_mp):
                if use_gate:
                    self.gatelin[_][2].weight.fill_(0.0)
                    if use_rev:
                        self.revgatelin[_][2].weight.fill_(0.0)

    def reset_parameters(self):
        for i in range(len(self.mpnn)):
            self.mpnn[i].reset_parameters()
            if self.gatelin is not None:
                self.gatelin[i][0].reset_parameters()

    def forward(self, node, mask, taskfeat,
                edge_index, edge_attr_type, edge_attr) -> Tensor:
        assert len(node) == len(mask)
        assert len(node) == len(taskfeat)
        num_node: int = len(node)
        taskfeat = checktaskfeat(taskfeat, node)
        nodeptr = [0] + [_.shape[0] for _ in taskfeat]
        nodeptr = np.cumsum(nodeptr)

        for i in range(self.num_mp):
            if i == 0:
                x = torch.concat([
                    _attention_merge_feat_self_attention(
                        self.nodefeataggr[i], feat, colfeat, mask[j],
                    )
                    for j, (colfeat, feat) in enumerate(node)
                ], dim=0)
            else:
                x = x + torch.concat([
                    _attention_merge_feat_with_relation(
                        self.nodefeataggr[i], feat, colfeat,
                        self.ln(taskfeat[j]), mask[j],
                    )
                    for j, (colfeat, feat) in enumerate(node)
                ], dim=0)

            # SMPNN Sub-block 1: GNN
            lnx_gnn = self.ln_gnn[i](x)
            mlpx = self.mlp[i](lnx_gnn)
            gnn_out = (
                (self.gatelin[i](lnx_gnn) if self.use_gate else 1)
                * self.mpnn[i](mlpx, edge_index, edge_attr_type, edge_attr)
            ) + (
                ((self.revgatelin[i](lnx_gnn) if self.use_gate else 1)
                 * self.revmpnn[i](
                     mlpx,
                     None if edge_index is None else edge_index[[1, 0]],
                     edge_attr_type, edge_attr))
                if self.use_rev else 0
            )
            x = x + self.alpha_gnn[i] * gnn_out

            # SMPNN Sub-block 2: FF
            lnx_ff = self.ln_ff[i](x)
            x = x + self.alpha_ff[i] * self.mlp2[i](lnx_ff)

            if i < self.num_mp - 1:
                ntaskfeat = self.lintask[i](self.ln(x))
                for j in range(num_node):
                    taskfeat[j] = ntaskfeat[nodeptr[j]:nodeptr[j + 1]] + taskfeat[j]

        return self.ln(x)
