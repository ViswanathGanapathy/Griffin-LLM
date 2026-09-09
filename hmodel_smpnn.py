"""
hmodel_smpnn.py — SMPNN variant of Griffin's GNN module.

Drop-in replacement for hmodel.py. Same GriffinMod class name and forward
signature; vanilla Griffin checkpoints warm-start cleanly because the
SMPNN-only params (ln_gnn / ln_ff / alpha_gnn / alpha_ff) are missing in
vanilla checkpoints and default-init at near-identity (alpha_init=1e-6).

Architecture (per layer i):

    # Sub-block 1: GNN
    h1   = ln_gnn[i](x)                          # optional via use_gnn_ln
    gnn  = gate * RMPNN(mlp(h1))                 # + revRMPNN if use_rev
    if use_attention:
        h1g  = ln_global[i](x)
        gnn  = gnn + LinearGlobalAttention(h1g)
    x   = x + alpha_gnn[i] * gnn                 # alpha=1 if use_alpha=False

    # Sub-block 2: FF (skipped if use_ff=False)
    h2  = ln_ff[i](x)
    x   = x + alpha_ff[i] * mlp2(h2)

Ablation flags (all in GriffinMod.__init__):
    use_attention : add parallel linear global attention (Appendix A)
    num_heads     : number of attention heads (default 1)
    use_alpha     : enable learnable alpha scaling (vs fixed alpha=1)
    use_ff        : enable pointwise feedforward sub-block
    use_gnn_ln    : enable Pre-LayerNorm before GNN sub-block

Reference: Saez de Ocariz Borde et al. (2024), "Scalable Message Passing
Neural Networks: No Need for Attention in Large Graph Representation
Learning" (arXiv:2411.00835).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union
from torch import Tensor
from torch_geometric.nn.aggr import MeanAggregation, MaxAggregation
import numpy as np


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


class LinearGlobalAttention(nn.Module):
    """Linear global attention via a virtual node (O(N) complexity).

    From SMPNN paper Eqs. 13-16:
        Q = X*WQ;  K = X*WK;  V = X*WV
        Qsn = sum(Q) / ||sum(Q)||_2          (single D-dim query)
        Kn  = K / ||K||_2                     (normalised keys)
        A   = softmax(Qsn * Kn^T)            (1xN attention weights)
        out = (A kron 1_N) * V               (all nodes get same weighted sum)
    """

    def __init__(self, hiddim: int, num_heads: int = 1, dropout: float = 0.0):
        super().__init__()
        assert hiddim % num_heads == 0, (
            f"hiddim ({hiddim}) must be divisible by num_heads ({num_heads})"
        )
        self.hiddim = hiddim
        self.num_heads = num_heads
        self.head_dim = hiddim // num_heads
        self.wq = nn.Linear(hiddim, hiddim, bias=False)
        self.wk = nn.Linear(hiddim, hiddim, bias=False)
        self.wv = nn.Linear(hiddim, hiddim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        N, D = x.shape
        H, d = self.num_heads, self.head_dim
        Q = self.wq(x).view(N, H, d)
        K = self.wk(x).view(N, H, d)
        V = self.wv(x).view(N, H, d)
        Q_sum = Q.sum(dim=0)                                       # (H, d)
        Q_sn = Q_sum / (Q_sum.norm(dim=-1, keepdim=True) + 1e-8)   # (H, d)
        K_norm = K / (K.norm(dim=-1, keepdim=True) + 1e-8)         # (N, H, d)
        scores = torch.einsum('hd,nhd->hn', Q_sn, K_norm)          # (H, N)
        attn = F.softmax(scores, dim=-1)                            # (H, N)
        attn = self.dropout(attn)
        global_feat = torch.einsum('hn,nhd->hd', attn, V)          # (H, d)
        return global_feat.unsqueeze(0).expand(N, -1, -1).reshape(N, D)


class GriffinMod(nn.Module):
    """SMPNN-style Griffin GNN with optional global attention and per-component
    ablation switches.

    State-dict is a superset of vanilla Griffin under default flags. Disabling
    a component (e.g. use_ff=False) removes the corresponding params entirely,
    which makes checkpoints incompatible across flag settings — ablations
    should typically train from scratch.
    """

    def __init__(self, hiddim: int = 256, num_tf: int = 1, num_mp: int = 2,
                 use_rev: bool = True, use_gate: bool = True,
                 alpha_init: float = 1e-6,
                 # Attention (Appendix A)
                 use_attention: bool = False, num_heads: int = 1,
                 # Component ablations
                 use_alpha: bool = True,
                 use_ff: bool = True,
                 use_gnn_ln: bool = True):
        super().__init__()

        self.use_attention = use_attention
        self.use_alpha = use_alpha
        self.use_ff = use_ff
        self.use_gnn_ln = use_gnn_ln
        self.num_mp = num_mp
        self.use_rev = use_rev
        self.use_gate = use_gate

        # Cross-attention aggregators (Griffin's per-table value aggregation)
        self.nodefeataggr = nn.ModuleList(
            [SelfAverageAggregator(hiddim, num_layer=num_tf)]
            + [SelfAttentionAggregator(hiddim, num_layer=num_tf)
               for _ in range(num_mp - 1)]
        )

        # RMPNN message-passing layers
        self.mpnn = nn.ModuleList([RMPNN(hiddim) for _ in range(num_mp)])
        self.lintask = nn.ModuleList(
            [nn.Linear(hiddim, hiddim, bias=False) for _ in range(num_mp - 1)]
        )
        self.mlp = nn.ModuleList([
            nn.Sequential(nn.Linear(hiddim, hiddim, bias=False),
                          nn.SiLU(inplace=True))
            for _ in range(num_mp)
        ])

        # Pointwise FF sub-block (skippable via use_ff)
        if use_ff:
            self.mlp2 = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hiddim, hiddim, bias=False),
                    nn.SiLU(inplace=True),
                    nn.Linear(hiddim, hiddim, bias=False),
                )
                for _ in range(num_mp)
            ])

        # Per-sub-block LayerNorms
        if use_gnn_ln:
            self.ln_gnn = nn.ModuleList([nn.LayerNorm(hiddim) for _ in range(num_mp)])
        if use_ff:
            self.ln_ff = nn.ModuleList([nn.LayerNorm(hiddim) for _ in range(num_mp)])
        # Shared LN used in taskfeat projection and final output
        self.ln = nn.LayerNorm(hiddim)

        # Learnable alpha scaling (DiT-style identity init)
        if use_alpha:
            self.alpha_gnn = nn.ParameterList(
                [nn.Parameter(torch.tensor(alpha_init)) for _ in range(num_mp)]
            )
            if use_ff:
                self.alpha_ff = nn.ParameterList(
                    [nn.Parameter(torch.tensor(alpha_init)) for _ in range(num_mp)]
                )

        # Linear Global Attention (Appendix A, Eqs. 17-21)
        if use_attention:
            self.ln_global = nn.ModuleList(
                [nn.LayerNorm(hiddim) for _ in range(num_mp)]
            )
            self.global_attn = nn.ModuleList([
                LinearGlobalAttention(hiddim, num_heads=num_heads)
                for _ in range(num_mp)
            ])

        # Gating (zero-init keeps the network near-identity at step 0)
        self.gatelin = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hiddim, int(hiddim**0.5)),
                nn.SiLU(inplace=True),
                nn.Linear(int(hiddim**0.5), 1, bias=False),
            )
            for _ in range(num_mp)
        ]) if use_gate else None
        self.revmpnn = nn.ModuleList(
            [RMPNN(hiddim) for _ in range(num_mp)]
        ) if use_rev else None
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

    def _alpha_gnn(self, i):
        return self.alpha_gnn[i] if self.use_alpha else 1

    def _alpha_ff(self, i):
        return self.alpha_ff[i] if (self.use_alpha and self.use_ff) else 1

    def _gnn_ff_step(self, x, i, edge_index, edge_attr_type, edge_attr):
        # Sub-block 1: GNN (local) + optional attention (global), parallel
        lnx_gnn = self.ln_gnn[i](x) if self.use_gnn_ln else x
        mlpx = self.mlp[i](lnx_gnn)

        local_out = (
            (self.gatelin[i](lnx_gnn) if self.use_gate else 1)
            * self.mpnn[i](mlpx, edge_index, edge_attr_type, edge_attr)
        )
        if self.use_rev:
            local_out = local_out + (
                (self.revgatelin[i](lnx_gnn) if self.use_gate else 1)
                * self.revmpnn[i](
                    mlpx,
                    None if edge_index is None else edge_index[[1, 0]],
                    edge_attr_type, edge_attr,
                )
            )

        if self.use_attention:
            lnx_global = self.ln_global[i](x)
            gnn_out = local_out + self.global_attn[i](lnx_global)
        else:
            gnn_out = local_out

        x = x + self._alpha_gnn(i) * gnn_out

        # Sub-block 2: FF (optional)
        if self.use_ff:
            lnx_ff = self.ln_ff[i](x)
            x = x + self._alpha_ff(i) * self.mlp2[i](lnx_ff)

        return x

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

            x = self._gnn_ff_step(x, i, edge_index, edge_attr_type, edge_attr)

            if i < self.num_mp - 1:
                ntaskfeat = self.lintask[i](self.ln(x))
                for j in range(num_node):
                    taskfeat[j] = ntaskfeat[nodeptr[j]:nodeptr[j + 1]] + taskfeat[j]

        return self.ln(x)
