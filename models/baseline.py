from models.newmodel import UnifiedTwoFlowBase,TwoFlowBase
import numpy as np
import random

import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import math
from torch.autograd import Variable
from torch.distributions import Normal



def _mem_mlp(in_dim, hidden_dim, out_dim, depth=2, drop=0.2):
    layers = []
    d_in = in_dim
    for _ in range(depth - 1):
        layers += [
            nn.Linear(d_in, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(drop),
        ]
        d_in = hidden_dim
    layers += [nn.Linear(d_in, out_dim)]
    return nn.Sequential(*layers)

def _map_gate_raw(x, mode="sigmoid", a=1.0, b=0.5):
    if mode == "tanh":
        return torch.tanh(x)
    elif mode == "sigmoid":
        return torch.sigmoid(x)
    elif mode == "shifted_tanh":
        return 1.0 + 0.5 * torch.tanh(x)
    elif mode == "affine":
        return a + b * torch.tanh(x)
    else:
        raise ValueError(f"Unknown gate mode: {mode}")




class ProfileBiasNet(UnifiedTwoFlowBase):
    """
    Profile-only prior baseline: memory predicts a logit bias.
    Physiology is ignored.
    """
    def __init__(self, input_dim, mem_input_dim, hidden, num_class, drop=0.2, **kwargs):
        super().__init__(input_dim, mem_input_dim, hidden, num_class, drop)
        self.mem_to_bias = nn.Sequential(
            nn.Linear(mem_input_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(drop),
            nn.Linear(hidden, num_class),
        )

    def forward(self, phys_feat, good_embed, bad_embed):
        mem = torch.cat([good_embed, bad_embed], dim=-1)
        logits = self.mem_to_bias(mem)
        return {"logits": logits}
class DecisionScaleOnlyNet(UnifiedTwoFlowBase):
    """
    Profile-conditioned decision sensitivity only.
    No feature gating.
    """
    def __init__(
        self,
        input_dim,
        mem_input_dim,
        hidden,
        num_class,
        drop=0.2,
        mem_hidden=256,
        **kwargs,
    ):
        super().__init__(input_dim, mem_input_dim, hidden, num_class, drop)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(drop),
        )
        self.classifier = nn.Linear(hidden, num_class)

        # local helper
        self.mem_to_logit_scale = _mem_mlp(
            mem_input_dim,
            mem_hidden // 2,
            1,
            depth=2,
            drop=drop,
        )

    def forward(self, phys_feat, good_embed, bad_embed):
        mem = torch.cat([good_embed, bad_embed], dim=-1)

        t = self.encoder(phys_feat)
        logits = self.classifier(t)

        scale = F.softplus(self.mem_to_logit_scale(mem)).squeeze(-1)
        logits = logits * scale.unsqueeze(-1)

        return {
            "logits": logits,
            "logit_scale": scale,
        }

class PhysGateNet(UnifiedTwoFlowBase):
    """
    Invalid-but-informative baseline:
    Feature sensitivity is predicted from physiology instead of memory.
    """
    def __init__(
        self,
        input_dim,
        mem_input_dim,
        hidden,
        num_class,
        drop=0.2,
        groups=16,
        gate_mode="sigmoid",
        **kwargs,
    ):
        super().__init__(input_dim, mem_input_dim, hidden, num_class, drop)

        self.groups = groups
        self.gate_mode = gate_mode

        self.register_buffer(
            "group_ids",
            torch.arange(input_dim) * groups // input_dim,
        )

        self.phys_to_gate = nn.Sequential(
            nn.Linear(input_dim, groups),
            nn.GELU(),
            nn.LayerNorm(groups),
        )

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(drop),
        )

        self.classifier = nn.Linear(hidden, num_class)

    def forward(self, phys_feat, good_embed, bad_embed):
        gate_raw = self.phys_to_gate(phys_feat)
        gate_g = _map_gate_raw(gate_raw, mode=self.gate_mode)
        gate = gate_g[:, self.group_ids]

        x = phys_feat * gate
        t = self.encoder(x)
        logits = self.classifier(t)

        return {
            "logits": logits,
            "gate": gate,
            "sparse_metric": gate.mean(),
        }


class PersonalizedOnlyNet(UnifiedTwoFlowBase):
    """
    Personalized encoder only.
    No baseline anchor, no fusion.
    """
    def __init__(
        self,
        input_dim,
        mem_input_dim,
        hidden,
        num_class,
        drop=0.2,
        groups=16,
        mem_hidden=256,
        **kwargs,
    ):
        super().__init__(input_dim, mem_input_dim, hidden, num_class, drop)

        self.groups = groups
        self.register_buffer(
            "group_ids",
            torch.arange(input_dim) * groups // input_dim,
        )

        self.mem_to_gate = _mem_mlp(
            mem_input_dim,
            mem_hidden,
            groups,
            depth=2,
            drop=drop,
        )

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(drop),
        )
        self.classifier = nn.Linear(hidden, num_class)

    def forward(self, phys_feat, good_embed, bad_embed):
        mem = torch.cat([good_embed, bad_embed], dim=-1)
        gate = torch.sigmoid(self.mem_to_gate(mem))[:, self.group_ids]

        x = phys_feat * gate
        t = self.encoder(x)
        logits = self.classifier(t)

        return {
            "logits": logits,
            "gate": gate,
            "sparse_metric": gate.mean(),
        }

class FixedFusionNet(UnifiedTwoFlowBase):
    """
    Fixed fusion weight between baseline and personalized encoding.
    """
    def __init__(
        self,
        input_dim,
        mem_input_dim,
        hidden,
        num_class,
        drop=0.2,
        alpha=0.5,
        groups=16,
        mem_hidden=256,
        **kwargs,
    ):
        super().__init__(input_dim, mem_input_dim, hidden, num_class, drop)

        self.alpha = alpha
        self.groups = groups

        self.register_buffer(
            "group_ids",
            torch.arange(input_dim) * groups // input_dim,
        )

        self.mem_to_gate = _mem_mlp(
            mem_input_dim,
            mem_hidden,
            groups,
            depth=2,
            drop=drop,
        )

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(drop),
        )
        self.classifier = nn.Linear(hidden, num_class)

    def forward(self, phys_feat, good_embed, bad_embed):
        mem = torch.cat([good_embed, bad_embed], dim=-1)

        t_base = self.baseline_embed(phys_feat)

        gate = torch.sigmoid(self.mem_to_gate(mem))[:, self.group_ids]
        x = phys_feat * gate
        t_new = self.encoder(x)

        t = self.alpha * t_base + (1.0 - self.alpha) * t_new
        logits = self.classifier(t)

        return {
            "logits": logits,
            "gate": gate,
            "sparse_metric": gate.mean(),
        }


class EnvInvariantNet(UnifiedTwoFlowBase):
    """
    Shared backbone for IRM / V-REx / DRO baselines.
    Subject is treated as environment.
    """
    def __init__(self, input_dim, mem_input_dim, hidden, num_class, drop=0.2, **kwargs):
        super().__init__(input_dim, mem_input_dim, hidden, num_class, drop)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(drop),
        )
        self.classifier = nn.Linear(hidden, num_class)

    def forward(self, phys_feat, good_embed, bad_embed):
        t = self.encoder(phys_feat)
        logits = self.classifier(t)
        return {
            "logits": logits,
            "repr": t,  
        }


