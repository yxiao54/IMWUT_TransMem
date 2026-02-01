import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================
# Helpers
# =====================================================

def mem_mlp(in_dim, hidden_dim, out_dim, depth=2, drop=0.2):
    """Small memory-conditioned MLP with LayerNorm."""
    layers = []
    d = in_dim
    for _ in range(depth - 1):
        layers += [
            nn.Linear(d, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(drop),
        ]
        d = hidden_dim
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


def map_gate_raw(x, mode="sigmoid", a=1.0, b=0.5):
    """Map raw gate outputs to bounded ranges."""
    if mode == "tanh":
        return torch.tanh(x)
    if mode == "sigmoid":
        return torch.sigmoid(x)
    if mode == "shifted_tanh":
        return 1.0 + 0.5 * torch.tanh(x)
    if mode == "affine":
        return a + b * torch.tanh(x)
    raise ValueError(f"Unknown gate mode: {mode}")


# =====================================================
# Base classes
# =====================================================



class UnifiedTwoFlowBase(nn.Module):
    """
    Unified baseline-compatible backbone with shared utilities.

    - fc1–fc3: physiological embedding layers
    - fc4: kept for compatibility with pretrained stress/baseline checkpoints
    - Provides utilities for sparse regularization and checkpoint loading
    """

    def __init__(
        self,
        input_dim: int,
        mem_input_dim: int,
        hidden: int,
        num_class: int,
        drop: float = 0.2,
        freeze_baseline: bool = True,
    ):
        super().__init__()

        # Baseline backbone (shared with stress classifier)
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, num_class)  # compatibility only

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop)
        self.norm = nn.LayerNorm(hidden)

        if freeze_baseline:
            for p in self.parameters():
                p.requires_grad = False

    # -------------------------------------------------
    # Embedding
    # -------------------------------------------------
    def baseline_embed(self, phys_feat: torch.Tensor) -> torch.Tensor:
        """
        Compute baseline physiological embedding using fc1–fc3.
        Output shape: (B, hidden)
        """
        x = self.dropout(self.relu(self.fc1(phys_feat)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.norm(self.dropout(self.relu(self.fc3(x))))
        return x

    # -------------------------------------------------
    # Sparse regularization helpers
    # -------------------------------------------------
    @staticmethod
    def sparse_penalty_on_scale(scale: torch.Tensor) -> torch.Tensor:
        """Encourage logit scale to stay close to 1."""
        return (scale - 1.0).abs().mean()

    @staticmethod
    def sparse_penalty_on_gate(gate: torch.Tensor) -> torch.Tensor:
        """Encourage sparse feature gating."""
        return gate.mean()

    # -------------------------------------------------
    # Checkpoint utilities
    # -------------------------------------------------
    def load_baseline_from_ckpt(self, ckpt_path: str):
        """
        Load pretrained baseline/stress backbone parameters (fc1–fc4).
        """
        state = torch.load(ckpt_path, map_location="cpu")
        own_state = self.state_dict()
        loaded = []

        for name, param in state.items():
            if name in own_state and name.startswith(("fc1.", "fc2.", "fc3.", "fc4.")):
                own_state[name].copy_(param)
                loaded.append(name)

        if not loaded:
            raise RuntimeError(f"No baseline parameters loaded from {ckpt_path}")

        print(f"[UnifiedTwoFlowBase] Loaded {len(loaded)} baseline parameters.")


# =====================================================
# Proposed model: ReSPIRE
# =====================================================

class ReSPIRE(UnifiedTwoFlowBase):
    """
    Resilience-guided model with feature gating and logit scaling.
    """

    def __init__(self, input_dim, mem_input_dim, hidden, num_class,
                 drop=0.2, groups=16, mem_hidden=256, mem_depth=2,
                 gate_mode="sigmoid"):
        super().__init__(input_dim, mem_input_dim, hidden, num_class, drop)

        self.groups = groups
        self.register_buffer("group_ids",
                             torch.arange(input_dim) * groups // input_dim)

        self.mem_to_gate = mem_mlp(mem_input_dim, mem_hidden, groups,
                                   depth=mem_depth, drop=drop)
        self.mem_to_logit_scale = mem_mlp(mem_input_dim, mem_hidden // 2, 1)

        self.input_ln = nn.LayerNorm(input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(drop),
        )

        self.fusion_w = nn.Parameter(torch.ones(2))
        self.classifier = nn.Linear(hidden, num_class)
        self.gate_mode = gate_mode

    def forward(self, phys_feat, good_embed, bad_embed):
        mem = torch.cat([good_embed, bad_embed], dim=-1)

        t_base = self.baseline_embed(phys_feat)

        gate_raw = self.mem_to_gate(mem)
        gate = map_gate_raw(gate_raw, self.gate_mode)[:, self.group_ids]
        x = self.input_ln(phys_feat * gate)

        t_new = self.encoder(x)
        w = F.softmax(self.fusion_w, dim=0)
        logits = self.classifier(w[0] * t_base + w[1] * t_new)

        logit_scale = F.softplus(self.mem_to_logit_scale(mem)).squeeze(-1)
        logits = logits * logit_scale.unsqueeze(-1)

        return {
            "logits": logits,
            "gate": gate,
            "logit_scale": logit_scale,
        }


# =====================================================
# Baselines
# =====================================================
class BaselineNet(UnifiedTwoFlowBase):
    """
    Physiology-only DNNs
    """
    def __init__(self, input_dim, mem_input_dim, hidden, num_class, drop=0.2, **kwargs):
        super().__init__(input_dim, mem_input_dim, hidden, num_class, drop)

        self.mem_proj = nn.Linear(mem_input_dim, hidden)
        self.classifier_base = nn.Linear(hidden, num_class)
        self.classifier_mem = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(drop),
            nn.Linear(hidden, num_class),
        )
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(drop),
        )

    def forward(self, phys_feat, good_embed, bad_embed):
        mem = torch.cat([good_embed, bad_embed], dim=-1)
        t_base = self.encoder(phys_feat)
        #t_mem = self.mem_proj(mem)

        logits = self.classifier_base(t_base) #+ self.classifier_mem(t_mem)
        return {"logits": logits}
 
        


class ConcatNet(UnifiedTwoFlowBase):
    """Early fusion baseline."""

    def __init__(self, input_dim, mem_input_dim, hidden, num_class, drop=0.2):
        super().__init__(input_dim, mem_input_dim, hidden, num_class, drop)
        self.mem_proj = nn.Linear(mem_input_dim, hidden)
        self.classifier = nn.Linear(hidden * 2, num_class)

    def forward(self, phys_feat, good_embed, bad_embed):
        mem = torch.cat([good_embed, bad_embed], dim=-1)
        t = torch.cat([self.baseline_embed(phys_feat), self.mem_proj(mem)], dim=-1)
        return {"logits": self.classifier(t)}


class SelfAttnNet(UnifiedTwoFlowBase):
    """Self-attention over physiology and memory tokens."""

    def __init__(self, input_dim, mem_input_dim, hidden, num_class, drop=0.2):
        super().__init__(input_dim, mem_input_dim, hidden, num_class, drop)
        self.mem_proj = nn.Linear(mem_input_dim, hidden)
        self.attn = nn.MultiheadAttention(hidden, 4, dropout=drop, batch_first=True)
        self.classifier = nn.Linear(hidden, num_class)

    def forward(self, phys_feat, good_embed, bad_embed):
        mem = torch.cat([good_embed, bad_embed], dim=-1)
        tokens = torch.stack(
            [self.baseline_embed(phys_feat), self.mem_proj(mem)], dim=1
        )
        out, _ = self.attn(tokens, tokens, tokens)
        return {"logits": self.classifier(out.mean(dim=1))}


class CrossAttnNet(UnifiedTwoFlowBase):
    """Memory attends to physiological embedding."""

    def __init__(self, input_dim, mem_input_dim, hidden, num_class, drop=0.2):
        super().__init__(input_dim, mem_input_dim, hidden, num_class, drop)
        self.mem_proj = nn.Linear(mem_input_dim, hidden)
        self.attn = nn.MultiheadAttention(hidden, 4, dropout=drop, batch_first=True)
        self.classifier = nn.Linear(hidden, num_class)

    def forward(self, phys_feat, good_embed, bad_embed):
        mem = self.mem_proj(torch.cat([good_embed, bad_embed], dim=-1)).unsqueeze(1)
        phys = self.baseline_embed(phys_feat).unsqueeze(1)
        out, _ = self.attn(mem, phys, phys)
        return {"logits": self.classifier(out.squeeze(1))}


class MoENet(UnifiedTwoFlowBase):
    """Mixture-of-Experts baseline."""

    def __init__(self, input_dim, mem_input_dim, hidden, num_class,
                 drop=0.2, n_experts=4):
        super().__init__(input_dim, mem_input_dim, hidden, num_class, drop)
        self.router = nn.Linear(mem_input_dim, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
                nn.Dropout(drop),
            ) for _ in range(n_experts)
        ])
        self.classifier = nn.Linear(hidden, num_class)

    def forward(self, phys_feat, good_embed, bad_embed):
        mem = torch.cat([good_embed, bad_embed], dim=-1)
        t = self.baseline_embed(phys_feat)
        w = torch.softmax(self.router(mem), dim=-1)
        t = sum(w[:, i:i+1] * ex(t) for i, ex in enumerate(self.experts))
        return {"logits": self.classifier(t)}


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


class PersonalizedOnlyNet(UnifiedTwoFlowBase):
    """
    Personalized encoder only.
    No baseline anchor
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


class PhysOnlyNet(UnifiedTwoFlowBase):
    """Physiology-only baseline (no language)."""

    def __init__(self, input_dim, mem_input_dim, hidden, num_class, drop=0.2):
        super().__init__(input_dim, mem_input_dim, hidden, num_class, drop)
        self.classifier = nn.Linear(hidden, num_class)

    def forward(self, phys_feat, *_):
        t = self.baseline_embed(phys_feat)
        return {"logits": self.classifier(t)}

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
