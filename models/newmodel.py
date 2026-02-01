import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================
# Base classes & helpers
# =====================================================

class TwoFlowBase(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def sparse_penalty_on_scale(scale):
        return (scale - 1.0).abs().mean()

    @staticmethod
    def sparse_penalty_on_gate(gate):
        return gate.mean()

    def load_baseline_from_ckpt(self, ckpt_path: str):
        state = torch.load(ckpt_path, map_location="cpu")
        own_state = self.state_dict()
        loaded_keys = []

        for name, param in state.items():
            if name in own_state and name.startswith(("fc1.", "fc2.", "fc3.", "fc4.")):
                try:
                    own_state[name].copy_(param)
                    loaded_keys.append(name)

                except Exception:

                    pass
        if len(loaded_keys) == 0:
            raise RuntimeError(f"No baseline parameters loaded from {ckpt_path}")
        print(f"[TwoFlowBase] Loaded {len(loaded_keys)} baseline params.")


class UnifiedTwoFlowBase(TwoFlowBase):
    """
    Unified base: defines fc1..fc4 for baseline compatibility.
    By default baseline layers are frozen; training script may unfreeze.
    """
    def __init__(self, input_dim, mem_input_dim, hidden, num_class, drop, freeze_baseline=True, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.mem_input_dim = mem_input_dim
        self.hidden = hidden
        self.num_class = num_class

        # baseline layers (fc1..fc3 used for embedding; fc4 kept for compatibility)
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, num_class)

        if freeze_baseline:
            for fc in [self.fc1, self.fc2, self.fc3, self.fc4]:
                for p in fc.parameters():
                    p.requires_grad = False

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop)
        self.norm_base = nn.LayerNorm(hidden)

    def baseline_embed(self, phys_feat):
        """
        Compute baseline embedding using fc1..fc3 (keeps shape (B, hidden)).
        We intentionally DO NOT apply fc4 here so that baseline participates at embedding level.
        """
        x = self.dropout(self.relu(self.fc1(phys_feat)))
        x = self.dropout(self.relu(self.fc2(x)))
        t_base = self.norm_base(self.dropout(self.relu(self.fc3(x))))
        return t_base
import torch
import torch.nn as nn
import torch.nn.functional as F

# Assumes UnifiedTwoFlowBase exists in scope and implements:
# - self.fc1..self.fc4 (frozen baseline)
# - self.relu, self.dropout, self.norm_base
# - baseline_embed(phys_feat) -> (B, hidden)
# If not, import or paste the UnifiedTwoFlowBase definition above.

def mem_mlp(in_dim, hidden_dim, out_dim, depth=2, drop=0.2):
    """Helper: small MLP with LayerNorm between layers."""
    layers = []
    d_in = in_dim
    for i in range(depth - 1):
        layers += [
            nn.Linear(d_in, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(drop),
        ]
        d_in = hidden_dim
    layers += [nn.Linear(d_in, out_dim)]
    return nn.Sequential(*layers)


def map_gate_raw(x, mode="tanh", a=1.0, b=0.5):
    """
    Map raw values to desired gating range.
    - mode == 'tanh' -> [-1, 1]
    - mode == 'sigmoid' -> [0, 1]
    - mode == 'shifted_tanh' -> [0.5, 1.5] via 1.0 + 0.5 * tanh(...)
    - mode == 'affine' -> a + b * tanh(...)
    """
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
        


class ReSPIRE(UnifiedTwoFlowBase):
    """
    Gate + mem-conditioned logit temperature (no additive bias).
    """
    def __init__(self, input_dim, mem_input_dim, hidden, num_class, drop=0.2,
                 groups=16, mem_hidden=256, mem_depth=2, gate_mode="sigmoid", **kwargs):
        super().__init__(input_dim, mem_input_dim, hidden, num_class, drop, **kwargs)

        self.groups = groups
        self.register_buffer("group_ids", torch.arange(input_dim) * groups // input_dim)

        self.mem_to_gate = mem_mlp(mem_input_dim, mem_hidden, groups, depth=mem_depth, drop=drop)
        self.input_ln = nn.LayerNorm(input_dim)
        self.pre_ln = nn.LayerNorm(hidden)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            self.pre_ln,
            nn.Dropout(drop),
        )

        self.fusion_w = nn.Parameter(torch.tensor([1.0, 1.0]))
        self.classifier = nn.Linear(hidden, num_class)

        self.mem_to_logit_scale = mem_mlp(mem_input_dim, mem_hidden//2, 1, depth=2, drop=drop)
        self.gate_mode = gate_mode

    def forward(self, phys_feat, good_embed, bad_embed):
        mem = torch.cat([good_embed, bad_embed], dim=-1)
        t_base = self.baseline_embed(phys_feat)

        gate_raw = self.mem_to_gate(mem)
        gate_g = map_gate_raw(gate_raw, mode=self.gate_mode)
        gate = gate_g[:, self.group_ids]

        x = phys_feat * gate
        x = self.input_ln(x)

        t_new = self.encoder(x)
        w = F.softmax(self.fusion_w, dim=0)
        logits = self.classifier(w[0] * t_base + w[1] * t_new)

        logit_scale = F.softplus(self.mem_to_logit_scale(mem)).squeeze(-1)
        logits = logits * logit_scale.unsqueeze(-1)

        return {
            "logits": logits,
            "t_base": t_base,
            "t_new": t_new,
            "gate": gate,
            "logit_scale": logit_scale,
            "sparse_metric": gate.mean(),
        }
        
        
class ConcatNet(UnifiedTwoFlowBase):
    """
    Naive baseline: concatenate baseline embedding and memory embedding.
    """
    def __init__(self, input_dim, mem_input_dim, hidden, num_class, drop=0.2, **kwargs):
        super().__init__(input_dim, mem_input_dim, hidden, num_class, drop)

        self.mem_proj = nn.Linear(mem_input_dim, hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
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
        t_mem = self.mem_proj(mem)

        t = torch.cat([t_base, t_mem], dim=-1)
        logits = self.classifier(t)
        return {"logits": logits}
   

class BaselineNet(UnifiedTwoFlowBase):
    """
    Late fusion: independent classifiers, logits are summed.
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
        logits = self.classifier_base(t_base)
        return {"logits": logits}
 
        
    
class SelfAttnNet(UnifiedTwoFlowBase):
    """
    Self-attention over baseline and memory tokens.
    """
    def __init__(self, input_dim, mem_input_dim, hidden, num_class, drop=0.2, **kwargs):
        super().__init__(input_dim, mem_input_dim, hidden, num_class, drop)

        self.mem_proj = nn.Linear(mem_input_dim, hidden)
        self.attn = nn.MultiheadAttention(hidden, num_heads=4, dropout=drop, batch_first=True)
        self.classifier = nn.Linear(hidden, num_class)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(drop),
        )


    def forward(self, phys_feat, good_embed, bad_embed):
        mem = torch.cat([good_embed, bad_embed], dim=-1)
        t_base = self.encoder(phys_feat).unsqueeze(1)
        t_mem = self.mem_proj(mem).unsqueeze(1)

        tokens = torch.cat([t_base, t_mem], dim=1)
        attn_out, _ = self.attn(tokens, tokens, tokens)

        pooled = attn_out.mean(dim=1)
        logits = self.classifier(pooled)
        return {"logits": logits}

class CrossAttnNet(UnifiedTwoFlowBase):
    """
    Cross-attention: memory attends to physiological embedding.
    """
    def __init__(self, input_dim, mem_input_dim, hidden, num_class, drop=0.2, **kwargs):
        super().__init__(input_dim, mem_input_dim, hidden, num_class, drop)

        self.mem_proj = nn.Linear(mem_input_dim, hidden)
        self.attn = nn.MultiheadAttention(hidden, num_heads=4, dropout=drop, batch_first=True)
        self.classifier = nn.Linear(hidden, num_class)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(drop),
        )

    def forward(self, phys_feat, good_embed, bad_embed):
        mem = torch.cat([good_embed, bad_embed], dim=-1)
        q = self.mem_proj(mem).unsqueeze(1)
        k = self.encoder(phys_feat).unsqueeze(1)

        attn_out, _ = self.attn(q, k, k)
        logits = self.classifier(attn_out.squeeze(1))
        return {"logits": logits}


    
class MoENet(UnifiedTwoFlowBase):
    """
    MoE baseline: memory routes between multiple physiological experts.
    """
    def __init__(self, input_dim, mem_input_dim, hidden, num_class, drop=0.2, n_experts=4, **kwargs):
        super().__init__(input_dim, mem_input_dim, hidden, num_class, drop)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
                nn.Dropout(drop),
            ) for _ in range(n_experts)
        ])
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(drop),
        )
        self.router = nn.Linear(mem_input_dim, n_experts)
        self.classifier = nn.Linear(hidden, num_class)

    def forward(self, phys_feat, good_embed, bad_embed):
        mem = torch.cat([good_embed, bad_embed], dim=-1)
        t_base = self.encoder(phys_feat)

        weights = torch.softmax(self.router(mem), dim=-1)
        t = sum(w.unsqueeze(-1) * ex(t_base) for w, ex in zip(weights.t(), self.experts))
        logits = self.classifier(t)
        return {"logits": logits}
    
class EnsembleNet(UnifiedTwoFlowBase):
    """
    Multi-head ensemble baseline.
    """
    def __init__(self, input_dim, mem_input_dim, hidden, num_class, drop=0.2, n_heads=4, **kwargs):
        super().__init__(input_dim, mem_input_dim, hidden, num_class, drop)

        self.heads = nn.ModuleList([
            nn.Linear(hidden, num_class) for _ in range(n_heads)
        ])
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(drop),
        )

    def forward(self, phys_feat, good_embed, bad_embed):
        t_base = self.encoder(phys_feat)
        logits = torch.stack([h(t_base) for h in self.heads], dim=0).mean(dim=0)
        return {"logits": logits}



