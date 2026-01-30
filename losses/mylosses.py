# losses/twoflow_step1.py

import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseLoss(ABC):
    """
    Base interface for all loss functions.
    """

    @abstractmethod
    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, C)
            labels: (B,)
            kwargs: optional extra info (env, group, etc.)

        Returns:
            scalar loss
        """
        pass

class TransMemNetLoss(BaseLoss):
    def __init__(
        self,
        lambda_sparse=0.0,
        
        **kwargs,
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

        self.lambda_sparse = lambda_sparse

    def __call__(self, output, labels, **kwargs):
        """
        Always return (loss, log_dict)
        """
        # --------------------------
        # Plain tensor output
        # --------------------------
        if isinstance(output, torch.Tensor):
            loss = self.ce(output, labels)
            return loss, {"cls": loss.detach()}

        total_loss = 0.0
        log_dict = {}

        # --------------------------
        # Main classification loss
        # --------------------------
        cls_loss = self.ce(output["logits"], labels)
        total_loss += cls_loss
        log_dict["cls"] = cls_loss.detach()




        if self.lambda_sparse > 0:
            if "sparse_metric" in output:
                sparse = output["sparse_metric"]
                sparse_loss = self.lambda_sparse * sparse
                total_loss += sparse_loss
                log_dict["sparse"] = sparse.detach()

        return total_loss, log_dict

class IRMLoss(nn.Module):
    """
    IRM loss with subject as environment.
    """
    def __init__(self, lambda_irm=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.lambda_irm = lambda_irm
        self.scale = torch.tensor(1.0, requires_grad=True)

    def forward(self, outputs, labels, env_ids):
        logits = outputs["logits"]
        loss = 0.0
        penalty = 0.0

        for env in torch.unique(env_ids):
            idx = env_ids == env
            if idx.sum() == 0:
                continue

            env_logits = logits[idx] * self.scale
            env_labels = labels[idx]

            env_loss = self.ce(env_logits, env_labels)
            grad = torch.autograd.grad(
                env_loss, self.scale, create_graph=True
            )[0]
            penalty += grad.pow(2)
            loss += env_loss

        loss = loss / len(torch.unique(env_ids))
        penalty = penalty / len(torch.unique(env_ids))

        total = loss + self.lambda_irm * penalty
        return total, {
            "cls": loss.detach(),
            "irm_penalty": penalty.detach(),
        }
class VRExLoss(nn.Module):
    """
    V-REx loss: penalize risk variance across subjects.
    """
    def __init__(self, lambda_vrex=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.lambda_vrex = lambda_vrex

    def forward(self, outputs, labels, env_ids):
        logits = outputs["logits"]
        risks = []

        for env in torch.unique(env_ids):
            idx = env_ids == env
            if idx.sum() == 0:
                continue
            env_loss = self.ce(logits[idx], labels[idx])
            risks.append(env_loss)

        risks = torch.stack(risks)
        mean_risk = risks.mean()
        var_risk = risks.var(unbiased=False)

        total = mean_risk + self.lambda_vrex * var_risk
        return total, {
            "cls": mean_risk.detach(),
            "vrex_var": var_risk.detach(),
        }
class GroupDROLoss(nn.Module):
    """
    Group DRO loss with softmax weighting over subjects.
    """
    def __init__(self, eta=0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction="none")
        self.eta = eta

    def forward(self, outputs, labels, env_ids):
        logits = outputs["logits"]
        env_losses = []

        for env in torch.unique(env_ids):
            idx = env_ids == env
            if idx.sum() == 0:
                continue
            env_loss = self.ce(logits[idx], labels[idx]).mean()
            env_losses.append(env_loss)

        env_losses = torch.stack(env_losses)
        weights = torch.softmax(env_losses / self.eta, dim=0)
        total = (weights * env_losses).sum()

        return total, {
            "cls": total.detach(),
            "dro_max": env_losses.max().detach(),
        }
