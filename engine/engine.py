import torch
import numpy as np
from torch import nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return 0.0
    return roc_auc_score(y_true, y_score)


class Engine:
    def __init__(self, model, optimizer, loss_module, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_module = loss_module   # train-time loss (IRM / DRO / etc.)
        self.device = device

    # ======================================================
    # Train ONE epoch
    # ======================================================
    def train_one_epoch(self, loader):
        self.model.train()

        all_y, all_yhat, all_prob = [], [], []
        total_loss = 0.0
        total_steps = 0

        for batch in loader:
            (
                data,
                label,
                emb_good,
                emb_bad,
                user_id,
                task,
                user_str,
            ) = batch

            data = data.to(self.device).float()
            label = label.to(self.device).long()
            emb_good = emb_good.to(self.device).float()
            emb_bad = emb_bad.to(self.device).float()
            user_id = user_id.to(self.device).long()

            self.optimizer.zero_grad()

            output = self.model(data, emb_good, emb_bad)
            logits = output["logits"] if isinstance(output, dict) else output

            # ? train-time loss£¨ÔÊÐí env-aware£©
            loss = self.loss_module(
                output,
                label,
                env_ids=user_id,
            )

            if isinstance(loss, tuple):
                loss = loss[0]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_steps += 1

            with torch.no_grad():
                yhat = logits.argmax(dim=1)
                prob = torch.softmax(logits, dim=1)[:, 1]

                all_y.extend(label.cpu().tolist())
                all_yhat.extend(yhat.cpu().tolist())
                all_prob.extend(prob.cpu().tolist())

        metrics = {
            "loss": total_loss / max(total_steps, 1),
            "acc": accuracy_score(all_y, all_yhat),
            "balanced_acc": balanced_accuracy_score(all_y, all_yhat),
            "f1": f1_score(all_y, all_yhat, average="macro"),
            "auc": safe_auc(all_y, all_prob),
        }
        return metrics

    # ======================================================
    # Unified Evaluate (summary + raw)
    # ======================================================
    def evaluate(self, loader, return_raw=True):
        self.model.eval()

        all_y = []
        all_yhat = []
        all_prob = []
        all_logits = []
        all_loss = []
        all_users = []
        all_user_ids = []
        all_tasks = []

        total_loss = 0.0
        total_steps = 0

        ce = nn.CrossEntropyLoss(reduction="none")

        with torch.no_grad():
            for batch in loader:
                (
                    data,
                    label,
                    emb_good,
                    emb_bad,
                    user_id,
                    task,
                    user_str,
                ) = batch

                data = data.to(self.device).float()
                label = label.to(self.device).long()
                emb_good = emb_good.to(self.device).float()
                emb_bad = emb_bad.to(self.device).float()

                output = self.model(data, emb_good, emb_bad)
                logits = output["logits"] if isinstance(output, dict) else output

                yhat = logits.argmax(dim=1)
                prob = torch.softmax(logits, dim=1)[:, 1]

                loss_vec = ce(logits, label)
                loss = loss_vec.mean()

                total_loss += loss.item()
                total_steps += 1

                all_y.extend(label.cpu().tolist())
                all_yhat.extend(yhat.cpu().tolist())
                all_prob.extend(prob.cpu().tolist())
                all_logits.extend(logits.cpu().tolist())
                all_loss.extend(loss_vec.cpu().tolist())

                all_user_ids.extend(user_id.cpu().tolist())
                all_tasks.extend(list(task))
                all_users.extend(list(user_str))

        # =====================
        # summary metrics
        # =====================
        y_true = np.array(all_y)
        y_pred = np.array(all_yhat)
        prob = np.array(all_prob)

        cf = confusion_matrix(y_true, y_pred, labels=[0, 1])

        summary = {
            "loss": total_loss / max(total_steps, 1),
            "acc": accuracy_score(y_true, y_pred),
            "balanced_acc": balanced_accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, average="macro"),
            "auc": safe_auc(y_true, prob),
            "cf": cf.tolist(),
        }

        if not return_raw:
            return {"summary": summary}

        raw = {
            "y_true": np.array(all_y),
            "y_pred": np.array(all_yhat),
            "prob": np.array(all_prob),
            "logits": np.array(all_logits),
            "loss": np.array(all_loss),
            "user": np.array(all_users),
            "user_id": np.array(all_user_ids),
            "task": np.array(all_tasks),
        }

        return {
            "summary": summary,
            "raw": raw,
        }
