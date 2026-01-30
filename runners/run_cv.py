import os
import json
import numpy as np
import torch


def run_one_fold_cv(
    fold_idx,
    engine,
    loaders,
    epochs,
    out_dir,
    backbone_name,
    normalizer_name=None,
    task_set_name=None,
):

    os.makedirs(out_dir, exist_ok=True)

    history = {
        "train": [],
        "test": [],
    }

    # =========================
    # Training loop
    # =========================
    for epoch in range(epochs):
        train_m = engine.train_one_epoch(loaders["train"])
        history["train"].append({
            "epoch": epoch,
            **train_m,
        })

        eval_out = engine.evaluate(loaders["test"], return_raw=False)
        summary = eval_out["summary"]

        history["test"].append({
            "epoch": epoch,
            "loss": summary["loss"],
            "acc": summary["acc"],
            "bal_acc": summary["balanced_acc"],
            "f1": summary["f1"],
            "auc": summary["auc"],
            "cf": summary["cf"],   
        })

    # =========================
    # Final evaluation (last epoch)
    # =========================
    eval_out = engine.evaluate(loaders["test"], return_raw=True)
    test_final = eval_out["summary"]
    raw = eval_out["raw"]




    torch.save(
        engine.model.state_dict(),
        os.path.join(out_dir, "best.ckpt"),
    )


    np.savez(
        os.path.join(out_dir, "eval_raw.npz"),
        **raw,
    )


    summary_all = {
        "mode": "foldcross",
        "fold": fold_idx,
        "backbone": backbone_name,
        "epochs": epochs,
        "normalizer": normalizer_name,
        "task_set": task_set_name,
        "history": history,
        "test": test_final,
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(summary_all, f, indent=2)

    print(
        f"[DONE] Fold {fold_idx} | "
        f"balanced_acc={summary['balanced_acc']:.4f}"
    )

    return summary_all
