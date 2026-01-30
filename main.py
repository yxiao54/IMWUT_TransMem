# scripts/main.py
import argparse
import os
import random
import numpy as np
import torch

from dataloaders.dataloader import get_dataloaders
from models.factory import get_model
from engine.engine import Engine

from losses.factory import get_loss
from runners.run_cv import run_one_fold_cv




# --------------------------------------------------
# Reproducibility
# --------------------------------------------------
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------
# Baseline checkpoint resolver
# --------------------------------------------------
def resolve_baseline_ckpt(args):
    if args.backbone.lower() == "baseline":
        return None

    ckpt = os.path.join(
        "new_ckpt",
        args.mode,
        "stress",
        "baseline",
        args.task_set,
        f"norm_{args.normalizer}",
        f"fold_{args.fold_idx}",
        "best.ckpt",
    )

    return ckpt if os.path.exists(ckpt) else None



def main(args):

    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------
    # Hyperparameters 
    # ------------------------------
    lr = args.lr 
    batch_size = args.batch
    epochs =args.epoch

    # ------------------------------
    # Dataloaders
    # ------------------------------
    loaders = get_dataloaders(
        mode=args.mode,
        fold_idx=args.fold_idx,
        label_type=args.label_type,

        embedding_mode=args.embedding_mode,
        batch_size=batch_size,
        normalization=args.normalizer,
        
    )

    # ------------------------------
    # Model
    # ------------------------------
    input_dim = 334
    baseline_ckpt = resolve_baseline_ckpt(args)

    model = get_model(
        model_name=args.backbone,
        input_dim=input_dim,
        num_class=2,
        baseline_ckpt=baseline_ckpt,
        embedding_mode=args.embedding_mode,
        groups=args.groups,
        hidden=args.hidden,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        
    )

    loss_module = get_loss(backbone=args.backbone,lambda_sparse=args.lambda_sparse)

    engine = Engine(
        model=model,
        optimizer=optimizer,
        loss_module=loss_module,
        device=device,
    )

    # ------------------------------
    # Output dir 
    # ------------------------------
    out_dir = os.path.join(
        args.log_dir,
        args.mode,
        f"{args.embedding_mode}_{args.label_type}_{args.backbone}_{args.task_set}_{args.normalizer}_{str(args.groups)}_{str(args.hidden)}_{str(args.seed)}",
        f"fold_{args.fold_idx}",
    )
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------
    # Run one fold
    # ------------------------------
    summary = run_one_fold_cv(
        fold_idx=args.fold_idx,
        engine=engine,
        loaders=loaders,
        epochs=epochs,
        out_dir=out_dir,
        backbone_name=args.backbone,
        normalizer_name=args.normalizer,
        task_set_name=args.task_set,
    )

    return 

   


# --------------------------------------------------
# CLI 
# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Foldcross")

    parser.add_argument("--fold_idx", type=int, default=0)
    parser.add_argument("--label_type", type=str, default="craving", choices=["stress", "craving"])
    parser.add_argument("--task_set", type=str, default="set2")
    parser.add_argument("--normalizer", type=str, default="user_ema")
    parser.add_argument("--backbone", type=str, default="ours")
    parser.add_argument("--embedding_mode", type=str, default="small")
    parser.add_argument("--mode", type=str, default="foldcross")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epoch", type=int, default=40)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=32)

    
    parser.add_argument("--lambda_sparse", type=float, default=0.01)

    parser.add_argument("--parquet_path", type=str, default="data/windows.parquet")
    parser.add_argument("--ema_parquet_path", type=str, default="data/ema_window_features.parquet")
    parser.add_argument("--embedding_path", type=str, default="data/embedding.pickle")
    parser.add_argument("--log_dir", type=str, default="logs")


    args = parser.parse_args()
    main(args)
