# train.py

import argparse
import json
import random

import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from data import build_dataloaders
from model import build_mobilenet_v2


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = loss_fn(logits, y)

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(1)

            correct += (preds == y).sum().item()
            total += x.size(0)

    return total_loss / total, correct / total

def train(model, train_loader, val_loader, epochs, lr, device):

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    loss_fn = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(1, epochs + 1):

        model.train()
        run_loss, correct, total = 0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            run_loss += loss.item() * x.size(0)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += x.size(0)

        train_loss = run_loss / total
        train_acc = correct / total

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            loss_fn,
            device
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_acc", train_acc, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_acc", val_acc, step=epoch)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_acc={train_acc:.3f} | val_acc={val_acc:.3f}"
        )

    return history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--experiment_name", default="Pet_Transfer")
    parser.add_argument("--registered_model_name", default="pet_mobilenet")
    parser.add_argument("--run_name")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    mlflow.set_experiment(args.experiment_name)
    experiment = mlflow.get_experiment_by_name(args.experiment_name)

    # Define configurations to run
    # You can add more configurations here
    configs = [
        {"epochs": 1, "lr": 1e-3},
        {"epochs": 3, "lr": 1e-3},
        {"epochs": 3, "lr": 5e-4},
    ]

    for conf in configs:
        # Override args with config values if acceptable, or just use config values directly
        current_epochs = conf["epochs"]
        current_lr = conf["lr"]
        
        # Construct a unique run name
        run_name = f"mobilenet_bs{args.batch_size}_lr{current_lr}_ep{current_epochs}"

        # Check if run exists
        existing_runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{run_name}'",
            run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY
        )
        
        if not existing_runs.empty:
            print(f"⚠️ Run '{run_name}' already exists. Skipping...")
            continue
            
        print(f"🚀 Starting run: {run_name}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Re-build dataloaders/model to ensure clean state (especially model weights)
        train_loader, val_loader, classes = build_dataloaders(
            images_dir=args.data_dir,
            batch_size=args.batch_size,
            seed=args.seed,
        )

        model = build_mobilenet_v2(len(classes))
        model.to(device)

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "model": "mobilenet_v2",
                "epochs": current_epochs,
                "batch_size": args.batch_size,
                "lr": current_lr,
                "seed": args.seed,
                "num_classes": len(classes),
                "dataset": "Oxford-IIIT Pet"
            })

            history = train(
                model,
                train_loader,
                val_loader,
                current_epochs,
                current_lr,
                device,
            )

            with open("labels.json", "w") as f:
                json.dump(classes, f)

            mlflow.log_artifact("labels.json")
            mlflow.log_metric("final_train_acc", history["train_acc"][-1])
            mlflow.log_metric("final_val_acc", history["val_acc"][-1])

            mlflow.pytorch.log_model(
                model,
                artifact_path="model",
                registered_model_name=args.registered_model_name
            )

    print("✅ Training finished")


if __name__ == "__main__":
    main()
