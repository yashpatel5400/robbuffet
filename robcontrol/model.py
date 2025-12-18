"""
Train a dynamics predictor on saved LQR datasets and save the model artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robbuffet import OfflineDataset
from robcontrol.data import load_dataset


class DynamicsMLP(nn.Module):
    def __init__(self, theta_dim: int, out_dim: int, hidden: Tuple[int, ...] = (256, 256, 128), dropout: float = 0.1):
        super().__init__()
        layers = []
        in_dim = theta_dim
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(h))
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_loaders(thetas: np.ndarray, C_flat: np.ndarray, batch_size: int, seed: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    thetas = thetas.astype(np.float32)
    C_flat = C_flat.astype(np.float32)
    offline = OfflineDataset(thetas, C_flat, train_frac=0.6, cal_frac=0.2, seed=seed, shuffle=True)
    train_loader = DataLoader(offline.train, batch_size=batch_size, shuffle=True)
    cal_loader = DataLoader(offline.calibration, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(offline.test, batch_size=1, shuffle=False)
    return train_loader, cal_loader, test_loader


def train(model: nn.Module, loader: DataLoader, epochs: int, lr: float, device: str = "cpu"):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        for theta, C_flat in loader:
            theta = theta.to(device=device, dtype=torch.float32)
            C_flat = C_flat.to(device=device, dtype=torch.float32)
            pred = model(theta)
            loss = loss_fn(pred, C_flat)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.to("cpu")


def main():
    parser = argparse.ArgumentParser(description="Train dynamics predictor on saved dataset.")
    parser.add_argument("--dataset", required=True, help="Path to NPZ dataset.")
    parser.add_argument("--out-model", default=None, help="Path to save model .pt file.")
    parser.add_argument("--out-meta", default=None, help="Path to save metadata .json file.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    data = load_dataset(args.dataset)
    thetas = data["thetas"]
    A_true = data["A_true"]
    B_true = data["B_true"]
    n = A_true.shape[1]
    m = B_true.shape[2]
    Cs = np.concatenate([A_true, B_true], axis=2)
    C_flat = Cs.reshape(Cs.shape[0], -1)

    train_loader, cal_loader, test_loader = build_loaders(thetas, C_flat, args.batch_size, args.seed)

    model = DynamicsMLP(theta_dim=thetas.shape[1], out_dim=C_flat.shape[1])
    train(model, train_loader, epochs=args.epochs, lr=args.lr, device=args.device)

    out_model = Path(args.out_model or (Path(args.dataset).with_suffix("") .as_posix() + "_model.pt"))
    out_meta = Path(args.out_meta or (Path(args.dataset).with_suffix("") .as_posix() + "_meta.json"))
    torch.save(model.state_dict(), out_model)
    meta = {
        "theta_dim": int(thetas.shape[1]),
        "state_dim": int(n),
        "control_dim": int(m),
        "output_dim": int(C_flat.shape[1]),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
    }
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved model to {out_model}")
    print(f"Saved metadata to {out_meta}")


if __name__ == "__main__":
    main()
