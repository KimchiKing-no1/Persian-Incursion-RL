# trainer.py
import json, random, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import ValueNet
from features import state_to_features

def load_dataset(data_dir: str, limit_files: int | None = None):
    files = sorted(Path(data_dir).glob("*.jsonl"))
    if limit_files:
        files = files[-limit_files:]
    X, y = [], []
    for p in files:
        for line in p.open("r", encoding="utf-8"):
            r = json.loads(line)
            X.append(np.array(r["x"], dtype=np.float32))
            y.append(float(r["z"]))
    X = np.stack(X, axis=0)
    y = np.array(y, dtype=np.float32)
    return X, y

def train_value(data_dir: str, out_ckpt: str, epochs: int = 3, bs: int = 256, lr: float = 1e-3):
    X, y = load_dataset(data_dir)
    in_dim = X.shape[1]
    model = ValueNet(in_dim)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)

    for ep in range(1, epochs+1):
        idx = torch.randperm(X_t.shape[0])
        X_t, y_t = X_t[idx], y_t[idx]
        for i in range(0, X_t.shape[0], bs):
            xb = X_t[i:i+bs]
            yb = y_t[i:i+bs]
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"[train] epoch {ep}/{epochs} loss={loss.item():.4f}")

    torch.save(model.state_dict(), out_ckpt)
    print(f"[train] saved {out_ckpt}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_ckpt", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    train_value(args.data_dir, args.out_ckpt, args.epochs, args.batch_size, args.lr)
