# trainer.py
import json, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model import ValueNet
from features import state_to_features
from rules_global import RULES # Import the global rules object

class ValueDataset(Dataset):
    def __init__(self, data_dir: str, limit_files: int | None = None):
        files = sorted(Path(data_dir).glob("*.jsonl"))
        if limit_files:
            files = files[-limit_files:]
        
        self.samples = []
        for p in files:
            for line in p.open("r", encoding="utf-8"):
                try:
                    r = json.loads(line)
                    # The state is now nested inside the 'state' key from self-play logs
                    state_dict = r.get("state", {})
                    if state_dict: # Ensure state is not empty
                        self.samples.append((state_dict, float(r["z"])))
                except (json.JSONDecodeError, KeyError):
                    continue
    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        state_dict, z_value = self.samples[idx]
        # Pass both state and RULES to the feature function
        feature_vector = state_to_features(state_dict, RULES)
        return feature_vector, np.array([z_value], dtype=np.float32)

def train_value(data_dir: str, out_ckpt: str, epochs: int = 5, bs: int = 256, lr: float = 1e-3):
    # Determine input dimension by creating one feature vector
    # This requires a dummy state and the RULES object
    dummy_state = {"turn": {}, "players": {}, "opinion": {}}
    in_dim = state_to_features(dummy_state, RULES).shape[0]
    
    dataset = ValueDataset(data_dir)
    if not dataset:
        print(f"Warning: No valid data found in {data_dir}. Skipping training.")
        return
        
    loader = DataLoader(dataset, batch_size=bs, shuffle=True)

    model = ValueNet(in_dim)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    print(f"Starting training with {len(dataset)} samples...")
    for ep in range(1, epochs + 1):
        running_loss = 0.0
        for i, (xb, yb) in enumerate(loader):
            pred = model(xb)
            loss = loss_fn(pred, yb.squeeze())
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(loader)
        print(f"[train] epoch {ep}/{epochs} loss={avg_loss:.4f}")

    torch.save(model.state_dict(), out_ckpt)
    print(f"[train] saved model to {out_ckpt}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_ckpt", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    train_value(args.data_dir, args.out_ckpt, args.epochs, args.batch_size, args.lr)
