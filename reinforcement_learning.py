# reinforcement_learning.py
import argparse, json, os, random, math, sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import importlib.util

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Import the new feature/action space functions
from features import initialize_action_space, state_to_features, action_to_id, action_space_size, ACTION_MAP

# ---------------------------------------------------------------------
# 0) Rules loader (so we can derive an action space from your cards)
# ---------------------------------------------------------------------
def load_rules(project_dir: str) -> Dict[str, Any]:
    rules_path = Path(project_dir) / "rules_blob.py"
    spec = importlib.util.spec_from_file_location("rules_blob", str(rules_path))
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Could not find or load rules_blob.py at {rules_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rules_blob"] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, "RULES")

# ---------------------------------------------------------------------
# 1) Model
# ---------------------------------------------------------------------
class PolicyValueNet(nn.Module):
    def __init__(self, state_vector_size: int, action_space_size: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_vector_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(256, action_space_size),
            nn.LogSoftmax(dim=-1),   # for KLDivLoss
        )
        self.value_head = nn.Sequential(
            nn.Linear(256, 1),
            nn.Tanh(),               # -1 (Iran win) .. +1 (Israel win)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        log_policy = self.policy_head(h)
        value = self.value_head(h)
        return log_policy, value

# ---------------------------------------------------------------------
# 2) Dataset
# ---------------------------------------------------------------------
class SelfPlayDataset(Dataset):
    def __init__(self, data_dir: str, rules: Dict[str, Any]):
        self.files = sorted(Path(data_dir).rglob("*.json"))
        self.rules = rules
        self.samples: List[Tuple[Dict[str,Any], torch.Tensor, float]] = []
        self._load()

    def _load(self):
        for p in self.files:
            try:
                data = json.loads(p.read_text())
            except Exception:
                continue

            w = (data.get("winner") or "").lower()
            final_val = 1.0 if w == "israel" else (-1.0 if w == "iran" else 0.0)

            turns = data.get("history") or data.get("plies") or []
            for item in turns:
                state = item.get("state") or {}
                side = (item.get("side") or "").lower()
                pol = item.get("mcts_policy") or {}
                
                policy_vec = torch.zeros(action_space_size(), dtype=torch.float32)
                if isinstance(pol, dict):
                    for action_str, prob in pol.items():
                        try:
                            action_dict = json.loads(action_str)
                            # Add side info for card plays to find the correct ID
                            if action_dict.get("type") == "Play Card":
                                action_dict["_side_"] = side
                            
                            idx = action_to_id(action_dict)
                            policy_vec[idx] = float(prob)
                        except Exception:
                            continue
                
                total = float(policy_vec.sum().item())
                if total > 0:
                    policy_vec /= total
                
                val = final_val if side == "israel" else -final_val
                self.samples.append((state, policy_vec, float(val)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        state, pol, val = self.samples[idx]
        # Pass the rules to state_to_features
        sv = state_to_features(state, self.rules)
        return sv, pol, torch.tensor([val], dtype=torch.float32)

# ---------------------------------------------------------------------
# 3) Training
# ---------------------------------------------------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Initialize action space from rules
    rules = load_rules(args.project_dir)
    initialize_action_space(rules)
    print(f"[info] action_space size = {action_space_size()}")

    # Data
    dataset = SelfPlayDataset(args.data_dir, rules)
    if len(dataset) == 0:
        raise SystemExit(f"No training samples found in: {args.data_dir}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"[info] samples = {len(dataset)} | batches/epoch = {len(loader)}")

    # Model
    sv0, _, _ = dataset[0]
    model = PolicyValueNet(state_vector_size=sv0.shape[0], action_space_size=action_space_size()).to(device)
    opt = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    policy_loss_fn = nn.KLDivLoss(reduction="batchmean")
    value_loss_fn  = nn.MSELoss()

    best_loss = math.inf
    out_dir = Path(args.model_save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for sv, tgt_policy, tgt_value in loader:
            sv, tgt_policy, tgt_value = sv.to(device), tgt_policy.to(device), tgt_value.to(device)

            opt.zero_grad()
            logp, v = model(sv)
            loss_p = policy_loss_fn(logp, tgt_policy)
            loss_v = value_loss_fn(v.squeeze(), tgt_value.squeeze())
            loss = loss_p + loss_v
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        print(f"Epoch {ep}/{args.epochs}  loss={avg_loss:.4f}  (policy+value)")

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt = {
                "state_dict": model.state_dict(),
                "action_space": ACTION_MAP, # Save the initialized action map
                "state_dim": sv0.shape[0],
                "epoch": ep,
            }
            torch.save(ckpt, out_dir / "persian_incursion_pv_net.pt")

    (out_dir / "action_space.json").write_text(json.dumps(ACTION_MAP, indent=2))
    print(f"Saved best model & action_space to: {out_dir}")

# ---------------------------------------------------------------------
# 4) CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train a policy-value net from self-play logs.")
    ap.add_argument("--project_dir", type=str, default=".", help="Directory with rules_blob.py")
    ap.add_argument("--data_dir", type=str, required=True, help="Directory with self-play JSON logs.")
    ap.add_argument("--model_save_dir", type=str, default="trained_models")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--learning_rate", type=float, default=1e-3)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    train(args)
