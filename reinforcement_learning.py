# reinforcement_learning.py
import argparse, json, os, random, math
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------
# 0) Rules loader (so we can derive an action space from your cards)
# ---------------------------------------------------------------------
def load_rules(project_dir: str) -> Dict[str, Any]:
    import importlib.util, sys
    rules_path = Path(project_dir) / "rules_blob.py"
    spec = importlib.util.spec_from_file_location("rules_blob", str(rules_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rules_blob"] = mod
    spec.loader.exec_module(mod)  # type: ignore
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
# 2) Action space & vectorization
# ---------------------------------------------------------------------
def build_action_space(rules: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Global action list used for training. Keep stable across training/runs."""
    acts: List[Dict[str, Any]] = [{"type": "Pass"}]

    # “Play Card” for every card that exists in either deck
    def add_cards(key: str):
        for c in rules.get(key, []):
            no = c.get("No")
            if isinstance(no, int):
                acts.append({"type": "Play Card", "card_no": no})
    add_cards("ISRAEL_CARDS")
    add_cards("ISRAELI_CARDS")  # tolerate alt name
    add_cards("IRAN_CARDS")
    add_cards("IRANIAN_CARDS")

    # (Optional) other primitives your logs may contain
    # acts += [{"type":"Order Airstrike"}, {"type":"Order Ballistic Missile"}]  # uncomment if you log them

    # Ensure uniqueness and stable order
    uniq = []
    seen = set()
    for a in acts:
        k = json.dumps(a, sort_keys=True)
        if k not in seen:
            seen.add(k)
            uniq.append(a)
    return uniq

def vectorize_state(state: Dict[str, Any]) -> torch.Tensor:
    """Simple, robust featurizer. Keep it consistent between train/inference."""
    players = state.get("players", {})
    iz = players.get("israel", {}).get("resources", {})
    ir = players.get("iran", {}).get("resources", {})

    # resources normalized (cap at 30 to be safe)
    def norm(v): return min(float(v or 0), 30.0) / 30.0
    res = [norm(iz.get("pp")), norm(iz.get("ip")), norm(iz.get("mp")),
           norm(ir.get("pp")), norm(ir.get("ip")), norm(ir.get("mp"))]

    # opinions normalized from [-10,10] → [-1,1]
    op = state.get("opinion", {})
    dom = op.get("domestic", {})
    tp  = op.get("third_parties", {}) or op.get("third_party", {}) or op.get("third", {})
    def onorm(v): return max(-10.0, min(10.0, float(v or 0))) / 10.0
    ops = [
        onorm(dom.get("israel", 0)),
        onorm(dom.get("iran",   0)),
        onorm(tp.get("us", 0)),
        onorm(tp.get("russia", 0)),
        onorm(tp.get("china", 0)),
        onorm(tp.get("sa", 0) or tp.get("gcc", 0)),
        onorm(tp.get("un", 0)),
        onorm(tp.get("jordan", 0)),
        onorm(tp.get("turkey", 0)),
    ]

    # turn info
    t = state.get("turn", {})
    turn_no = float(t.get("turn_number") or t.get("number") or 1)
    turn_feat = [min(turn_no, 42.0) / 42.0]
    cur = (t.get("current_player") or "").lower()
    cur_onehot = [1.0 if cur == "israel" else 0.0, 1.0 if cur == "iran" else 0.0]

    # river sizes (gives the net a hint how many options exist)
    iz_river = len(players.get("israel", {}).get("river", []) or [])
    ir_river = len(players.get("iran", {}).get("river", []) or [])
    rivers = [iz_river/10.0, ir_river/10.0]

    feat = res + ops + turn_feat + cur_onehot + rivers
    return torch.tensor(feat, dtype=torch.float32)

# ---------------------------------------------------------------------
# 3) Dataset
# ---------------------------------------------------------------------
class SelfPlayDataset(Dataset):
    def __init__(self, data_dir: str, action_list: List[Dict[str, Any]]):
        self.files = sorted(Path(data_dir).rglob("*.json"))
        self.action_list = action_list
        self.action_index = {json.dumps(a, sort_keys=True): i for i, a in enumerate(action_list)}
        self.samples: List[Tuple[Dict[str,Any], torch.Tensor, float]] = []
        self._load()

    def _load(self):
        for p in self.files:
            try:
                data = json.loads(p.read_text())
            except Exception:
                continue

            # Winner → final reward (+1 Israel, -1 Iran, 0 draw/unknown)
            w = (data.get("winner") or data.get("Winner") or "").lower()
            final_val = 1.0 if w == "israel" else (-1.0 if w == "iran" else 0.0)

            # Allow two log schemas: history[] or plies[]
            turns = data.get("history") or data.get("plies") or []
            for item in turns:
                state = item.get("state") or item.get("State") or {}
                side  = (item.get("side") or item.get("Side") or "").lower()
                # policy could be dict {action_json: prob} or {"index": prob}
                pol   = item.get("mcts_policy") or item.get("policy") or {}
                policy_vec = torch.zeros(len(self.action_list), dtype=torch.float32)
                if isinstance(pol, dict):
                    # try both encodings
                    for k, v in pol.items():
                        try:
                            # case 1: key is serialized action
                            akey = json.dumps(json.loads(k), sort_keys=True)
                            idx = self.action_index.get(akey, None)
                            if idx is not None:
                                policy_vec[idx] = float(v)
                                continue
                        except Exception:
                            pass
                        # case 2: key is integer index stored as string
                        try:
                            idx2 = int(k)
                            if 0 <= idx2 < len(self.action_list):
                                policy_vec[idx2] = float(v)
                        except Exception:
                            pass
                # Normalize & smooth to avoid log(0)
                total = float(policy_vec.sum().item())
                if total <= 0:
                    policy_vec += 1.0 / len(self.action_list)
                else:
                    policy_vec /= total
                eps = 1e-5
                policy_vec = (1.0 - eps*len(self.action_list)) * policy_vec + eps

                # value from perspective of side to move at that state
                val = final_val if side == "israel" else -final_val
                self.samples.append((state, policy_vec, float(val)))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        state, pol, val = self.samples[idx]
        sv = vectorize_state(state)
        return sv, pol, torch.tensor([val], dtype=torch.float32)

# ---------------------------------------------------------------------
# 4) Training
# ---------------------------------------------------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Build action space from rules
    rules = load_rules(args.project_dir)
    actions = build_action_space(rules)
    print(f"[info] action_space size = {len(actions)}")

    # Data
    dataset = SelfPlayDataset(args.data_dir, actions)
    if len(dataset) == 0:
        raise SystemExit(f"No training samples found in: {args.data_dir}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"[info] samples = {len(dataset)} | batches/epoch = {len(loader)}")

    # Model
    # Peek one batch to get input dim
    sv0, _, _ = dataset[0]
    model = PolicyValueNet(state_vector_size=sv0.shape[0], action_space_size=len(actions)).to(device)
    opt = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    policy_loss_fn = nn.KLDivLoss(reduction="batchmean")  # expects log-probs vs probs
    value_loss_fn  = nn.MSELoss()

    best_loss = math.inf
    out_dir = Path(args.model_save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs+1):
        model.train()
        running = 0.0
        for sv, tgt_policy, tgt_value in loader:
            sv = sv.to(device)
            tgt_policy = tgt_policy.to(device)
            tgt_value  = tgt_value.to(device)

            opt.zero_grad()
            logp, v = model(sv)
            loss_p = policy_loss_fn(logp, tgt_policy)
            loss_v = value_loss_fn(v, tgt_value)
            loss = loss_p + loss_v
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            running += loss.item()

        avg = running / len(loader)
        print(f"Epoch {ep}/{args.epochs}  loss={avg:.4f}  (policy+value)")

        # checkpoint
        if avg < best_loss:
            best_loss = avg
            ckpt = {
                "state_dict": model.state_dict(),
                "action_space": actions,
                "state_dim": sv0.shape[0],
                "epoch": ep,
            }
            torch.save(ckpt, out_dir / "persian_incursion_net.pt")

    # Save meta (JSON) too, for easy loading outside PyTorch
    (out_dir / "action_space.json").write_text(json.dumps(actions, indent=2))
    print(f"Saved best model & action_space to: {out_dir}")

# ---------------------------------------------------------------------
# 5) CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train a policy-value net from self-play logs.")
    ap.add_argument("--project_dir", type=str, required=True,
                    help="Directory containing rules_blob.py")
    ap.add_argument("--data_dir", type=str, required=True,
                    help="Directory with self-play JSON files.")
    ap.add_argument("--model_save_dir", type=str, default="trained_models")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--learning_rate", type=float, default=1e-3)
    ap.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    args = ap.parse_args()
    train(args)
