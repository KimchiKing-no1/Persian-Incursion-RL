# tools/pipeline_turn1.py
# Run inside Docker container; no Colab/Drive.
import os, sys, json, copy, random, subprocess
from pathlib import Path
from typing import Any, Dict

# ---------- Config ----------
# Use env PROJECT_DIR or default to /workspace/codes (works in csh_rts)
PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", ".")).resolve()
SEEDS_DIR   = PROJECT_DIR / "seeds"
OUT_DIR     = PROJECT_DIR / "seeds_turn1_50"
RUNS_DIR    = PROJECT_DIR / "runs_out_selfplay"
SELFPLAY_DIR= PROJECT_DIR / "self_play_data"
MODELS_DIR  = PROJECT_DIR / "models"
STARTER     = os.environ.get("STARTER", "israel")  # "israel" or "iran"

EPISODES    = int(os.environ.get("EPISODES", "50"))
SIMS        = int(os.environ.get("SIMULATIONS", "300"))
MAX_PLIES   = int(os.environ.get("MAX_PLIES", "200"))

# ---------- Helpers ----------
def clamp(v, lo, hi): 
    return max(lo, min(hi, v))

def mutate_turn1(seed: Dict[str, Any], i: int, rng: random.Random) -> Dict[str, Any]:
    s = json.loads(json.dumps(seed))  # deep copy via JSON

    s.setdefault("turn", {})
    s["turn"]["turn_number"] = 1
    s["turn"]["phase"] = "morning"
    s["turn"]["current_player"] = STARTER

    s.setdefault("players", {}).setdefault("israel", {}).setdefault("resources", {})
    s.setdefault("players", {}).setdefault("iran",   {}).setdefault("resources", {})
    s.setdefault("opinion", {}).setdefault("domestic", {})
    s.setdefault("opinion", {}).setdefault("third_parties", {})

    for side in ("israel", "iran"):
        res = s["players"][side]["resources"]
        for k in list(res.keys()):
            delta = rng.choice([-1, 0, 0, 1])
            res[k] = clamp(int(res[k]), 0, 15)
            res[k] = clamp(int(res[k]) + delta, 0, 15)

    for k in list(s["opinion"]["domestic"].keys()):
        s["opinion"]["domestic"][k] = clamp(int(s["opinion"]["domestic"][k]) + rng.choice([-1,0,1]), -5, 5)
    for k in list(s["opinion"]["third_parties"].keys()):
        s["opinion"]["third_parties"][k] = clamp(int(s["opinion"]["third_parties"][k]) + rng.choice([-1,0,1]), -5, 5)

    s["_meta"] = {"seed_id": i+1, "turn_forced": 1, "phase_forced": "morning", "starter": STARTER}
    return s

def monkey_patch_mcts(mcts_path: Path):
    if not mcts_path.exists():
        print(f"[warn] {mcts_path} not found; skipping patch.")
        return
    orig = mcts_path.read_text(encoding="utf-8")
    MARK = "# === BEGIN PATCH: no-op warnings ==="
    if MARK in orig:
        print("[info] patch already present; skipping.")
        return
    patch = f"""
{MARK}
from typing import Any, Dict
import copy

def _safe_apply_patched(self, state: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
    if not getattr(self, "_patch_notice_shown", False):
        print("[PATCH] _safe_apply_patched active")
        self._patch_notice_shown = True
    if hasattr(self.engine, "apply_action"):
        try:
            before = self._state_key(state)
            new_state = self.engine.apply_action(state, action)
            after = self._state_key(new_state)
            if before == after and action.get("type") != "Pass" and getattr(self, "verbose", False):
                print("[WARN] apply_action produced no state change:", action)
            return new_state
        except Exception as e:
            if getattr(self, "verbose", False):
                print(f"[MCTS] apply_action error {{e}}; falling back to no-op.")
    return state

def _expand_patched(self, node: Node) -> Node:
    action = node.unexpanded_actions.pop(0)
    before = self._state_key(node.state)
    child_state = self._safe_apply(copy.deepcopy(node.state), action)
    after = self._state_key(child_state)
    if before == after and action.get("type") != "Pass" and getattr(self, "verbose", False):
        print("[WARN] EXPAND edge is no-op:", action)
    child = self._make_node(child_state, parent=node, incoming_action=action)
    node.children.append(child)
    return child

MCTSAgent._safe_apply = _safe_apply_patched
MCTSAgent._expand = _expand_patched
# === END PATCH: no-op warnings ===
""".rstrip() + "\n"
    with mcts_path.open("a", encoding="utf-8") as f:
        f.write("\n" + patch)
    print("[info] Patched mcts_agent.py")

# ---------- Main ----------
def main():
    print(f"[cfg] PROJECT_DIR={PROJECT_DIR}")
    for p in (SEEDS_DIR, OUT_DIR, RUNS_DIR, SELFPLAY_DIR, MODELS_DIR):
        p.mkdir(parents=True, exist_ok=True)

    # base seed
    base_seed_path = SEEDS_DIR / "1.json"
    if not base_seed_path.exists():
        existing = sorted(SEEDS_DIR.glob("*.json"))
        if existing:
            base_seed_path = existing[0]
        else:
            demo = {
                "turn": {"turn_number": 1, "current_player": "israel", "phase": "morning"},
                "players": {
                    "israel": {"resources": {"pp": 3, "ip": 3, "mp": 3}},
                    "iran":   {"resources": {"pp": 2, "ip": 2, "mp": 2}},
                },
                "opinion": {
                    "domestic": {"israel": 0, "iran": 0},
                    "third_parties": {"UN": 0, "US": 0}
                }
            }
            base_seed_path.write_text(json.dumps(demo, indent=2), encoding="utf-8")
            print(f"[seed] wrote demo: {base_seed_path}")

    base = json.loads(base_seed_path.read_text(encoding="utf-8"))
    rng = random.Random(42)

    # generate 50 seeds
    paths = []
    for i in range(50):
        s = mutate_turn1(base, i, rng)
        outp = OUT_DIR / f"seed_turn1_{i+1:02d}.json"
        outp.write_text(json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8")
        paths.append(str(outp))
    print(f"[done] wrote 50 seeds → {OUT_DIR}")
    print("First 5:\n - " + "\n - ".join(paths[:5]))

    # patch mcts if present
    monkey_patch_mcts(PROJECT_DIR / "mcts_agent.py")

    # run self-play
    run_dynamic = PROJECT_DIR / "run_dynamic.py"
    if run_dynamic.exists():
        cmd = [
            sys.executable, str(run_dynamic),
            "--seeds_dir", str(OUT_DIR),
            "--mode", "self_play",
            "--episodes", str(EPISODES),
            "--simulations", str(SIMS),
            "--max_plies", str(MAX_PLIES),
            "--save_dir", str(RUNS_DIR),
            "--self_play_dir", str(SELFPLAY_DIR),
        ]
        print("[run]", " ".join(cmd))
        subprocess.check_call(cmd)
    else:
        print(f"[warn] {run_dynamic} not found; skipping self-play.")

    # train
    trainer = PROJECT_DIR / "trainer.py"
    if trainer.exists():
        model_out = MODELS_DIR / "pv_model_v1.pt"
        cmd = [
            sys.executable, str(trainer),
            "--data_dir", str(SELFPLAY_DIR),
            "--out_path", str(model_out),
            "--epochs", "5",
            "--batch_size", "256",
        ]
        print("[train]", " ".join(cmd))
        subprocess.check_call(cmd)
    else:
        print(f"[warn] {trainer} not found; skipping training.")

if __name__ == "__main__":
    main()
