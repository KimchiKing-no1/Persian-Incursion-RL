# selfplay.py
import json, os, time, random, argparse
from pathlib import Path
import numpy as np
import torch

from features import state_to_features, legal_to_mask, initialize_action_space
from model import load_value_model
from run_dynamic import load_state, play_full_game
from game_engine import GameEngine
from mcts_agent import MCTSAgent
from rules_global import RULES

def run_selfplay(seeds_dir: str, out_dir: str, model_path: str | None, games: int = 50):
    from features import action_space_size
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    initialize_action_space(RULES)
    # Build value model (optional)
    in_dim = state_to_features({"turn":{}, "players":{}, "opinion":{}}).shape[0]
    vnet = load_value_model(model_path, in_dim) if model_path else None

    # Small wrapper to give MCTS the hooks
    def make_agent(engine, **kw):
        from mcts_agent import MCTSAgent
        return MCTSAgent(
            engine=engine,
            simulations=kw.get("simulations", 300),
            value_model=vnet,
            feature_fn=lambda s: state_to_features(s, RULES),
            verbose=False,
        )

    seeds = sorted([p for p in Path(seeds_dir).glob("*.json")])
    if not seeds:
        raise SystemExit(f"No seeds in {seeds_dir}")

    rows = []
    gid = 0
    for _ in range(games):
        seed_path = random.choice(seeds)
        env = make_env(seed_path)  # create your engine/env from seed
        agent = make_agent(env.engine, simulations=300)

        history = []  # [(features, current_player)]
        while not env.is_terminal():
            s = env.state
            history.append( (state_to_features(s), s["turn"]["current_player"]) )
            a = agent.select_action(s)  # your current call
            env = play_one_step(env, a) # apply and advance

        # Final result as z for each stored state, perspective-aware
        z_final = env.outcome_value()  # +1/-1 from current_player who just played? adapt:
        # If your API gives absolute winner, transform per history perspective
        # For now assume z_final is (+1 ISR win, -1 IRN win, 0 draw)
        for x, cur in history:
            z = z_final if cur == "israel" else -z_final
            rows.append({"x": x.tolist(), "z": float(z)})

    # Write one JSONL shard
    shard = out / f"selfplay_{int(time.time())}.jsonl"
    with shard.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[selfplay] wrote {len(rows)} rows to {shard}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--games", type=int, default=50)
    args = ap.parse_args()
    run_selfplay(args.seeds_dir, args.out_dir, args.model_path, args.games)
