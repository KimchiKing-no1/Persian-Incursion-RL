import argparse, json, os, random, time, csv
from pathlib import Path
from typing import Optional, Dict, Any, List

from game_engine import GameEngine
from mcts_agent import MCTSAgent
from rules_global import RULES

# ----------------------------- Optional Gemini hook -----------------------------
def real_gemini_caller(prompt: str, *, temperature: float = 0.4, max_tokens: int = 1024) -> str:
    """
    If you pass --use_gemini, we try to call Gemini with key in Colab secrets.
    Otherwise this function is never used.
    """
    try:
        from google.colab import userdata  # type: ignore
        import google.generativeai as genai  # type: ignore

        gemini_api_key = userdata.get('GEMINI_API_KEY')
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in Colab secrets. Add it via Colab > ⚙️ > Secrets.")

        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-2.5-pro')

        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        resp = model.generate_content(prompt, safety_settings=safety_settings)
        return getattr(resp, "text", "") or ""
    except Exception as e:
        print(f"[LLM API Error] {e} — falling back to dummy choice.")
        return json.dumps({"value": 0.0, "policy": {"0": 100}})

# ----------------------------- Utilities ---------------------------------------
def find_seed_paths(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*.json") if p.is_file()])

def load_state(path: Path) -> Dict[str, Any]:
    """
    Load a game state JSON and normalize to what GameEngine expects.
    """
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    state: Dict[str, Any] = {
        "turn": raw.get("turn", {}),
        "players": raw.get("players", {}),
        "opinion": raw.get("opinion", {}),
        "upgrades": raw.get("upgrades", {}),
        "ballistic_missiles": raw.get("ballistic_missiles", {}),
        "active_events_queue": raw.get("active_events_queue", []),
    }

    # Engine expects target_damage_status; remap if user used target_impacts
    if "target_impacts" in raw:
        state["target_damage_status"] = raw["target_impacts"]

    # Normalize air OOB if present
    state.setdefault("squadrons", {"israel": {}, "iran": {}})
    air = raw.get("air", {})
    for side_key, list_key in (("israel", "israel_squadrons"), ("iran", "iran_squadrons")):
        for sq in air.get(list_key, []) or []:
            sq_id = sq.get("id")
            if sq_id:
                state["squadrons"].setdefault(side_key, {})[sq_id] = sq

    # Ensure minimal player scaffolding
    state.setdefault("players", {}).setdefault("israel", {}).setdefault("resources", {})
    state.setdefault("players", {}).setdefault("iran", {}).setdefault("resources", {})
    return state

def save_json(data, out_path: Path) -> None:
    """
    JSON-safe dump: strips engine RNGs/functions/sets/tuples etc.
    """
    import random

    def _safe(obj):
        if isinstance(obj, dict):
            return {k: _safe(v) for k, v in obj.items() if k not in ("_rng", "rng")}
        if isinstance(obj, list):
            return [_safe(v) for v in obj]
        if isinstance(obj, tuple) or isinstance(obj, set):
            return [_safe(v) for v in obj]
        if isinstance(obj, random.Random):
            return "<Random>"
        try:
            json.dumps(obj)
            return obj
        except TypeError:
            return str(obj)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(_safe(data), f, indent=2, ensure_ascii=False)

def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

# ----------------------------- One-step runner ---------------------------------
def run_one_step(engine: GameEngine, agent: MCTSAgent, state: Dict[str, Any]) -> Dict[str, Any]:
    return agent.choose_action(state)

# ----------------------------- Full game runner --------------------------------
def play_full_game(
    engine: GameEngine,
    agent_israel: MCTSAgent,
    agent_iran: MCTSAgent,
    state: Dict[str, Any],
    max_plies: int = 200,
) -> Dict[str, Any]:
    """
    Alternate turns using agents until terminal or max_plies.
    Returns: {"winner": str|None, "plies": list, "terminal_state": dict}
    """
    plies: List[Dict[str, Any]] = []
    for ply in range(max_plies):
        cur = (state.get("turn", {}).get("current_player") or "").lower()
        if cur not in ("israel", "iran"):
            break

        # run_dynamic.py (AFTER)

        agent = agent_israel if cur == "israel" else agent_iran
        # Unpack the action and the policy from the agent's return value
        action, mcts_policy = agent.choose_action(state)
        
        # Add the mcts_policy to the data being saved for this ply
        plies.append({
            "ply": ply + 1, 
            "side": cur, 
            "action": action, 
            "mcts_policy": mcts_policy
        })

        state = engine.apply_action(state, action)
        t = state.get("turn", {})
        cur = t.get("current_player")
        phase = t.get("phase")
        tnum = t.get("turn_number")
        isr = state.get("players", {}).get("israel", {}).get("resources", {})
        irn = state.get("players", {}).get("iran", {}).get("resources", {})
        print(f"[TRACE] ply={ply+1} turn#{tnum} phase={phase} next={cur} "
              f"ISR(pp={isr.get('pp')},ip={isr.get('ip')},mp={isr.get('mp')}) "
              f"IRN(pp={irn.get('pp')},ip={irn.get('ip')},mp={irn.get('mp')})")

        winner = engine.is_game_over(state)
        if winner is not None:
            return {"winner": winner, "plies": plies, "terminal_state": state}

    return {"winner": None, "plies": plies, "terminal_state": state}

# ==============================================================================

def main():
    ap = argparse.ArgumentParser(description="Dynamic batch runner for Persian Incursion MCTSAgent.")
    ap.add_argument("--seeds_dir", type=str, required=True, help="Folder containing *.json initial states.")
    ap.add_argument("--mode", choices=["one_step", "full_game", "self_play"], default="full_game")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--shuffle", action="store_true", help="Shuffle seeds before running.")
    ap.add_argument("--simulations", type=int, default=100, help="MCTS simulations per move.")
    ap.add_argument("--c_puct", type=float, default=1.4, help="PUCT exploration constant (passed to agent).")
    ap.add_argument("--max_plies", type=int, default=200, help="Max plies in full_game mode.")
    ap.add_argument("--save_dir", type=str, default="runs_out", help="Where to save logs/CSV.")
    ap.add_argument("--self_play_dir", type=str, default="self_play_data", help="Where to save self-play data.")
    ap.add_argument("--use_gemini", action="store_true", help="Use Gemini for rollout guidance.")

    # === NEW: optional value model checkpoint ===
    ap.add_argument("--value_ckpt", type=str, default="", help="Path to a torch checkpoint for a value model (optional).")

    args = ap.parse_args()

    seeds = find_seed_paths(Path(args.seeds_dir))
    if not seeds:
        raise SystemExit(f"No *.json seeds found in: {args.seeds_dir}")
    if args.shuffle:
        random.shuffle(seeds)

    out_root = Path(args.save_dir); out_root.mkdir(parents=True, exist_ok=True)
    if args.mode == "self_play":
        Path(args.self_play_dir).mkdir(parents=True, exist_ok=True)

    summary_csv = out_root / f"summary_{args.mode}.csv"
    summary_rows: List[List[Any]] = []

    engine = GameEngine(rules=RULES)
    gem_fn = real_gemini_caller if args.use_gemini else None

    # === NEW: try to load value model + features (optional) ===================
    value_model = None
    state_to_features = None
    if args.value_ckpt:
        try:
            from features import state_to_features as _stf
            from model import load_value_model
            import numpy as np  # ensure available
            # Build an empty state to infer input dim (safe default fields)
            dummy = {"turn":{}, "players":{}, "opinion":{}}
            in_dim = _stf(dummy).shape[0]
            if os.path.exists(args.value_ckpt):
                value_model = load_value_model(args.value_ckpt, in_dim)
                state_to_features = _stf
                print(f"[run_dynamic] Loaded value model: {args.value_ckpt} (in_dim={in_dim})")
            else:
                print(f"[run_dynamic] --value_ckpt provided but file not found: {args.value_ckpt} (running without model)")
        except Exception as e:
            print(f"[run_dynamic] Could not load value model/features ({e}); running without model.")
            value_model = None
            state_to_features = None
    # ==========================================================================

    israel_agent = MCTSAgent(
        engine=engine, side="israel",
        simulations=args.simulations, c_puct=args.c_puct,
        gemini=gem_fn, verbose=True,
        value_model=value_model, feature_fn=state_to_features
    )
    iran_agent   = MCTSAgent(
        engine=engine, side="iran",
        simulations=args.simulations, c_puct=args.c_puct,
        gemini=gem_fn, verbose=True,
        value_model=value_model, feature_fn=state_to_features
    )
    model_path = "/content/drive/MyDrive/PersianIncursionAI/models/pv_model_v1.pt"
    if hasattr(israel_agent, "_attach_pv_model"):
        israel_agent._attach_pv_model(model_path)
        iran_agent._attach_pv_model(model_path)
        total_eps = min(args.episodes, len(seeds))
        print(f"[{now()}] Starting {total_eps} episodes in mode={args.mode}.")

    for ep in range(total_eps):
        seed_path = seeds[ep]
        initial_state = load_state(seed_path)

        # Use initial_state only; let engine normalize
        state = engine.bootstrap_rivers(initial_state)

        print(f"\n--- Episode {ep+1}/{total_eps} using seed: {seed_path.name} ---")

        if args.mode == "one_step":
            cur = (state.get("turn", {}).get("current_player") or "").lower()
            agent = israel_agent if cur == "israel" else iran_agent
            action = run_one_step(engine, agent, state)
            out = {"seed": seed_path.name, "action": action, "state_after": engine.apply_action(state, action)}
            save_json(out, out_root / f"one_step_{seed_path.stem}.json")
            print(f"[{ep+1:04d}] One-step action: {action}")
            summary_rows.append([ep + 1, seed_path.name, None, 1])
            continue

        # full_game / self_play
        game_log = play_full_game(
            engine=engine,
            agent_israel=israel_agent,
            agent_iran=iran_agent,
            state=state,
            max_plies=args.max_plies,
        )

        plies_count = len(game_log.get("plies", []))
        winner = game_log.get("winner")
        save_json(game_log, out_root / f"game_{ep+1:04d}_{seed_path.stem}.json")
        print(f"[{ep+1:04d}] Game Over. Winner: {winner} in {plies_count} plies.")
        summary_rows.append([ep + 1, seed_path.name, winner, plies_count])

        if args.mode == "self_play":
            training_data = {
                "game_id": f"game_{ep+1:04d}_{int(time.time())}",
                "winner": winner,
                "plies": game_log.get("plies", []),
            }
            save_json(training_data, Path(args.self_play_dir) / f"{training_data['game_id']}.json")
            print(f"[{ep+1:04d}] Self-play data saved.")

    with summary_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "seed_file", "winner", "plies"])
        writer.writerows(summary_rows)

    print(f"[{now()}] Done. Summary written to: {summary_csv}")

if __name__ == "__main__":
    main()
