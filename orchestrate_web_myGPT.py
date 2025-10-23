# orchestrate_web_myGPT.py
from __future__ import annotations
import json, time, re, textwrap
from pathlib import Path
from typing import Dict, Any, List
from playwright.sync_api import sync_playwright, TimeoutError

# --- CHANGE THESE ---
MYGPT_URL = "https://chatgpt.com/g/g-7G3RSLrVa-persian-incursion-strategist"
PERSISTENT_DIR = str(Path("./playwright-data").absolute())
MAX_PLIES = 21

# --- import your engine in-process (edit to your project) ---
from engine import Engine, make_initial_state

engine = Engine()
state = make_initial_state("testing/seed_turn1_01.json")


def j(obj): return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

def make_prompt(state: Dict[str,Any], side: str, legal: List[Dict[str,Any]]) -> str:
    actions_view = [{"i": i, "a": a} for i, a in enumerate(legal)]
    msg = f"""
    You are playing side="{side}" in Persian Incursion.

    State (compact JSON):
    {j(state)}

    Legal actions (choose ONE):
    {j(actions_view)}

    Respond with ONLY one of:
    1) a minimal JSON object exactly equal to a listed action; or
    2) {{"i": <index>}} where <index> is one of the 'i' above.
    Return ONLY JSON. No extra text.
    """
    return textwrap.dedent(msg).strip()

def extract_json(text: str) -> str:
    blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, re.I)
    candidates = blocks if blocks else [text]
    for chunk in reversed(candidates):
        s, e = chunk.find("{"), chunk.rfind("}")
        if s!=-1 and e> s:
            return chunk[s:e+1]
    nums = re.findall(r"\b\d+\b", text.strip())
    if nums:
        return json.dumps({"i": int(nums[-1])})
    raise ValueError("No JSON found")

def pick_action(reply_text: str, legal: List[Dict[str,Any]]) -> Dict[str,Any]:
    obj = json.loads(extract_json(reply_text))
    if isinstance(obj, dict) and "i" in obj:
        idx = obj["i"]
        if isinstance(idx, int) and 0 <= idx < len(legal): return legal[idx]
        raise ValueError(f"Index out of range: {idx}")
    for a in legal:
        if obj == a: return a
    # tolerant fallback: same type & same keys
    if isinstance(obj, dict) and "type" in obj:
        for a in legal:
            if a.get("type")==obj.get("type") and set(a.keys())==set(obj.keys()):
                return a
    raise ValueError("Reply JSON did not match any legal action")

def get_textarea(page):
    for sel in ["textarea","div[contenteditable='true']","form textarea"]:
        loc = page.locator(sel)
        if loc.count(): return loc.first
    raise RuntimeError("chat textbox not found")

def send_prompt(page, prompt: str):
    ta = get_textarea(page)
    ta.click(); ta.fill(""); ta.type(prompt); ta.press("Enter")

def wait_last_assistant(page) -> str:
    sel = "[data-message-author-role='assistant'], .markdown"
    for _ in range(240):  # up to ~120s
        try:
            loc = page.locator(sel)
            if loc.count():
                txt = loc.last.inner_text(timeout=1000)
                if txt.strip(): return txt
        except TimeoutError:
            pass
        page.wait_for_timeout(500)
    raise TimeoutError("Assistant reply not found")

def main():
    eng = Engine()
    state = make_initial_state()
    transcript = []
    turn_log = Path(f"duel_{int(time.time())}.jsonl")

    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(PERSISTENT_DIR, headless=False)
        page = ctx.new_page()
        page.goto(MYGPT_URL, wait_until="networkidle")
        print("[INFO] If login is needed, log in in the browser. Then press ENTER here.")
        input()

        for ply in range(1, MAX_PLIES+1):
            winner = eng.is_game_over(state)
            if winner is not None:
                print(f"[END] winner={winner}")
                break

            side = (state.get("turn",{}).get("current_player") or "").lower()
            legal = eng.get_legal_actions(state, side) or [{"type":"Pass"}]

            prompt = make_prompt(state, side, legal)
            print(f"[PLY {ply}] {side} → sending to MyGPT...")
            send_prompt(page, prompt)

            try:
                reply = wait_last_assistant(page)
            except TimeoutError:
                # one retry
                send_prompt(page, "Return ONLY JSON for one valid action, or {\"i\": <index>} from the list.")
                reply = wait_last_assistant(page)

            try:
                action = pick_action(reply, legal)
            except Exception:
                # repair prompt retry
                send_prompt(page, "Your previous reply was invalid. Return ONLY one valid JSON action or {\"i\": <index>}.")
                reply = wait_last_assistant(page)
                action = pick_action(reply, legal)

            new_state = eng.apply_action(state, action)
            row = {"ply": ply, "side": side, "action": action}
            transcript.append(row)
            with turn_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False)+"\n")

            print(f"[OK] applied: {action}")
            state = new_state

        # final save (you can load this later)
        out = {
            "winner": eng.is_game_over(state),
            "plies": transcript,
            "final_state": state,
        }
        out_name = turn_log.with_suffix(".final.json")
        out_name.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[SAVED] {out_name}")
        ctx.close()

if __name__ == "__main__":
    main()
