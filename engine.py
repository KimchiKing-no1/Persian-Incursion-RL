# engine.py  (프로젝트 루트에 새 파일)
import json
from typing import Dict, Any
from game_engine import GameEngine as Engine  # 네 GameEngine을 Engine 이름으로 노출

def make_initial_state(seed_path: str = "testing/seed_turn1_01.json") -> Dict[str, Any]:
    """
    testing/seed_turn1_01.json을 읽고, 엔진에 필요한 최소 필드를 보정한 뒤 반환.
    orchestrator나 run_dynamic이 그대로 쓸 수 있게 state만 리턴한다.
    """
    with open(seed_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    eng = Engine()

    # 상태에 턴/리버 등 필수 구조가 비어 있으면 채워준다.
    # (seed가 이미 가지고 있으면 아래 호출은 안전하게 no-op에 가깝게 동작)
    state = eng.bootstrap_rivers(state)

    # 턴 메타 보정(혹시 없을 경우를 대비)
    t = state.setdefault("turn", {})
    t.setdefault("turn_number", 1)
    t.setdefault("phase", "morning")
    t.setdefault("current_player", "israel")
    t.setdefault("per_impulse_card_played", {"israel": False, "iran": False})
    t.setdefault("consecutive_passes", 0)

    # 큐와 로그 기본값
    state.setdefault("active_events_queue", [])
    state.setdefault("log", [])

    return state
