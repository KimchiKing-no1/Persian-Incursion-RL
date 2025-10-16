# features.py
from typing import Dict, Any, List, Tuple
import numpy as np

ACTION_TYPES = [
    "Pass", "Play Card", "Order Special Warfare",
    "Order Ballistic Missile", "Order Terror Attack"
]
# Max discrete actions per type (safe over-approx)
MAX_ACTIONS_PER_TYPE = {
    "Pass": 1, "Play Card": 40, "Order Special Warfare": 2,
    "Order Ballistic Missile": 5, "Order Terror Attack": 1
}

def action_space_size() -> int:
    return sum(MAX_ACTIONS_PER_TYPE.values())

def action_to_id(a: Dict[str, Any]) -> int:
    """
    Deterministic mapping from an action dict -> flat action id.
    """
    offset = 0
    t = a.get("type", "Pass")
    for tname in ACTION_TYPES:
        cap = MAX_ACTIONS_PER_TYPE[tname]
        if t == tname:
            # choose index within type
            if t == "Pass":
                idx = 0
            elif t == "Play Card":
                idx = int(a.get("card_id", 0))
                idx = max(0, min(cap-1, idx))
            elif t in ("Order Special Warfare", "Order Ballistic Missile", "Order Terror Attack"):
                idx = 0  # refine if you later have sub-ids
            else:
                idx = 0
            return offset + idx
        offset += cap
    return 0

def id_to_action(aid: int) -> Dict[str, Any]:
    """
    Only needed if you’ll ever sample from the policy directly.
    """
    offset = 0
    for tname in ACTION_TYPES:
        cap = MAX_ACTIONS_PER_TYPE[tname]
        if offset <= aid < offset + cap:
            idx = aid - offset
            if tname == "Pass":
                return {"type": "Pass"}
            if tname == "Play Card":
                return {"type": "Play Card", "card_id": idx}
            return {"type": tname}
        offset += cap
    return {"type": "Pass"}

def state_to_features(s: Dict[str, Any]) -> np.ndarray:
    """
    VERY SMALL starter feature vector.
    Expand with anything predictive (locations, readiness, etc.)
    """
    t = s.get("turn", {})
    p_isr = s.get("players", {}).get("israel", {}).get("resources", {})
    p_irn = s.get("players", {}).get("iran", {}).get("resources", {})
    op = s.get("opinion", {})
    dom = op.get("domestic", {})
    tp = op.get("third_parties", {})

    vec = [
        float(t.get("turn_number", 1)),
        1.0 if t.get("phase") == "morning" else 0.0,
        1.0 if t.get("phase") == "afternoon" else 0.0,
        1.0 if t.get("phase") == "night" else 0.0,
        1.0 if t.get("current_player") == "israel" else 0.0,
        1.0 if t.get("current_player") == "iran" else 0.0,

        float(p_isr.get("pp", 0)), float(p_isr.get("ip", 0)), float(p_isr.get("mp", 0)),
        float(p_irn.get("pp", 0)), float(p_irn.get("ip", 0)), float(p_irn.get("mp", 0)),

        float(dom.get("israel", 0)), float(dom.get("iran", 0)),
        float(tp.get("UN", 0)), float(tp.get("US", 0)),
    ]
    return np.array(vec, dtype=np.float32)

def legal_to_mask(legal: List[Dict[str, Any]]) -> np.ndarray:
    A = action_space_size()
    m = np.zeros((A,), dtype=np.float32)
    for a in legal:
        m[action_to_id(a)] = 1.0
    return m
