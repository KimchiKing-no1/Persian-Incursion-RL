# features.py
from typing import Dict, Any, List
import numpy as np

# --- Feature Engineering Constants ---

# Key targets for the AI to monitor for damage
KEY_TARGETS = [
    "Natanz Uranium Enrichment Facility", "Arak Heavy Water Reactor",
    "Esfahan Conversion Facility", "Fordow Enrichment Site",
    "Kharg Island Oil Terminal", "Abadan Oil Refinery"
]

# All possible upgrades for both sides
ISRAEL_UPGRADES = [
    "Third Arrow-2 battalion", "Iron Dome expanded", "MALD (Miniature Air Launched Decoy)",
    "More aerial tankers", "Jam-resistant GPS receivers (US-built ordnance only)",
    "Jam-resistant GPS receivers (US-built and Israeli-built ordnance)",
    "AIM-120D AMRAAM", "EGBU-28C", "AGM-88 HARM Block 5"
]
IRAN_UPGRADES = [
    "Improved early warning radars", "Improved air defense network",
    "Bodyguard laser decoys/dazzlers (for Nuclear infrastructure)",
    "Bodyguard laser decoys/dazzlers (for Oil infrastructure)",
    "Bodyguard laser decoys/dazzlers (for Military targets)",
    "GPS jammers (for Nuclear infrastructure)",
    "GPS jammers (for Nuclear and Oil infrastructure)",
    "High-fidelity decoys of an entire SAM battery",
    "Pantsyr-S1E [SA-22] mobile gun/SAM system",
    "Additional Tor-M1 [SA-15] batteries", "S-300PMU-1 [SA-20] batteries",
    "Buk-M1 [SA-11] batteries", "HQ-9 batteries",
    "Sejil-2 ballistic missile battalion", "R-27ER1 AAM upgrade for Iranian MiG-29",
    "PL-5E and PL-8 AAMs to replace AIM-9/Fatter from PRC",
    "EM-55 Guided propelled deepwater mines"
]


# --- Granular Action Space ---

# These global variables will be populated by initialize_action_space()
ACTION_MAP = []
ACTION_LOOKUP = {}

def initialize_action_space(rules: Dict[str, Any]):
    """
    Dynamically builds the action space from the game rules.
    This must be called once after the rules are loaded.
    """
    global ACTION_MAP, ACTION_LOOKUP
    if ACTION_MAP:  # Prevent re-initialization
        return

    actions = []

    # 1. Basic actions
    actions.append({"type": "Pass"})
    actions.append({"type": "End Impulse"})

    # 2. Card playing actions for each side
    for side in ["israel", "iran"]:
        card_map = rules.get("cards_structured", {}).get(side, {})
        for card_id in card_map.keys():
            actions.append({"type": "Play Card", "card_id": int(card_id), "_side_": side})

    # 3. Targeted operational actions
    targets = rules.get("VALID_TARGETS", [])
    for target in targets:
        actions.append({"type": "Order Airstrike", "target": target})
        actions.append({"type": "Order Ballistic Missile", "target": target})
        actions.append({"type": "Order Special Warfare", "target": target})

    # Build the final map and lookup dictionary for unique actions
    temp_action_map = []
    seen = set()
    for action in actions:
        key = frozenset(action.items())
        if key not in seen:
            temp_action_map.append(action)
            seen.add(key)

    ACTION_MAP[:] = temp_action_map
    ACTION_LOOKUP.clear()
    for i, action in enumerate(ACTION_MAP):
        key = frozenset(action.items())
        ACTION_LOOKUP[key] = i


def action_space_size() -> int:
    """Returns the total number of unique actions."""
    return len(ACTION_MAP)

def action_to_id(a: Dict[str, Any]) -> int:
    """Converts an action dictionary to its unique integer ID."""
    key = frozenset(a.items())
    return ACTION_LOOKUP.get(key, 0) # Default to 'Pass' if not found

def id_to_action(aid: int) -> Dict[str, Any]:
    """Converts a unique integer ID back to its action dictionary."""
    if 0 <= aid < len(ACTION_MAP):
        return ACTION_MAP[aid]
    return {"type": "Pass"} # Default to 'Pass' on invalid ID

def legal_to_mask(legal_actions: List[Dict[str, Any]], side: str) -> np.ndarray:
    """Converts a list of legal action dictionaries into a binary mask vector."""
    mask = np.zeros(action_space_size(), dtype=np.float32)
    for action in legal_actions:
        if action.get("type") == "Play Card":
            action_with_side = action.copy()
            action_with_side["_side_"] = side
            key = frozenset(action_with_side.items())
            if key in ACTION_LOOKUP:
                 mask[ACTION_LOOKUP[key]] = 1.0
        else:
            key = frozenset(action.items())
            if key in ACTION_LOOKUP:
                mask[ACTION_LOOKUP[key]] = 1.0
    return mask


# --- Enhanced Feature Vector ---

def state_to_features(s: Dict[str, Any], rules: Dict[str, Any]) -> np.ndarray:
    """
    Converts the raw game state dictionary into a detailed, normalized feature vector
    for the neural network. Requires the 'rules' dictionary to accurately calculate damage.
    """
    vec = []

    # 1. Turn Information (Normalized)
    t = s.get("turn", {})
    vec.extend([
        float(t.get("turn_number", 1)) / 42.0,
        1.0 if t.get("phase") == "morning" else 0.0,
        1.0 if t.get("phase") == "afternoon" else 0.0,
        1.0 if t.get("phase") == "night" else 0.0,
        1.0 if t.get("current_player") == "israel" else 0.0,
    ])

    # 2. Resource Information (Normalized 0-20)
    p_isr_res = s.get("players", {}).get("israel", {}).get("resources", {})
    p_irn_res = s.get("players", {}).get("iran", {}).get("resources", {})
    vec.extend([
        min(p_isr_res.get("pp", 0), 20) / 20.0,
        min(p_isr_res.get("ip", 0), 20) / 20.0,
        min(p_isr_res.get("mp", 0), 20) / 20.0,
        min(p_irn_res.get("pp", 0), 20) / 20.0,
        min(p_irn_res.get("ip", 0), 20) / 20.0,
        min(p_irn_res.get("mp", 0), 20) / 20.0,
    ])

    # 3. Opinion Information (Normalized from -10 to +10 -> -1.0 to +1.0)
    op = s.get("opinion", {})
    dom = op.get("domestic", {})
    tp = op.get("third_parties", {})
    vec.extend([
        dom.get("israel", 0) / 10.0,
        dom.get("iran", 0) / 10.0,
        tp.get("UN", 0) / 10.0,
        tp.get("US", 0) / 10.0,
        tp.get("Russia", 0) / 10.0,
        tp.get("China", 0) / 10.0,
        tp.get("Saudi Arabia/GCC", 0) / 10.0,
    ])

    # 4. Key Target Damage Status (0.0 for undamaged, 1.0 for destroyed)
    target_damage_state = s.get("target_damage_status", {})
    for target_name in KEY_TARGETS:
        damage_info_state = target_damage_state.get(target_name, {})
        target_rules = rules.get("TARGET_DEFENSES", {}).get(target_name, {})
        total_hits = 0
        total_max_damage = 0
        
        for group_key in ["Primary_Targets", "Secondary_Targets"]:
            group_rules = target_rules.get(group_key, {})
            for comp_name, comp_rules in group_rules.items():
                comp_damage_state = damage_info_state.get(comp_name, {})
                
                if isinstance(comp_damage_state, dict):
                    total_hits += comp_damage_state.get("damage_boxes_hit", 0)
                elif isinstance(comp_damage_state, int):
                    total_hits += comp_damage_state

                total_max_damage += comp_rules.get("max_damage_for_destroyed", 2)
        
        damage_ratio = total_hits / total_max_damage if total_max_damage > 0 else 0.0
        vec.append(min(damage_ratio, 1.0))

    # 5. Upgrades (1.0 if purchased, 0.0 otherwise)
    isr_upgrades = s.get("upgrades", {}).get("israel", [])
    irn_upgrades = s.get("upgrades", {}).get("iran", [])
    for upgrade in ISRAEL_UPGRADES:
        vec.append(1.0 if upgrade in isr_upgrades else 0.0)
    for upgrade in IRAN_UPGRADES:
        vec.append(1.0 if upgrade in irn_upgrades else 0.0)

    # 6. Card Counts (Normalized by a typical deck size of ~40)
    p_isr = s.get("players", {}).get("israel", {})
    p_irn = s.get("players", {}).get("iran", {})
    vec.extend([
        len(p_isr.get("deck", [])) / 40.0,
        len(p_isr.get("discard", [])) / 40.0,
        len(p_irn.get("deck", [])) / 40.0,
        len(p_irn.get("discard", [])) / 40.0,
    ])

    return np.array(vec, dtype=np.float32)
