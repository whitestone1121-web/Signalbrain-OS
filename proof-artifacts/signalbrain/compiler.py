"""
signalbrain.compiler — Public interface to the Apex17 Policy Compiler.

When running inside the full SignalBrain-OS environment, this module
delegates to the production runtime. When running standalone (e.g. from
a fresh git clone), it operates in DEMO mode with a synthetic policy
engine that demonstrates the API contract.

Public API:
  - draft(snapshot, agent_name) → PolicyDecision | None
  - get_stats() → dict
"""
import hashlib
import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ── Runtime resolution ──
_RUNTIME_MODULE = None
_DEMO_MODE = False


@dataclass
class PolicyDecision:
    """Result of a policy draft evaluation."""
    signal_action: str
    confidence: float
    policy_hash: str


def _resolve_runtime():
    """Attempt to load the production runtime. Fall back to demo mode."""
    global _RUNTIME_MODULE, _DEMO_MODE

    if _RUNTIME_MODULE is not None:
        return True

    # Try the installed runtime
    for mod_name in ["neural_chat.apex17_policy_compiler"]:
        try:
            _RUNTIME_MODULE = importlib.import_module(mod_name)
            return True
        except ImportError:
            pass

    # Try resolving from source tree
    root = Path(__file__).resolve().parent.parent.parent
    src = root / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
        try:
            _RUNTIME_MODULE = importlib.import_module(
                "neural_chat.apex17_policy_compiler"
            )
            return True
        except ImportError:
            pass

    # No runtime found — activate demo mode
    _DEMO_MODE = True
    return False


# Resolve on import
_resolve_runtime()

if _DEMO_MODE:
    print("┌─────────────────────────────────────────────────────┐")
    print("│  ⚠  DEMO MODE — SignalBrain-OS runtime not found   │")
    print("│  Results are synthetic. Run inside the full         │")
    print("│  SignalBrain-OS environment for production data.    │")
    print("└─────────────────────────────────────────────────────┘")
    print()


# ── Demo engine (synthetic deterministic policy) ──
_DEMO_STATS = {"evaluations": 0, "bypassed": 0, "demo_mode": True}

_AGENT_SIGNALS = {
    "TechnicalAgent": {"buy_threshold": 30, "sell_threshold": 70, "field": "rsi_14"},
    "SentimentAgent": {"buy_threshold": -0.1, "sell_threshold": 0.1, "field": "macd_hist"},
    "FlowAgent": {"buy_threshold": 2.0, "sell_threshold": 0.5, "field": "volume_ratio"},
    "VolatilityAgent": {"buy_threshold": 20, "sell_threshold": 80, "field": "vol_regime_percentile"},
}


def _demo_draft(snapshot, agent_name: str) -> Optional[PolicyDecision]:
    """Synthetic policy draft — deterministic, no LLM required."""
    _DEMO_STATS["evaluations"] += 1

    config = _AGENT_SIGNALS.get(agent_name)
    if config is None:
        return None

    field = config["field"]
    value = getattr(snapshot, field, 50.0)

    # Deterministic signal logic
    if agent_name == "TechnicalAgent":
        if value < config["buy_threshold"]:
            action, conf = "BUY", 0.45 + (config["buy_threshold"] - value) / 100
        elif value > config["sell_threshold"]:
            action, conf = "SELL", 0.45 + (value - config["sell_threshold"]) / 100
        else:
            return None  # No bypass — neutral zone
    elif agent_name == "SentimentAgent":
        if value > config["sell_threshold"]:
            action, conf = "BUY", min(0.55, 0.40 + abs(value))
        elif value < config["buy_threshold"]:
            action, conf = "SELL", min(0.55, 0.40 + abs(value))
        else:
            return None
    elif agent_name == "FlowAgent":
        if value > config["buy_threshold"]:
            action, conf = "BUY", min(0.52, 0.40 + value / 20)
        elif value < config["sell_threshold"]:
            action, conf = "SELL", min(0.52, 0.40 + (1 - value) / 5)
        else:
            return None
    elif agent_name == "VolatilityAgent":
        if value < config["buy_threshold"]:
            action, conf = "BUY", 0.42
        elif value > config["sell_threshold"]:
            action, conf = "SELL", 0.42
        else:
            return None
    else:
        return None

    _DEMO_STATS["bypassed"] += 1

    # Deterministic hash: same inputs → same hash
    hash_input = f"{snapshot.symbol}:{agent_name}:{action}:{conf:.4f}"
    policy_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    return PolicyDecision(
        signal_action=action,
        confidence=round(conf, 4),
        policy_hash=policy_hash,
    )


# ── Public API ──

def draft(snapshot, agent_name: str) -> Optional[PolicyDecision]:
    """Evaluate a policy draft for a market snapshot and agent.

    Returns a PolicyDecision with .signal_action, .confidence, .policy_hash
    or None if the compiler does not produce a bypass for this input.
    """
    if _DEMO_MODE:
        return _demo_draft(snapshot, agent_name)
    return _RUNTIME_MODULE.draft(snapshot, agent_name)


def get_stats() -> dict:
    """Return cumulative Apex17 compiler statistics."""
    if _DEMO_MODE:
        return dict(_DEMO_STATS)
    return _RUNTIME_MODULE.get_stats()
