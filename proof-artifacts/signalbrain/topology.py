"""
signalbrain.topology — Public interface to the Market Topology Engine.

When running inside the full SignalBrain-OS environment, this module
delegates to the production H₀ Persistent Homology engine. When running
standalone (e.g. from a fresh git clone), it operates in DEMO mode with
a synthetic topology engine that demonstrates the API contract.

Public API:
  - compute_market_topology(values) → MarketTopologyResult
"""
import hashlib
import importlib
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# ── Runtime resolution ──
_RUNTIME_MODULE = None
_DEMO_MODE = False


@dataclass
class MarketTopologyResult:
    """Result of H₀ persistent homology computation."""
    stability: float
    entropy: float
    max_persistence: float
    num_components: int
    num_significant: int
    regime_hash: str
    is_stable: bool
    is_transitioning: bool
    total_persistence: float
    pairs: list


def _resolve_runtime():
    """Attempt to load the production runtime. Fall back to demo mode."""
    global _RUNTIME_MODULE, _DEMO_MODE

    if _RUNTIME_MODULE is not None:
        return True

    for mod_name in ["neural_chat.market_topology"]:
        try:
            _RUNTIME_MODULE = importlib.import_module(mod_name)
            return True
        except ImportError:
            pass

    root = Path(__file__).resolve().parent.parent.parent
    src = root / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
        try:
            _RUNTIME_MODULE = importlib.import_module(
                "neural_chat.market_topology"
            )
            return True
        except ImportError:
            pass

    _DEMO_MODE = True
    return False


_resolve_runtime()

if _DEMO_MODE and not os.environ.get("_SIGNALBRAIN_TOPO_BANNER_SHOWN"):
    os.environ["_SIGNALBRAIN_TOPO_BANNER_SHOWN"] = "1"
    print("  [Topology] DEMO MODE — synthetic H₀ engine")


# ── Demo engine ──

def _demo_compute(values: List[float]) -> MarketTopologyResult:
    """Synthetic topology: deterministic, proves the API contract."""
    n = len(values)
    if n < 2:
        return MarketTopologyResult(
            stability=1.0, entropy=0.0, max_persistence=0.0,
            num_components=1, num_significant=0, regime_hash="0x0",
            is_stable=True, is_transitioning=False,
            total_persistence=0.0, pairs=[],
        )

    # Compute simple persistence-like metrics from value differences
    diffs = [abs(values[i+1] - values[i]) for i in range(n-1)]
    max_diff = max(diffs) if diffs else 1.0
    mean_diff = sum(diffs) / len(diffs) if diffs else 0.0

    # Stability: monotonic = 1.0, noisy = lower
    direction_changes = sum(
        1 for i in range(len(diffs)-1) if diffs[i] > 0 and
        ((values[i+2] - values[i+1]) * (values[i+1] - values[i]) < 0)
    ) if n > 2 else 0
    stability = max(0.0, 1.0 - direction_changes / max(1, n - 2))

    # Entropy from distribution of differences
    if max_diff > 0:
        probs = [d / sum(diffs) for d in diffs if d > 0]
        entropy = -sum(p * math.log(p + 1e-12) for p in probs)
    else:
        entropy = 0.0

    # Components ~ number of significant peaks
    threshold = mean_diff * 1.5
    num_sig = sum(1 for d in diffs if d > threshold)

    # Deterministic hash
    hash_input = f"{n}:{stability:.6f}:{entropy:.6f}:{max_diff:.6f}"
    regime_hash = "0x" + hashlib.sha256(hash_input.encode()).hexdigest()[:16].upper()

    return MarketTopologyResult(
        stability=round(stability, 6),
        entropy=round(entropy, 6),
        max_persistence=round(max_diff, 6),
        num_components=max(1, num_sig),
        num_significant=num_sig,
        regime_hash=regime_hash,
        is_stable=stability > 0.7,
        is_transitioning=stability < 0.3,
        total_persistence=round(sum(diffs), 6),
        pairs=[(diffs[i], diffs[i] + mean_diff) for i in range(min(5, len(diffs)))],
    )


# ── Public API ──

def compute_market_topology(values: List[float]) -> MarketTopologyResult:
    """Compute H₀ persistent homology on a value series.

    Returns a MarketTopologyResult with .stability, .entropy, .regime_hash, etc.
    """
    if _DEMO_MODE:
        return _demo_compute(values)
    return _RUNTIME_MODULE.compute_market_topology(values)
