"""
signalbrain.regime_memory — Public interface to the Regime Memory system.

When running inside the full SignalBrain-OS environment, this module
delegates to the production RegimeMemory. When standalone, it provides
a synthetic demo that proves the API contract: 20-dim fingerprints,
O(1) hash recall, ring-buffer capacity enforcement.

Public API:
  - RegimeFingerprint(symbol, spectral_regime, ...) → fingerprint
  - RegimeMemory(max_per_symbol, max_total) → memory instance
"""
import hashlib
import importlib
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Runtime resolution ──
_RUNTIME_MODULE = None
_DEMO_MODE = False


def _resolve_runtime():
    global _RUNTIME_MODULE, _DEMO_MODE
    if _RUNTIME_MODULE is not None:
        return True
    for mod_name in ["neural_chat.regime_memory"]:
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
            _RUNTIME_MODULE = importlib.import_module("neural_chat.regime_memory")
            return True
        except ImportError:
            pass
    _DEMO_MODE = True
    return False


_resolve_runtime()

if _DEMO_MODE and not os.environ.get("_SIGNALBRAIN_REGIME_BANNER_SHOWN"):
    os.environ["_SIGNALBRAIN_REGIME_BANNER_SHOWN"] = "1"
    print("  [RegimeMemory] DEMO MODE — synthetic fingerprints")


# ── Export production or demo classes ──

if not _DEMO_MODE and _RUNTIME_MODULE is not None:
    RegimeFingerprint = _RUNTIME_MODULE.RegimeFingerprint
    RegimeMemory = _RUNTIME_MODULE.RegimeMemory
else:
    # Demo implementations

    _REGIME_MAP = {
        "Uptrend": 0, "Downtrend": 1, "Choppy": 2, "Stable": 3,
        "Volatile": 4, "MeanReverting": 5, "Trending": 6,
    }

    @dataclass
    class RegimeFingerprint:
        """20-dim market regime fingerprint (demo)."""
        symbol: str
        spectral_regime: str
        spectral_energy: float
        volatility_percentile: float
        implied_vol: float
        realized_vol: float
        bb_width: float
        var_95: float
        kelly_fraction: float
        trend_slope: float
        momentum_score: float
        rsi_14: float
        volume_ratio: float
        put_call_ratio: float
        skew: float
        spread_zscore: float
        correlation_spy: float
        persistence_stability: float = 0.0
        persistence_entropy: float = 0.0
        max_persistence: float = 0.0
        topological_hash: str = ""
        timestamp: float = 0.0
        outcome_direction: str = ""
        outcome_confidence: float = 0.0
        outcome_pnl: float = 0.0

        def to_vector(self) -> np.ndarray:
            """Convert to 20-dim numeric vector."""
            regime_code = _REGIME_MAP.get(self.spectral_regime, 7) / 7.0
            return np.array([
                regime_code, self.spectral_energy,
                self.volatility_percentile / 100.0, self.implied_vol,
                self.realized_vol, self.bb_width,
                self.var_95, self.kelly_fraction,
                self.trend_slope, self.momentum_score,
                self.rsi_14 / 100.0, self.volume_ratio / 10.0,
                self.put_call_ratio, self.skew,
                self.spread_zscore, self.correlation_spy,
                self.persistence_stability, self.persistence_entropy / 10.0,
                self.max_persistence / 10.0,
                int(self.topological_hash[:10] or "0", 16) / (16**10)
                if self.topological_hash else 0.0,
            ], dtype=np.float32)

        def fingerprint_id(self) -> str:
            raw = f"{self.symbol}:{self.spectral_regime}:{self.timestamp}"
            return hashlib.sha256(raw.encode()).hexdigest()[:16]

        def summary(self) -> str:
            return (
                f"{self.spectral_regime} | "
                f"vol={self.volatility_percentile:.0f}p "
                f"IV={self.implied_vol:.0f} "
                f"rsi={self.rsi_14:.0f} "
                f"VaR={self.var_95:.4f} "
                f"Kelly={self.kelly_fraction:.2f} "
                f"topo={self.persistence_stability:.2f}"
            )

    class RegimeMemory:
        """Ring-buffer regime memory with O(1) hash recall (demo)."""

        def __init__(self, max_per_symbol: int = 500, max_total: int = 25000,
                     recall_top_k: int = 5, similarity_threshold: float = 0.75,
                     decay_halflife_s: float = 3600.0):
            self.max_per_symbol = max_per_symbol
            self.max_total = max_total
            self.recall_top_k = recall_top_k
            self.similarity_threshold = similarity_threshold
            self.decay_halflife_s = decay_halflife_s
            self._memory: Dict[str, List[RegimeFingerprint]] = {}
            self._global: List[RegimeFingerprint] = []
            self._topo_index: Dict[str, List[RegimeFingerprint]] = {}
            self._store_count = 0
            self._recall_count = 0

        def store(self, fp: RegimeFingerprint):
            self._store_count += 1
            # Per-symbol ring buffer
            if fp.symbol not in self._memory:
                self._memory[fp.symbol] = []
            buf = self._memory[fp.symbol]
            buf.append(fp)
            if len(buf) > self.max_per_symbol:
                evicted = buf.pop(0)
                self._global.remove(evicted)
            # Global ring buffer
            self._global.append(fp)
            if len(self._global) > self.max_total:
                self._global.pop(0)
            # Topo hash index
            if fp.topological_hash:
                if fp.topological_hash not in self._topo_index:
                    self._topo_index[fp.topological_hash] = []
                self._topo_index[fp.topological_hash].append(fp)

        def recall_by_hash(self, topo_hash: str) -> List[RegimeFingerprint]:
            return self._topo_index.get(topo_hash, [])

        def get_risk_multiplier(self, symbol: Optional[str] = None,
                                regime_context: str = "",
                                snapshot: Any = None,
                                risk_assessment: Any = None) -> float:
            if symbol and symbol in self._memory:
                fps = self._memory[symbol]
                if fps:
                    avg_pnl = sum(fp.outcome_pnl for fp in fps) / len(fps)
                    return max(0.7, min(1.3, 1.0 + avg_pnl * 10))
            return 1.0

        def get_bias_adjustment(self, symbol: Optional[str] = None, **kw):
            return None

        def get_stats(self) -> Dict[str, Any]:
            return {
                "total_fingerprints": len(self._global),
                "symbols_tracked": len(self._memory),
                "market_regimes": 0,
                "topo_hashes": len(self._topo_index),
                "store_count": self._store_count,
                "recall_count": self._recall_count,
                "max_per_symbol": self.max_per_symbol,
                "max_total": self.max_total,
                "vector_dims": 20,
                "decay_halflife_s": self.decay_halflife_s,
            }
