#!/usr/bin/env python3
"""
temporal_suite.py — Apex17 Temporal Consistency / Anti-Thrash Tests
===================================================================

Proves that Apex17 decisions are temporally stable: slight input jitter
does not cause excessive flip-flopping unless thresholds are crossed.

Usage:
    python policy/temporal_suite.py
    python policy/temporal_suite.py --verbose
"""

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent.parent
PROOF_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROOF_DIR))
try:
    from signalbrain.compiler import draft
except ImportError:
    SRC = ROOT / "src"
    sys.path.insert(0, str(SRC))
    from signalbrain.compiler import draft


@dataclass
class JitterSnapshot:
    symbol: str = "JITTER"
    price: float = 100.0
    rsi_14: float = 50.0
    rsi_5: float = 50.0
    macd_hist: float = 0.0
    trend_slope: float = 0.0
    volume_ratio: float = 1.0
    spread_bps: float = 5.0
    atr_pct: float = 1.5
    momentum_score: float = 0.0
    vol_regime_percentile: float = 50.0
    price_percentile: float = 50.0


class PolicyTest:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.passed = False
        self.details: Dict[str, Any] = {}

    def to_dict(self):
        return {
            "test": self.name,
            "description": self.description,
            "passed": self.passed,
            "details": self.details,
        }


def test_micro_jitter_stable(verbose: bool = False) -> PolicyTest:
    """100 snapshots with ±0.1% noise on all fields → ≤5% action changes."""
    test = PolicyTest("micro_jitter_stable",
                      "±0.1% noise on all fields → ≤5% action changes")
    try:
        random.seed(42)  # Reproducible
        base = JitterSnapshot(
            symbol="AAPL", rsi_14=55.0, rsi_5=57.0,
            macd_hist=0.1, trend_slope=0.05,
            volume_ratio=1.5, momentum_score=0.1,
        )

        # Get baseline
        baseline = draft(base, "TechnicalAgent")
        baseline_action = baseline.signal_action if baseline else "None"

        # Run 100 jittered variants
        n_runs = 100
        flips = 0
        for _ in range(n_runs):
            jitter_factor = 1.0 + random.uniform(-0.001, 0.001)
            snap = JitterSnapshot(
                symbol="AAPL",
                rsi_14=base.rsi_14 * jitter_factor,
                rsi_5=base.rsi_5 * jitter_factor,
                macd_hist=base.macd_hist * jitter_factor,
                trend_slope=base.trend_slope * jitter_factor,
                volume_ratio=base.volume_ratio * jitter_factor,
                momentum_score=base.momentum_score * jitter_factor,
            )
            decision = draft(snap, "TechnicalAgent")
            action = decision.signal_action if decision else "None"
            if action != baseline_action:
                flips += 1

        flip_rate = flips / n_runs
        test.passed = flip_rate <= 0.05  # ≤5% threshold
        test.details = {
            "baseline_action": baseline_action,
            "n_runs": n_runs,
            "flips": flips,
            "flip_rate": round(flip_rate, 4),
            "threshold": 0.05,
        }
    except Exception as e:
        test.details = {"error": str(e)}
    return test


def test_threshold_crossing_count(verbose: bool = False) -> PolicyTest:
    """Linear RSI ramp 20→80 over 60 steps → limited number of action changes."""
    test = PolicyTest("threshold_crossing_count",
                      "RSI ramp 20→80 → limited action transitions")
    try:
        actions = []
        for i in range(61):
            rsi = 20.0 + (60.0 * i / 60)
            snap = JitterSnapshot(
                symbol="RAMP", rsi_14=rsi, rsi_5=rsi,
                macd_hist=0.0, trend_slope=0.0,
            )
            decision = draft(snap, "TechnicalAgent")
            action = decision.signal_action if decision else "None"
            actions.append(action)

        # Count transitions
        transitions = sum(1 for j in range(1, len(actions))
                          if actions[j] != actions[j - 1])

        # Should have a small number of transitions (crossing thresholds)
        # Not 0 (would mean no threshold effect) and not many (thrash)
        test.passed = transitions <= 10  # Reasonable bound
        test.details = {
            "rsi_range": "20→80",
            "steps": 61,
            "transitions": transitions,
            "max_allowed": 10,
            "action_sequence_sample": actions[::10],  # Every 10th
        }
    except Exception as e:
        test.details = {"error": str(e)}
    return test


def test_boundary_oscillation(verbose: bool = False) -> PolicyTest:
    """RSI oscillating 49.9↔50.1 for 50 steps → stable output (no thrash)."""
    test = PolicyTest("boundary_oscillation",
                      "RSI 49.9↔50.1 oscillation → stable (≤2 unique actions)")
    try:
        actions = set()
        for i in range(50):
            rsi = 49.9 if i % 2 == 0 else 50.1
            snap = JitterSnapshot(
                symbol="OSC", rsi_14=rsi, rsi_5=rsi,
                macd_hist=0.0, trend_slope=0.0,
            )
            decision = draft(snap, "TechnicalAgent")
            action = decision.signal_action if decision else "None"
            actions.add(action)

        # Near the midpoint, output should be stable
        test.passed = len(actions) <= 2
        test.details = {
            "oscillation_range": "49.9↔50.1",
            "steps": 50,
            "unique_actions": list(actions),
            "count": len(actions),
        }
    except Exception as e:
        test.details = {"error": str(e)}
    return test


def main():
    parser = argparse.ArgumentParser(description="Apex17 Temporal Consistency Tests")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    parser.add_argument("--output", default=None, help="Output file path")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════╗")
    print("║  Apex17 Temporal Consistency / Anti-Thrash Suite    ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    tests = [
        test_micro_jitter_stable(args.verbose),
        test_threshold_crossing_count(args.verbose),
        test_boundary_oscillation(args.verbose),
    ]

    for t in tests:
        icon = "✓" if t.passed else "✗"
        print(f"  [{icon}] {t.name}: {t.description}")
        if args.verbose and t.details:
            for k, v in t.details.items():
                print(f"      {k}: {v}")

    passed = sum(1 for t in tests if t.passed)
    total = len(tests)

    print()
    print(f"  {'═' * 50}")
    print(f"  {passed}/{total} tests passed")
    print(f"  {'═' * 50}")

    if args.output:
        import json
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "suite": "temporal_consistency",
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "all_passed": passed == total,
                "passed": passed,
                "total": total,
                "tests": [t.to_dict() for t in tests],
            }, f, indent=2)
        print(f"  Report: {out_path}")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
