#!/usr/bin/env python3
"""
stress_suite.py — Apex17 Latency & Overload Tests
===================================================

Confirms that worst-case latency remains within claimed bounds at
realistic symbol counts and burst loads.

Usage:
    python policy/stress_suite.py
    python policy/stress_suite.py --verbose
"""

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

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
class BenchSnapshot:
    symbol: str = "BENCH"
    price: float = 150.0
    rsi_14: float = 55.0
    rsi_5: float = 58.0
    macd_hist: float = 0.12
    trend_slope: float = 0.08
    volume_ratio: float = 1.8
    spread_bps: float = 4.0
    atr_pct: float = 1.2
    momentum_score: float = 0.15
    vol_regime_percentile: float = 45.0
    price_percentile: float = 55.0


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


AGENTS = ["TechnicalAgent", "SentimentAgent", "FlowAgent", "VolatilityAgent"]


def test_single_symbol_latency(verbose: bool = False) -> PolicyTest:
    """10,000 draft() calls → p99 < 500μs (generous bound for Python)."""
    test = PolicyTest("single_symbol_latency",
                      "10,000 draft() calls → p99 < 500μs")
    try:
        snap = BenchSnapshot(symbol="SPY")
        latencies_us: List[float] = []

        # Warmup
        for _ in range(100):
            draft(snap, "TechnicalAgent")

        # Benchmark
        for _ in range(10_000):
            t0 = time.perf_counter_ns()
            draft(snap, "TechnicalAgent")
            elapsed_us = (time.perf_counter_ns() - t0) / 1000
            latencies_us.append(elapsed_us)

        latencies_us.sort()
        p50 = latencies_us[len(latencies_us) // 2]
        p99 = latencies_us[int(len(latencies_us) * 0.99)]
        p999 = latencies_us[int(len(latencies_us) * 0.999)]
        max_lat = latencies_us[-1]

        test.passed = p99 < 500  # 500μs generous bound for pure Python
        test.details = {
            "calls": 10_000,
            "p50_us": round(p50, 1),
            "p99_us": round(p99, 1),
            "p999_us": round(p999, 1),
            "max_us": round(max_lat, 1),
            "threshold_p99_us": 500,
        }
    except Exception as e:
        test.details = {"error": str(e)}
    return test


def test_batch_50_symbols(verbose: bool = False) -> PolicyTest:
    """50 symbols × 4 agents = 200 calls → total < 50ms."""
    test = PolicyTest("batch_50_symbols",
                      "50 symbols × 4 agents (200 calls) → total < 50ms")
    try:
        symbols = [f"SYM{i:03d}" for i in range(50)]
        snaps = [BenchSnapshot(symbol=s, rsi_14=40 + i * 0.5)
                 for i, s in enumerate(symbols)]

        # Warmup
        for snap in snaps[:5]:
            for agent in AGENTS:
                draft(snap, agent)

        t0 = time.perf_counter()
        for snap in snaps:
            for agent in AGENTS:
                draft(snap, agent)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        test.passed = elapsed_ms < 50
        test.details = {
            "symbols": 50,
            "agents": 4,
            "total_calls": 200,
            "elapsed_ms": round(elapsed_ms, 2),
            "threshold_ms": 50,
            "per_call_us": round(elapsed_ms * 1000 / 200, 1),
        }
    except Exception as e:
        test.details = {"error": str(e)}
    return test


def test_batch_200_symbols(verbose: bool = False) -> PolicyTest:
    """200 symbols × 4 agents = 800 calls → total < 200ms."""
    test = PolicyTest("batch_200_symbols",
                      "200 symbols × 4 agents (800 calls) → total < 200ms")
    try:
        symbols = [f"SYM{i:03d}" for i in range(200)]
        snaps = [BenchSnapshot(symbol=s, rsi_14=20 + i * 0.3)
                 for i, s in enumerate(symbols)]

        t0 = time.perf_counter()
        for snap in snaps:
            for agent in AGENTS:
                draft(snap, agent)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        test.passed = elapsed_ms < 200
        test.details = {
            "symbols": 200,
            "agents": 4,
            "total_calls": 800,
            "elapsed_ms": round(elapsed_ms, 2),
            "threshold_ms": 200,
            "per_call_us": round(elapsed_ms * 1000 / 800, 1),
        }
    except Exception as e:
        test.details = {"error": str(e)}
    return test


def test_burst_determinism(verbose: bool = False) -> PolicyTest:
    """1000 rapid-fire calls in burst → all produce identical hashes."""
    test = PolicyTest("burst_determinism",
                      "1000 burst calls → identical hashes (no degradation)")
    try:
        snap = BenchSnapshot(symbol="BURST", rsi_14=65.0, rsi_5=62.0,
                             macd_hist=0.25, trend_slope=0.12)
        hashes = set()
        actions = set()

        t0 = time.perf_counter()
        for _ in range(1000):
            decision = draft(snap, "TechnicalAgent")
            if decision:
                hashes.add(decision.policy_hash)
                actions.add(decision.signal_action)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        test.passed = len(hashes) <= 1 and len(actions) <= 1
        test.details = {
            "burst_calls": 1000,
            "unique_hashes": len(hashes),
            "unique_actions": list(actions),
            "elapsed_ms": round(elapsed_ms, 2),
        }
    except Exception as e:
        test.details = {"error": str(e)}
    return test


def main():
    parser = argparse.ArgumentParser(description="Apex17 Latency & Overload Tests")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    parser.add_argument("--output", default=None, help="Output file path")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════╗")
    print("║  Apex17 Latency & Overload Stress Suite             ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    tests = [
        test_single_symbol_latency(args.verbose),
        test_batch_50_symbols(args.verbose),
        test_batch_200_symbols(args.verbose),
        test_burst_determinism(args.verbose),
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
                "suite": "latency_stress",
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
