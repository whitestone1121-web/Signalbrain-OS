#!/usr/bin/env python3
"""
run_canonical.py — SignalBrain-OS Canonical Benchmark
=====================================================

Runs the Apex17 policy compiler against a synthetic market snapshot matrix
and measures:
  - Policy evaluation latency (per-agent, per-symbol)
  - LLM bypass rate (deterministic decisions / total)
  - Throughput (evaluations/sec)
  - VRAM cap adherence (via nvidia-smi)

Usage:
    python benchmarks/run_canonical.py --duration 10 --output results/canonical.json
    python benchmarks/run_canonical.py --preset blackwell_82gb --duration 60

Output:  results/<run_id>/report.json, results/<run_id>/metrics.csv
"""

import argparse
import hashlib
import json
import os
import sys
import time
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Ensure SignalBrain-OS runtime is importable ──
ROOT = Path(__file__).resolve().parent.parent.parent
try:
    from signalbrain.compiler import draft, get_stats
except ImportError:
    # Fallback: resolve from local source tree
    SRC = ROOT / "src"
    sys.path.insert(0, str(SRC))
    from signalbrain.compiler import draft, get_stats


@dataclass
class BenchmarkConfig:
    preset: str = "blackwell_82gb"
    duration_sec: int = 10
    symbols: List[str] = None
    agents: List[str] = None
    seed: int = 42

    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["SPY", "NVDA", "TSLA", "AAPL", "MSFT",
                            "GOOGL", "AMZN", "META", "AMD", "AVGO"]
        if self.agents is None:
            self.agents = ["TechnicalAgent", "SentimentAgent",
                           "FlowAgent", "VolatilityAgent"]


@dataclass
class SyntheticSnapshot:
    """Minimal MarketSnapshot stub for policy evaluation."""
    symbol: str
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


def _generate_snapshots(symbols: List[str], seed: int) -> List[SyntheticSnapshot]:
    """Generate a deterministic set of market snapshots for benchmarking."""
    import random
    rng = random.Random(seed)
    snapshots = []
    for sym in symbols:
        for _ in range(10):  # 10 scenarios per symbol
            snapshots.append(SyntheticSnapshot(
                symbol=sym,
                price=round(rng.uniform(50, 500), 2),
                rsi_14=round(rng.uniform(10, 90), 1),
                rsi_5=round(rng.uniform(10, 90), 1),
                macd_hist=round(rng.gauss(0, 0.5), 4),
                trend_slope=round(rng.gauss(0, 0.3), 4),
                volume_ratio=round(rng.uniform(0.2, 8.0), 2),
                spread_bps=round(rng.uniform(1, 30), 1),
                atr_pct=round(rng.uniform(0.5, 5.0), 2),
                momentum_score=round(rng.gauss(0, 0.4), 4),
                vol_regime_percentile=round(rng.uniform(5, 95), 1),
                price_percentile=round(rng.uniform(5, 95), 1),
            ))
    return snapshots


def _query_gpu() -> Dict[str, Any]:
    """Get GPU metrics via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return {"error": result.stderr.strip()}
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        return {
            "gpu_name": parts[0] if len(parts) > 0 else "Unknown",
            "vram_used_mb": float(parts[1]) if len(parts) > 1 else 0,
            "vram_total_mb": float(parts[2]) if len(parts) > 2 else 0,
            "gpu_util_pct": float(parts[3]) if len(parts) > 3 else 0,
            "temp_c": float(parts[4]) if len(parts) > 4 else 0,
            "power_w": float(parts[5]) if len(parts) > 5 else 0,
        }
    except Exception as e:
        return {"error": str(e)}


def run_benchmark(config: BenchmarkConfig) -> Dict[str, Any]:
    """Execute the canonical benchmark."""
    # draft() and get_stats() imported at module level

    snapshots = _generate_snapshots(config.symbols, config.seed)
    run_id = hashlib.sha256(f"{time.time()}-{config.seed}".encode()).hexdigest()[:12]

    # Pre-benchmark GPU state
    gpu_before = _query_gpu()

    # Run evaluations for the specified duration
    latencies: List[float] = []
    decisions: List[Dict[str, Any]] = []
    eval_count = 0
    bypassed = 0
    start = time.monotonic()

    while (time.monotonic() - start) < config.duration_sec:
        for snap in snapshots:
            for agent in config.agents:
                t0 = time.perf_counter_ns()
                result = draft(snap, agent)
                t1 = time.perf_counter_ns()
                latency_us = (t1 - t0) / 1000.0
                latencies.append(latency_us)
                eval_count += 1

                if result is not None:
                    bypassed += 1
                    decisions.append({
                        "symbol": snap.symbol,
                        "agent": agent,
                        "action": result.signal_action,
                        "confidence": result.confidence,
                        "policy_hash": result.policy_hash,
                        "latency_us": latency_us,
                    })

                if (time.monotonic() - start) >= config.duration_sec:
                    break
            if (time.monotonic() - start) >= config.duration_sec:
                break

    elapsed = time.monotonic() - start

    # Post-benchmark GPU state
    gpu_after = _query_gpu()

    # Compute statistics
    latencies_sorted = sorted(latencies) if latencies else [0]
    n = len(latencies_sorted)
    stats = get_stats()

    report = {
        "run_id": run_id,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": asdict(config),
        "results": {
            "total_evaluations": eval_count,
            "duration_sec": round(elapsed, 3),
            "throughput_eval_per_sec": round(eval_count / elapsed, 1) if elapsed > 0 else 0,
            "bypass_rate": round(bypassed / eval_count, 4) if eval_count > 0 else 0,
            "decisions_produced": len(decisions),
            "latency": {
                "p50_us": round(latencies_sorted[int(n * 0.50)], 2),
                "p95_us": round(latencies_sorted[int(n * 0.95)], 2),
                "p99_us": round(latencies_sorted[int(n * 0.99)], 2),
                "mean_us": round(sum(latencies) / n, 2),
                "min_us": round(latencies_sorted[0], 2),
                "max_us": round(latencies_sorted[-1], 2),
            },
        },
        "apex17_stats": stats,
        "gpu": {
            "before": gpu_before,
            "after": gpu_after,
        },
        "verification": {
            "config_hash": hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest(),
            "decision_digest": hashlib.sha256(
                json.dumps([d["policy_hash"] for d in decisions], sort_keys=True).encode()
            ).hexdigest(),
        },
    }

    return report


def main():
    parser = argparse.ArgumentParser(description="SignalBrain-OS Canonical Benchmark")
    parser.add_argument("--preset", default="blackwell_82gb", help="Hardware preset")
    parser.add_argument("--duration", type=int, default=10, help="Benchmark duration in seconds")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    parser.add_argument("--output", default=None, help="Output file path")
    args = parser.parse_args()

    config = BenchmarkConfig(preset=args.preset, duration_sec=args.duration, seed=args.seed)

    print(f"╔══════════════════════════════════════════════════════╗")
    print(f"║  SignalBrain-OS Canonical Benchmark                 ║")
    print(f"║  Preset: {config.preset:<43s} ║")
    print(f"║  Duration: {config.duration_sec}s | Seed: {config.seed:<27} ║")
    print(f"╚══════════════════════════════════════════════════════╝")
    print()

    report = run_benchmark(config)

    # Write output
    if args.output:
        out_path = Path(args.output)
    else:
        results_dir = Path(__file__).parent.parent / "results" / report["run_id"]
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / "report.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    r = report["results"]
    lat = r["latency"]
    print(f"  Evaluations:      {r['total_evaluations']:,}")
    print(f"  Throughput:       {r['throughput_eval_per_sec']:,.1f} eval/sec")
    print(f"  Bypass Rate:      {r['bypass_rate'] * 100:.1f}%")
    print(f"  Latency p50:      {lat['p50_us']:.1f} µs")
    print(f"  Latency p99:      {lat['p99_us']:.1f} µs")
    print(f"  Decision Digest:  {report['verification']['decision_digest'][:16]}...")
    print()
    print(f"  Report: {out_path}")
    print()

    if report.get("gpu", {}).get("after", {}).get("vram_used_mb"):
        gpu = report["gpu"]["after"]
        print(f"  GPU: {gpu.get('gpu_name', 'N/A')}")
        print(f"  VRAM: {gpu['vram_used_mb'] / 1024:.1f} / {gpu.get('vram_total_mb', 0) / 1024:.1f} GB")


if __name__ == "__main__":
    main()
