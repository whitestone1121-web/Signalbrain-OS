#!/usr/bin/env python3
"""
reproduce_published.py — Reproduce Website/Deck Benchmark Claims
=================================================================

Thin wrapper around run_canonical.py that uses the exact parameters
matching the claims published on signalbrain.ai.

Usage:
    python benchmarks/reproduce_published.py
    python benchmarks/reproduce_published.py --suite website_v1
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


SUITES = {
    "website_v1": {
        "description": "Reproduce signalbrain.ai published performance claims",
        "args": ["--preset", "blackwell_82gb", "--duration", "60", "--seed", "42"],
        "claims": {
            "bypass_rate_min": 0.40,
            "latency_p99_max_us": 500.0,
            "throughput_min_eval_per_sec": 1000,
        },
    },
    "quick_smoke": {
        "description": "10-second smoke test for CI",
        "args": ["--preset", "blackwell_82gb", "--duration", "10", "--seed", "42"],
        "claims": {},
    },
}


def main():
    parser = argparse.ArgumentParser(description="Reproduce Published Benchmarks")
    parser.add_argument("--suite", default="website_v1", choices=SUITES.keys())
    args = parser.parse_args()

    suite = SUITES[args.suite]
    print(f"╔══════════════════════════════════════════════════════╗")
    print(f"║  Reproduce Published Claims: {args.suite:<23s} ║")
    print(f"║  {suite['description']:<52s} ║")
    print(f"╚══════════════════════════════════════════════════════╝")
    print()

    # Run the canonical benchmark
    canonical = Path(__file__).parent / "run_canonical.py"
    output = Path(__file__).parent.parent / "results" / f"reproduce_{args.suite}.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(canonical)] + suite["args"] + ["--output", str(output)]
    print(f"  Running: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n  ✗ Benchmark failed with exit code {result.returncode}")
        sys.exit(1)

    # Validate claims
    if not suite["claims"]:
        print(f"\n  ✓ Smoke test completed (no claims to validate)")
        sys.exit(0)

    with open(output) as f:
        report = json.load(f)

    r = report["results"]
    claims = suite["claims"]
    failures = []

    print(f"\n  Validating published claims:")
    print(f"  {'─' * 50}")

    if "bypass_rate_min" in claims:
        actual = r["bypass_rate"]
        expected = claims["bypass_rate_min"]
        ok = actual >= expected
        icon = "✓" if ok else "✗"
        print(f"  [{icon}] Bypass rate: {actual * 100:.1f}% (min: {expected * 100:.1f}%)")
        if not ok:
            failures.append(f"bypass_rate {actual} < {expected}")

    if "latency_p99_max_us" in claims:
        actual = r["latency"]["p99_us"]
        expected = claims["latency_p99_max_us"]
        ok = actual <= expected
        icon = "✓" if ok else "✗"
        print(f"  [{icon}] Latency p99: {actual:.1f} µs (max: {expected:.1f} µs)")
        if not ok:
            failures.append(f"latency_p99 {actual} > {expected}")

    if "throughput_min_eval_per_sec" in claims:
        actual = r["throughput_eval_per_sec"]
        expected = claims["throughput_min_eval_per_sec"]
        ok = actual >= expected
        icon = "✓" if ok else "✗"
        print(f"  [{icon}] Throughput: {actual:,.1f} eval/sec (min: {expected:,})")
        if not ok:
            failures.append(f"throughput {actual} < {expected}")

    print(f"  {'─' * 50}")
    if failures:
        print(f"  ✗ {len(failures)} claim(s) not met: {', '.join(failures)}")
        sys.exit(1)
    else:
        print(f"  ✓ All published claims validated")
        sys.exit(0)


if __name__ == "__main__":
    main()
