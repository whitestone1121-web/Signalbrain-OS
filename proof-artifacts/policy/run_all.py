#!/usr/bin/env python3
"""
run_all.py — Apex17 Expanded Test Harness Orchestrator
=======================================================

Runs all policy test suites and produces a consolidated report.

Suites:
  1. rejection_suite     — Core policy enforcement (5 tests)
  2. adversarial_suite   — Schema robustness (6 tests)
  3. policy_matrix_suite — Golden scenarios (11 tests)
  4. temporal_suite      — Anti-thrash consistency (3 tests)
  5. stress_suite        — Latency & overload (4 tests)
  6. invariance_suite    — Cross-entropy invariance (2 tests)

Usage:
    python policy/run_all.py
    python policy/run_all.py --verbose --output results/expanded_policy_suite.json
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


SUITES = [
    ("rejection_suite", "Core Policy Enforcement"),
    ("adversarial_suite", "Adversarial Schema Robustness"),
    ("policy_matrix_suite", "Full Policy Matrix — Golden Scenarios"),
    ("temporal_suite", "Temporal Consistency / Anti-Thrash"),
    ("stress_suite", "Latency & Overload"),
    ("invariance_suite", "Cross-Entropy Invariance"),
]


def main():
    parser = argparse.ArgumentParser(description="Apex17 Expanded Test Harness")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output per suite")
    parser.add_argument("--output", default=None, help="Consolidated report output path")
    args = parser.parse_args()

    policy_dir = Path(__file__).resolve().parent
    results_dir = policy_dir.parent / "results"
    results_dir.mkdir(exist_ok=True)

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Apex17 Expanded Test Harness — HFT Risk Committee Grade   ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    all_results = []
    total_passed = 0
    total_tests = 0
    all_green = True

    for suite_name, suite_title in SUITES:
        suite_file = policy_dir / f"{suite_name}.py"
        if not suite_file.exists():
            print(f"  [⚠] {suite_name}: FILE NOT FOUND — {suite_file}")
            all_green = False
            continue

        # Run each suite as a subprocess for isolation
        suite_output = results_dir / f"{suite_name}_latest.json"
        cmd = [
            sys.executable, str(suite_file),
            "--output", str(suite_output),
        ]
        if args.verbose:
            cmd.append("--verbose")

        print(f"  ── {suite_title} ──")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )

            # Print suite output
            if result.stdout:
                for line in result.stdout.strip().split("\n"):
                    if line.strip() and not line.startswith("╔") and not line.startswith("╚"):
                        print(f"    {line}")

            if result.stderr and args.verbose:
                for line in result.stderr.strip().split("\n")[:5]:
                    print(f"    [stderr] {line}")

            # Load results
            if suite_output.exists():
                with open(suite_output) as f:
                    suite_data = json.load(f)
                passed = suite_data.get("passed", 0)
                total = suite_data.get("total", 0)
                total_passed += passed
                total_tests += total
                if not suite_data.get("all_passed", False):
                    all_green = False
                all_results.append(suite_data)
            else:
                all_green = False

        except subprocess.TimeoutExpired:
            print(f"    [✗] TIMEOUT (>120s)")
            all_green = False
        except Exception as e:
            print(f"    [✗] ERROR: {e}")
            all_green = False

        print()

    # Summary
    print("═" * 62)
    icon = "✓" if all_green else "✗"
    print(f"  [{icon}] TOTAL: {total_passed}/{total_tests} tests passed across {len(SUITES)} suites")
    print("═" * 62)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "harness": "apex17_expanded",
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "all_passed": all_green,
                "total_passed": total_passed,
                "total_tests": total_tests,
                "suites_run": len(all_results),
                "suites": all_results,
            }, f, indent=2)
        print(f"\n  Consolidated report: {out_path}")

    sys.exit(0 if all_green else 1)


if __name__ == "__main__":
    main()
