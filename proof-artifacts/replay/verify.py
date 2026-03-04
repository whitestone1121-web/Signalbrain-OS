#!/usr/bin/env python3
"""
verify.py — SignalBrain-OS Deterministic Replay Verification
=============================================================

Verifies that:
  1. USI archive integrity is maintained (HMAC verification)
  2. Merkle anchor roots match their leaf hashes
  3. Apex17 policy decisions are deterministic (same inputs → same outputs)

Usage:
    python replay/verify.py
    python replay/verify.py --run results/canonical.json --assert deterministic_actions
    python replay/verify.py --usi-db /app/data/usi_audit.db --depth 100
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent.parent
PROOF_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROOF_DIR))
try:
    from signalbrain.audit import USIPersistenceService
    from signalbrain.anchor import build_merkle_root
except ImportError:
    SRC = ROOT / "src"
    sys.path.insert(0, str(SRC))
    from signalbrain.audit import USIPersistenceService
    from signalbrain.anchor import build_merkle_root


def verify_usi_archive(db_path: str, depth: int = 100) -> Dict[str, Any]:
    """Verify HMAC integrity of USI archive records."""
    try:
        # USIPersistenceService imported at module level
        svc = USIPersistenceService(db_dir=Path(db_path).parent, db_name=Path(db_path).name)
        report = svc.verify_archive(n=depth)
        svc.shutdown()
        return {
            "check": "usi_hmac_integrity",
            "passed": report.get("tampered", 0) == 0,
            "records_checked": report.get("checked", 0),
            "records_valid": report.get("valid", 0),
            "records_tampered": report.get("tampered", 0),
            "tampered_hashes": report.get("tampered_hashes", []),
        }
    except Exception as e:
        return {"check": "usi_hmac_integrity", "passed": False, "error": str(e)}


def verify_merkle_anchors(data_dir: str) -> Dict[str, Any]:
    """Verify Merkle anchor receipts match their leaf hashes."""
    try:
        # build_merkle_root() imported at module level
        receipt_dir = Path(data_dir) / "anchor_receipts"
        if not receipt_dir.exists():
            return {"check": "merkle_anchors", "passed": True, "receipts_found": 0,
                    "note": "No anchor receipts directory found"}

        receipts = sorted(receipt_dir.glob("anchor_*.json"))
        if not receipts:
            ledger = receipt_dir / "usi_ledger.jsonl"
            if ledger.exists():
                return {"check": "merkle_anchors", "passed": True,
                        "receipts_found": 0, "ledger_exists": True,
                        "note": "Ledger present but no anchor receipt JSONs yet"}
            return {"check": "merkle_anchors", "passed": True, "receipts_found": 0}

        verified = 0
        failed = 0
        errors = []

        for rpath in receipts[-10:]:  # Check last 10 anchors
            try:
                with open(rpath) as f:
                    receipt = json.load(f)

                leaves = receipt.get("leaf_hashes", [])
                stored_root = receipt.get("merkle_root", "")

                if not leaves or not stored_root:
                    continue

                computed_root = build_merkle_root(leaves)
                if computed_root == stored_root:
                    verified += 1
                else:
                    failed += 1
                    errors.append({
                        "file": rpath.name,
                        "expected": stored_root[:16],
                        "computed": computed_root[:16],
                    })
            except Exception as e:
                errors.append({"file": rpath.name, "error": str(e)})

        return {
            "check": "merkle_anchors",
            "passed": failed == 0,
            "receipts_checked": verified + failed,
            "receipts_verified": verified,
            "receipts_failed": failed,
            "errors": errors if errors else None,
        }
    except Exception as e:
        return {"check": "merkle_anchors", "passed": False, "error": str(e)}


def verify_deterministic_replay(run_file: str) -> Dict[str, Any]:
    """Verify that Apex17 policy decisions are deterministic.

    Re-runs the same benchmark config and compares decision digests.
    """
    try:
        with open(run_file) as f:
            original = json.load(f)

        original_digest = original.get("verification", {}).get("decision_digest", "")
        original_config_hash = original.get("verification", {}).get("config_hash", "")
        config = original.get("config", {})

        if not original_digest:
            return {"check": "deterministic_replay", "passed": False,
                    "error": "Original run has no decision_digest"}

        # Re-run with same config
        sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))
        from run_canonical import BenchmarkConfig, run_benchmark

        replay_config = BenchmarkConfig(
            preset=config.get("preset", "blackwell_82gb"),
            duration_sec=min(config.get("duration_sec", 10), 10),  # Cap at 10s for replays
            seed=config.get("seed", 42),
            symbols=config.get("symbols"),
            agents=config.get("agents"),
        )

        replay_report = run_benchmark(replay_config)
        replay_digest = replay_report["verification"]["decision_digest"]
        replay_config_hash = replay_report["verification"]["config_hash"]

        return {
            "check": "deterministic_replay",
            "passed": original_digest == replay_digest,
            "original_digest": original_digest,
            "replay_digest": replay_digest,
            "config_hash_match": original_config_hash == replay_config_hash,
            "actions_deterministic": original_digest == replay_digest,
        }
    except Exception as e:
        return {"check": "deterministic_replay", "passed": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="SignalBrain-OS Replay Verification")
    parser.add_argument("--run", default=None, help="Path to a previous benchmark run JSON")
    parser.add_argument("--usi-db", default=None, help="Path to USI audit database")
    parser.add_argument("--data-dir", default=None, help="Path to data directory (for Merkle anchors)")
    parser.add_argument("--depth", type=int, default=100, help="Number of USI records to verify")
    parser.add_argument("--assert", dest="assertions", nargs="*", default=[],
                        help="Assertions: deterministic_actions, merkle_anchor, usi_integrity")
    parser.add_argument("--output", default=None, help="Output file path")
    args = parser.parse_args()

    # Resolve defaults
    data_dir = args.data_dir or str(ROOT / "data")
    usi_db = args.usi_db or os.path.join(data_dir, "usi_audit.db")

    print("╔══════════════════════════════════════════════════════╗")
    print("║  SignalBrain-OS Deterministic Replay Verification   ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    checks: List[Dict[str, Any]] = []

    # 1. USI Archive integrity
    print("  [1/3] Verifying USI archive HMAC integrity...")
    usi_result = verify_usi_archive(usi_db, args.depth)
    checks.append(usi_result)
    status = "✓ PASS" if usi_result["passed"] else "✗ FAIL"
    print(f"        {status} — {usi_result.get('records_checked', 0)} records checked")

    # 2. Merkle anchors
    print("  [2/3] Verifying Merkle anchor roots...")
    merkle_result = verify_merkle_anchors(data_dir)
    checks.append(merkle_result)
    status = "✓ PASS" if merkle_result["passed"] else "✗ FAIL"
    print(f"        {status} — {merkle_result.get('receipts_checked', 0)} anchors verified")

    # 3. Deterministic replay (only if --run provided)
    if args.run:
        print("  [3/3] Verifying deterministic replay...")
        replay_result = verify_deterministic_replay(args.run)
        checks.append(replay_result)
        status = "✓ PASS" if replay_result["passed"] else "✗ FAIL"
        print(f"        {status} — digest match: {replay_result.get('actions_deterministic', False)}")
    else:
        print("  [3/3] Skipped (no --run file provided)")

    # Summary
    all_passed = all(c["passed"] for c in checks)
    report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "all_checks_passed": all_passed,
        "checks": checks,
    }

    print()
    print(f"  {'═' * 50}")
    print(f"  Result: {'ALL CHECKS PASSED ✓' if all_passed else 'SOME CHECKS FAILED ✗'}")
    print(f"  {'═' * 50}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Report: {out_path}")

    # Assert mode for CI
    if args.assertions:
        for assertion in args.assertions:
            matching = [c for c in checks if assertion in c.get("check", "")]
            if not matching or not all(c["passed"] for c in matching):
                print(f"\n  ASSERTION FAILED: {assertion}")
                sys.exit(1)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
