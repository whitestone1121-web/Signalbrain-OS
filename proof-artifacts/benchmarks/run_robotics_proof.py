#!/usr/bin/env python3
"""
run_robotics_proof.py — Apex17 Robotics Engine Proof Suite
===========================================================

Orchestrates C++ engine tests, Python topology/memory tests, and produces
a structured JSON report with per-test timing, pass/fail, and aggregates.

Usage:
    python proof-artifacts/benchmarks/run_robotics_proof.py
    python proof-artifacts/benchmarks/run_robotics_proof.py --skip-cuda --output results/proof.json

Exit code: 0 if all suites pass, 1 otherwise.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
SRC  = ROOT / "src"
SIGNALBRAIN = Path(__file__).resolve().parent.parent  # proof-artifacts/
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SIGNALBRAIN))

# ══════════════════════════════════════════════════════════════════
# Data Structures
# ══════════════════════════════════════════════════════════════════

@dataclass
class TestResult:
    name: str
    suite: str
    passed: bool
    elapsed_ms: float = 0.0
    detail: str = ""
    skipped: bool = False


@dataclass
class SuiteResult:
    name: str
    tests: List[TestResult] = field(default_factory=list)
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    elapsed_ms: float = 0.0

    def ok(self) -> bool:
        return self.failed == 0


@dataclass
class ProofReport:
    run_id: str
    timestamp_utc: str
    suites: List[dict] = field(default_factory=list)
    total_tests: int = 0
    total_passed: int = 0
    total_failed: int = 0
    total_skipped: int = 0
    elapsed_sec: float = 0.0
    all_passed: bool = True
    digest: str = ""

    def stamp(self, suites: List[SuiteResult]):
        for s in suites:
            self.total_tests += len(s.tests)
            self.total_passed += s.passed
            self.total_failed += s.failed
            self.total_skipped += s.skipped
            self.suites.append(asdict(s))
        self.all_passed = self.total_failed == 0
        self.digest = hashlib.sha256(
            json.dumps(self.suites, sort_keys=True).encode()
        ).hexdigest()[:16]


# ══════════════════════════════════════════════════════════════════
# Suite 1: C++ Engine Tests (spatial_prior + spatial_council)
# ══════════════════════════════════════════════════════════════════

def _run_cpp_binary(binary_path: str, suite_name: str) -> SuiteResult:
    """Run a compiled C++ test binary and parse the PASS/SKIP/FAIL output."""
    suite = SuiteResult(name=suite_name)
    t0 = time.perf_counter()

    try:
        proc = subprocess.run(
            [binary_path],
            capture_output=True, text=True, timeout=120
        )
    except FileNotFoundError:
        suite.tests.append(TestResult(
            name="binary_exists", suite=suite_name,
            passed=False, detail=f"Binary not found: {binary_path}"
        ))
        suite.failed = 1
        return suite
    except subprocess.TimeoutExpired:
        suite.tests.append(TestResult(
            name="timeout", suite=suite_name,
            passed=False, detail="Test binary timed out after 120s"
        ))
        suite.failed = 1
        return suite

    suite.elapsed_ms = (time.perf_counter() - t0) * 1000

    # Parse output: each line like "  test_name                  [PASS]" or "[SKIP ...]"
    for line in proc.stdout.splitlines():
        line = line.strip()
        match = re.match(r'^(.+?)\s+\[(PASS|SKIP[^\]]*)\]', line)
        if match:
            name = match.group(1).strip()
            status = match.group(2)
            is_skip = status.startswith("SKIP")
            suite.tests.append(TestResult(
                name=name, suite=suite_name,
                passed=not is_skip, skipped=is_skip,
                detail=status if is_skip else ""
            ))

    suite.passed = sum(1 for t in suite.tests if t.passed and not t.skipped)
    suite.failed = 0 if proc.returncode == 0 else max(1, len(suite.tests) - suite.passed)
    suite.skipped = sum(1 for t in suite.tests if t.skipped)

    # If binary failed but we couldn't parse individual failures
    if proc.returncode != 0 and suite.failed == 0:
        suite.failed = 1
        suite.tests.append(TestResult(
            name="exit_code", suite=suite_name,
            passed=False, detail=f"Exit code {proc.returncode}: {proc.stderr[:200]}"
        ))

    return suite


def run_cpp_suites(build_dir: Path) -> List[SuiteResult]:
    """Run all C++ test binaries from the build directory."""
    results = []
    binaries = [
        ("spatial_prior_tests",  "C++ SpatialPrior"),
        ("spatial_council_tests", "C++ SpatialCouncil"),
        ("apex17_smoke_test",    "C++ Smoke"),
    ]
    for binary, name in binaries:
        path = build_dir / binary
        if path.exists():
            results.append(_run_cpp_binary(str(path), name))
        else:
            s = SuiteResult(name=name)
            s.tests.append(TestResult(
                name="binary_exists", suite=name,
                passed=False, detail=f"Not found: {path}"
            ))
            s.failed = 1
            results.append(s)
    return results


# ══════════════════════════════════════════════════════════════════
# Suite 2: Python Market Topology Engine
# ══════════════════════════════════════════════════════════════════

def run_market_topology_suite() -> SuiteResult:
    """Test the Python market topology engine (H₀ PH, fingerprinting)."""
    suite = SuiteResult(name="Market Topology Engine")
    t0 = time.perf_counter()

    try:
        from signalbrain.topology import compute_market_topology
    except ImportError as e:
        suite.tests.append(TestResult(
            name="import", suite=suite.name,
            passed=False, detail=str(e)
        ))
        suite.failed = 1
        return suite

    import numpy as np

    # Test 1: Linear trend → stability = 1.0
    def test_linear_stability():
        values = list(range(50))
        result = compute_market_topology(values)
        tr = TestResult(name="linear_trend_stability", suite=suite.name, passed=False)
        if result.stability >= 0.99:
            tr.passed = True
            tr.detail = f"stability={result.stability:.3f}"
        else:
            tr.detail = f"Expected ≥0.99, got {result.stability:.3f}"
        return tr

    # Test 2: Random walk → low stability
    def test_random_stability():
        np.random.seed(42)
        values = np.random.randn(100).cumsum().tolist()
        result = compute_market_topology(values)
        tr = TestResult(name="random_walk_low_stability", suite=suite.name, passed=False)
        if result.stability < 0.85:
            tr.passed = True
            tr.detail = f"stability={result.stability:.3f}"
        else:
            tr.detail = f"Expected <0.85, got {result.stability:.3f}"
        return tr

    # Test 3: Fingerprint hash is deterministic
    def test_deterministic_hash():
        values = [1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 8.0] * 5
        r1 = compute_market_topology(values)
        r2 = compute_market_topology(values)
        tr = TestResult(name="deterministic_hash", suite=suite.name, passed=False)
        if r1.regime_hash == r2.regime_hash:
            tr.passed = True
            tr.detail = f"hash={r1.regime_hash[:16]}..."
        else:
            tr.detail = f"{r1.regime_hash} != {r2.regime_hash}"
        return tr

    # Test 4: Latency under 5ms
    def test_latency():
        values = list(range(200))
        t_start = time.perf_counter()
        for _ in range(100):
            compute_market_topology(values)
        avg_ms = (time.perf_counter() - t_start) / 100 * 1000
        tr = TestResult(name="latency_under_5ms", suite=suite.name, passed=False)
        if avg_ms < 5.0:
            tr.passed = True
        tr.detail = f"{avg_ms:.3f}ms avg"
        return tr

    # Test 5: Entropy for complex signal
    def test_entropy():
        np.random.seed(99)
        values = (np.random.randn(100) * 10).tolist()
        result = compute_market_topology(values)
        tr = TestResult(name="complex_signal_entropy", suite=suite.name, passed=False)
        if result.entropy > 1.0:
            tr.passed = True
        tr.detail = f"entropy={result.entropy:.3f}"
        return tr

    # Test 6: Output fields complete
    def test_output_fields():
        values = list(range(30))
        result = compute_market_topology(values)
        required = ["stability", "entropy",
                     "max_persistence", "num_components", "regime_hash"]
        missing = [k for k in required if not hasattr(result, k)]
        tr = TestResult(name="output_fields_complete", suite=suite.name, passed=False)
        if not missing:
            tr.passed = True
            tr.detail = f"{len(required)} fields present"
        else:
            tr.detail = f"Missing: {missing}"
        return tr

    tests = [test_linear_stability, test_random_stability,
             test_deterministic_hash, test_latency,
             test_entropy, test_output_fields]

    for test_fn in tests:
        try:
            tr = test_fn()
        except Exception as e:
            tr = TestResult(name=test_fn.__name__, suite=suite.name,
                            passed=False, detail=str(e)[:200])
        suite.tests.append(tr)

    suite.elapsed_ms = (time.perf_counter() - t0) * 1000
    suite.passed = sum(1 for t in suite.tests if t.passed)
    suite.failed = sum(1 for t in suite.tests if not t.passed)
    return suite


# ══════════════════════════════════════════════════════════════════
# Suite 3: Regime Memory
# ══════════════════════════════════════════════════════════════════

def run_regime_memory_suite() -> SuiteResult:
    """Test the RegimeMemory / O(1) hash recall system."""
    suite = SuiteResult(name="Regime Memory")
    t0 = time.perf_counter()

    try:
        from signalbrain.regime_memory import RegimeFingerprint, RegimeMemory
    except ImportError as e:
        suite.tests.append(TestResult(
            name="import", suite=suite.name,
            passed=False, detail=str(e)
        ))
        suite.failed = 1
        return suite

    def _make_fp(symbol="NVDA", regime="Uptrend", stab=0.87, entropy=2.1,
                 max_p=3.5, thash="0xTEST"):
        """Build a RegimeFingerprint with all required fields."""
        return RegimeFingerprint(
            symbol=symbol, spectral_regime=regime, spectral_energy=0.8,
            volatility_percentile=50.0, implied_vol=0.3, realized_vol=0.25,
            bb_width=0.05, var_95=0.02, kelly_fraction=0.1,
            trend_slope=0.01, momentum_score=0.5, rsi_14=55.0,
            volume_ratio=1.2, put_call_ratio=0.8, skew=-0.1,
            spread_zscore=0.5, correlation_spy=0.85,
            persistence_stability=stab, persistence_entropy=entropy,
            max_persistence=max_p, topological_hash=thash,
            timestamp=time.time(), outcome_direction="LONG", outcome_pnl=0.02,
        )

    # Test 1: 20-dim vector
    def test_vector_dims():
        fp = _make_fp()
        vec = fp.to_vector()
        tr = TestResult(name="fingerprint_20d_vector", suite=suite.name, passed=False)
        if len(vec) == 20:
            tr.passed = True
            tr.detail = f"shape={vec.shape}"
        else:
            tr.detail = f"Expected 20 dims, got {len(vec)}"
        return tr

    # Test 2: Store and recall by hash
    def test_hash_recall():
        mem = RegimeMemory(max_per_symbol=100, max_total=1000)
        fp = _make_fp(symbol="AAPL", thash="0xDEADBEEF")
        mem.store(fp)
        matches = mem.recall_by_hash("0xDEADBEEF")
        tr = TestResult(name="o1_hash_recall", suite=suite.name, passed=False)
        if len(matches) == 1 and matches[0].symbol == "AAPL":
            tr.passed = True
        else:
            tr.detail = f"Expected 1 match, got {len(matches)}"
        return tr

    # Test 3: Unknown hash → empty
    def test_unknown_hash():
        mem = RegimeMemory(max_per_symbol=100, max_total=1000)
        matches = mem.recall_by_hash("0xNOTHING")
        tr = TestResult(name="unknown_hash_empty", suite=suite.name, passed=False)
        if len(matches) == 0:
            tr.passed = True
        else:
            tr.detail = f"Expected 0, got {len(matches)}"
        return tr

    # Test 4: Stats
    def test_stats():
        mem = RegimeMemory(max_per_symbol=100, max_total=1000)
        fp = _make_fp(symbol="TSLA", regime="Choppy", stab=0.2, entropy=4.5, thash="0xCHOP")
        mem.store(fp)
        stats = mem.get_stats()
        tr = TestResult(name="memory_stats", suite=suite.name, passed=False)
        if stats["vector_dims"] == 20 and stats["topo_hashes"] >= 1:
            tr.passed = True
            tr.detail = f"dims={stats['vector_dims']} hashes={stats['topo_hashes']}"
        else:
            tr.detail = str(stats)
        return tr

    # Test 5: Summary contains topo info
    def test_summary():
        fp = _make_fp(symbol="SPY", regime="Stable", stab=0.9, entropy=0.5, thash="0xSPY123")
        summary = fp.summary()
        tr = TestResult(name="summary_has_topo", suite=suite.name, passed=False)
        if "topo=" in summary:
            tr.passed = True
            tr.detail = summary[:60]
        else:
            tr.detail = f"Missing 'topo=' in: {summary[:60]}"
        return tr

    tests = [test_vector_dims, test_hash_recall, test_unknown_hash,
             test_stats, test_summary]

    for test_fn in tests:
        try:
            tr = test_fn()
        except Exception as e:
            tr = TestResult(name=test_fn.__name__, suite=suite.name,
                            passed=False, detail=str(e)[:200])
        suite.tests.append(tr)

    suite.elapsed_ms = (time.perf_counter() - t0) * 1000
    suite.passed = sum(1 for t in suite.tests if t.passed)
    suite.failed = sum(1 for t in suite.tests if not t.passed)
    return suite


# ══════════════════════════════════════════════════════════════════
# Suite 4: Topological Guard (Director Integration)
# ══════════════════════════════════════════════════════════════════

def run_topological_guard_suite() -> SuiteResult:
    """Test the topological guard / director integration path."""
    suite = SuiteResult(name="Topological Guard")
    t0 = time.perf_counter()

    try:
        from signalbrain.regime_memory import RegimeFingerprint, RegimeMemory
        from signalbrain.topology import compute_market_topology
    except ImportError as e:
        suite.tests.append(TestResult(
            name="import", suite=suite.name,
            passed=False, detail=str(e)
        ))
        suite.failed = 1
        return suite

    def _make_fp(symbol="NVDA", regime="Uptrend", stab=0.87, entropy=2.1,
                 max_p=3.5, thash="0xGUARD"):
        return RegimeFingerprint(
            symbol=symbol, spectral_regime=regime, spectral_energy=0.8,
            volatility_percentile=50.0, implied_vol=0.3, realized_vol=0.25,
            bb_width=0.05, var_95=0.02, kelly_fraction=0.1,
            trend_slope=0.01, momentum_score=0.5, rsi_14=55.0,
            volume_ratio=1.2, put_call_ratio=0.8, skew=-0.1,
            spread_zscore=0.5, correlation_spy=0.85,
            persistence_stability=stab, persistence_entropy=entropy,
            max_persistence=max_p, topological_hash=thash,
            timestamp=time.time(), outcome_direction="LONG", outcome_pnl=0.02,
        )

    # Test 1: Store and track topo hash index
    def test_topo_hash_index():
        mem = RegimeMemory(max_per_symbol=100, max_total=1000)
        fp1 = _make_fp(thash="0xAABBCCDD")
        fp2 = _make_fp(symbol="AAPL", thash="0xEEFF0011")
        mem.store(fp1)
        mem.store(fp2)
        stats = mem.get_stats()
        tr = TestResult(name="topo_hash_index", suite=suite.name, passed=False)
        if stats["topo_hashes"] >= 2 and stats["total_fingerprints"] == 2:
            tr.passed = True
            tr.detail = f"hashes={stats['topo_hashes']} fps={stats['total_fingerprints']}"
        else:
            tr.detail = str(stats)
        return tr

    # Test 2: Risk multiplier with bad outcome history
    def test_risk_multiplier():
        mem = RegimeMemory(max_per_symbol=100, max_total=1000)
        # Store several regimes with negative outcomes
        for i in range(5):
            fp = _make_fp(symbol="TSLA", stab=0.3, thash=f"0xBAD{i}")
            fp.outcome_pnl = -0.05
            fp.outcome_direction = "SHORT"
            mem.store(fp)
        mult = mem.get_risk_multiplier(symbol="TSLA")
        tr = TestResult(name="risk_multiplier_tightens", suite=suite.name, passed=False)
        if mult <= 1.0:
            tr.passed = True
            tr.detail = f"multiplier={mult:.3f}"
        else:
            tr.detail = f"Expected ≤1.0, got {mult:.3f}"
        return tr

    # Test 3: Topology → regime hash round-trip
    def test_topo_hash_roundtrip():
        import numpy as np
        np.random.seed(7)
        values = (np.random.randn(60) * 5).tolist()
        topo = compute_market_topology(values)
        fp = _make_fp(thash=topo.regime_hash, stab=topo.stability)
        tr = TestResult(name="topo_hash_roundtrip", suite=suite.name, passed=False)
        if fp.topological_hash == topo.regime_hash and fp.persistence_stability == topo.stability:
            tr.passed = True
            tr.detail = f"hash={topo.regime_hash[:16]}... stab={topo.stability:.3f}"
        else:
            tr.detail = "hash or stability mismatch"
        return tr

    # Test 4: Memory capacity enforcement
    def test_capacity_enforcement():
        mem = RegimeMemory(max_per_symbol=3, max_total=10)
        for i in range(20):
            fp = _make_fp(symbol="SPY", thash=f"0x{i:04X}")
            mem.store(fp)
        stats = mem.get_stats()
        tr = TestResult(name="capacity_enforcement", suite=suite.name, passed=False)
        if stats["total_fingerprints"] <= 10:
            tr.passed = True
            tr.detail = f"stored={stats['total_fingerprints']} (max_total=10)"
        else:
            tr.detail = f"Expected ≤10, got {stats['total_fingerprints']}"
        return tr

    tests = [test_topo_hash_index, test_risk_multiplier,
             test_topo_hash_roundtrip, test_capacity_enforcement]

    for test_fn in tests:
        try:
            tr = test_fn()
        except Exception as e:
            tr = TestResult(name=test_fn.__name__, suite=suite.name,
                            passed=False, detail=str(e)[:200])
        suite.tests.append(tr)

    suite.elapsed_ms = (time.perf_counter() - t0) * 1000
    suite.passed = sum(1 for t in suite.tests if t.passed)
    suite.failed = sum(1 for t in suite.tests if not t.passed)
    return suite


# ══════════════════════════════════════════════════════════════════
# CLI + Report
# ══════════════════════════════════════════════════════════════════

def print_suite(s: SuiteResult):
    status = "✅" if s.ok() else "❌"
    print(f"\n  {status} {s.name} — {s.passed}/{len(s.tests)} passed ({s.elapsed_ms:.1f}ms)")
    for t in s.tests:
        icon = "✓" if t.passed else ("⊘" if t.skipped else "✗")
        detail = f" — {t.detail}" if t.detail else ""
        print(f"    {icon} {t.name}{detail}")


def main():
    parser = argparse.ArgumentParser(
        description="Apex17 Robotics Engine Proof Suite"
    )
    parser.add_argument("--skip-cuda", action="store_true",
                        help="Skip CUDA-dependent tests")
    parser.add_argument("--skip-cpp", action="store_true",
                        help="Skip C++ binary tests")
    parser.add_argument("--output", default=None,
                        help="Output JSON path")
    parser.add_argument("--build-dir", default=None,
                        help="C++ build directory")
    args = parser.parse_args()

    build_dir = Path(args.build_dir) if args.build_dir else ROOT / "src" / "apex17-robotics" / "build"

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Apex17 Robotics Engine — Proof Suite                   ║")
    print(f"║  Time: {time.strftime('%Y-%m-%d %H:%M:%S'):<49s} ║")
    print("╚══════════════════════════════════════════════════════════╝")

    t_start = time.perf_counter()
    suites: List[SuiteResult] = []

    # C++ engine tests
    if not args.skip_cpp:
        print("\n  ─── C++ Engine Tests ───")
        suites.extend(run_cpp_suites(build_dir))
    else:
        print("\n  ⊘ C++ tests skipped")

    # Python suites
    print("\n  ─── Python Topology & Memory Tests ───")
    suites.append(run_market_topology_suite())
    suites.append(run_regime_memory_suite())
    suites.append(run_topological_guard_suite())

    elapsed = time.perf_counter() - t_start

    # Build report
    run_id = hashlib.sha256(f"{time.time()}".encode()).hexdigest()[:12]
    report = ProofReport(
        run_id=run_id,
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        elapsed_sec=round(elapsed, 3),
    )
    report.stamp(suites)

    # Print results
    print("\n" + "═" * 58)
    for s in suites:
        print_suite(s)

    print("\n" + "═" * 58)
    verdict = "✅ ALL PASSED" if report.all_passed else "❌ FAILURES DETECTED"
    print(f"  {verdict}")
    print(f"  {report.total_passed}/{report.total_tests} tests "
          f"({report.total_skipped} skipped) "
          f"in {report.elapsed_sec:.2f}s")
    print(f"  Digest: {report.digest}")

    # Write JSON
    if args.output:
        out_path = Path(args.output)
    else:
        results_dir = Path(__file__).parent.parent / "results" / run_id
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / "robotics_proof.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(asdict(report), f, indent=2)
    print(f"  Report: {out_path}")
    print("═" * 58)

    return 0 if report.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
