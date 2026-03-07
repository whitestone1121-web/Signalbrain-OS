# SignalBrain-OS Proof Artifacts

**Reproducible benchmarks and verification scripts for VC technical diligence.**

> These artifacts wrap SignalBrain-OS production modules. Every script targets
> live code — no mocks, no stubs. Clone, install, run, verify.

---

## Architecture at a Glance

```
SignalBrain-OS
├── Inference Plane
│   ├── Director (Llama-3.1-70B-AWQ-INT4)  — strategic reasoning
│   └── Council  (Dolphin3.0-Qwen2.5-3b)   — specialist consensus
│
├── Apex17 Policy Compiler
│   └── Probabilistic LLM intent → Deterministic compiled action
│       4 policies: Technical · Sentiment · Flow · Volatility
│
├── Apex17 Robotics Engine (CUDA + C++20)
│   ├── Spatial Perception: LiDAR → USI Topology → Motion Intent → Safety Policy
│   ├── H₀ Persistent Homology — topological fingerprinting
│   ├── SpatialCouncil: 4-agent deliberation (Solidity · Flow · Topology · Risk)
│   └── O(1) SceneMemory recall via topological hash
│
├── PicoAgent Swarm Runtime
│   ├── Sentinel Pipeline: Ingest → Router → Validator
│   ├── Swarm Mode: lightweight fire-and-forget execution
│   └── Evidence Packs: structured provenance per decision
│
├── Dual Trading Loops
│   ├── CIO Heartbeat:   macro strategy → Redis → approved universe
│   └── Titan Trader:    tactical execution (sub-second, 50 symbols)
│       Mutual exclusion: CIO publishes, Titan consumes — no conflict
│
├── Cryptographic Audit
│   ├── USI Archive (SQLite WAL + HMAC-SHA256)
│   └── Merkle Anchor (per-100-cycle root → GDrive/S3 WORM)
│
└── Hardware: RTX PRO 6000 Blackwell (96 GB · 81.6 GB allocated)
```

---

## Prerequisites

- **Python 3.10+** — no other dependencies required
- Works standalone (demo mode) or inside the full SignalBrain-OS environment

> **Demo Mode**: When cloned standalone, all scripts run in DEMO MODE with
> a synthetic policy engine. Results prove the API contract and determinism.
> For production data, run inside the full SignalBrain-OS environment.

---

## Quick Start

Clone and enter:

```bash
git clone https://github.com/whitestone1121-web/Signalbrain-OS.git
cd Signalbrain-OS
```

Run a 10-second benchmark:

```bash
python proof-artifacts/benchmarks/run_canonical.py --duration 10
```

Run the **Robotics Engine proof suite** (15 tests across 4 suites):

```bash
python proof-artifacts/benchmarks/run_robotics_proof.py --skip-cpp
```

Run the full expanded policy test suite (31 tests across 6 categories):

```bash
python proof-artifacts/policy/run_all.py --verbose
```

Run just the core policy enforcement tests:

```bash
python proof-artifacts/policy/rejection_suite.py --verbose
```

Verify deterministic replay (use the report path printed by the benchmark above):

```bash
python proof-artifacts/replay/verify.py --run proof-artifacts/results/YOUR_RUN_ID/report.json --assert deterministic_actions
```

---

## Expanded Policy Test Harness — 31 Tests Across 6 Suites

Run all suites with one command:

```bash
python proof-artifacts/policy/run_all.py --verbose --output proof-artifacts/results/expanded_policy_suite.json
```

### Suite 1: `rejection_suite.py` — Core Policy Enforcement (5 tests)

| Test | What It Proves |
|------|----------------|
| `flat_market_neutral` | No false positives in flat markets |
| `confidence_floor` | Ambiguous signals never breach confidence threshold |
| `deterministic_policy` | 1,000 iterations → identical output |
| `missing_fields` | Incomplete data → safe fallback (no crash) |
| `extreme_rsi` | Handles market extremes without policy violation |

### Suite 2: `adversarial_suite.py` — Schema Robustness (6 tests)

| Test | What It Proves |
|------|----------------|
| `nan_fields` | NaN inputs → safe fallback, no crash |
| `inf_fields` | +Inf/−Inf inputs → safe fallback |
| `negative_price` | Negative price → NEUTRAL or None |
| `zero_volume` | Zero volume → no ZeroDivisionError |
| `extreme_outliers` | RSI=1e12 → clamped, confidence ≤ 1.0 |
| `string_coercion` | String in numeric field → caught |

### Suite 3: `policy_matrix_suite.py` — Golden Scenarios (11 tests)

Each test constructs a specific market condition matching a documented DSL rule
and asserts the expected action fires.

Covers all 4 policy agents: Technical, Sentiment, Flow, Volatility.

### Suite 4: `temporal_suite.py` — Anti-Thrash Consistency (3 tests)

| Test | What It Proves |
|------|----------------|
| `micro_jitter_stable` | ±0.1% noise → ≤5% action changes |
| `threshold_crossing_count` | RSI ramp 20→80 → limited transitions |
| `boundary_oscillation` | RSI 49.9↔50.1 → stable output |

### Suite 5: `stress_suite.py` — Latency & Overload (4 tests)

| Test | What It Proves |
|------|----------------|
| `single_symbol_latency` | 10,000 calls → p99 < 500μs |
| `batch_50_symbols` | 200 calls (50 × 4 agents) → < 50ms |
| `batch_200_symbols` | 800 calls (200 × 4 agents) → < 200ms |
| `burst_determinism` | 1,000 burst calls → identical hashes |

### Suite 6: `invariance_suite.py` — Cross-Entropy Invariance (2 tests)

| Test | What It Proves |
|------|----------------|
| `multiprocess_invariance` | 4 child processes → identical hashes |
| `reimport_invariance` | Import → draft → reimport → same hashes |

---

## Apex17 Robotics Engine Proof Suite — 15 Tests Across 4 Suites

Run the full robotics proof suite:

```bash
python proof-artifacts/benchmarks/run_robotics_proof.py
python proof-artifacts/benchmarks/run_robotics_proof.py --skip-cpp   # Python-only
```

### Suite 1: Market Topology Engine (6 tests)

Tests the H₀ Persistent Homology engine — the same topological fingerprinting
used in both spatial perception (Apex17) and market regime detection.

| Test | What It Proves |
|------|----------------|
| `linear_trend_stability` | Monotonic signal → stability ≥ 0.99 |
| `random_walk_low_stability` | Random walk → lower stability (detects noise) |
| `deterministic_hash` | Same input → identical regime hash (deterministic) |
| `latency_under_5ms` | 100 iterations avg < 5ms (production-grade speed) |
| `complex_signal_entropy` | Complex signal → entropy > 1.0 (richness detected) |
| `output_fields_complete` | All 5 required output fields present |

### Suite 2: Regime Memory — O(1) Hash Recall (5 tests)

Tests the 20-dimensional RegimeFingerprint and the topological hash-based
memory recall system.

| Test | What It Proves |
|------|----------------|
| `fingerprint_20d_vector` | Vector shape is exactly (20,) |
| `o1_hash_recall` | Store → recall by topological hash → exact match |
| `unknown_hash_empty` | Unknown hash → empty result (no false positives) |
| `memory_stats` | Stats report 20 dims + correct hash count |
| `summary_has_topo` | Human-readable summary includes `topo=` field |

### Suite 3: Topological Guard — Director Integration (4 tests)

Tests the topological guard / veto mechanism that connects the perception
engine to the Director's decision-making.

| Test | What It Proves |
|------|----------------|
| `topo_hash_index` | Multiple fingerprints tracked by hash |
| `risk_multiplier_tightens` | Bad outcome history → multiplier ≤ 1.0 |
| `topo_hash_roundtrip` | Topology engine hash → fingerprint → match |
| `capacity_enforcement` | Ring buffer respects max_total capacity |

### Suite 4: C++ Engine (57 tests — requires CUDA build)

Orchestrates the compiled C++ test binaries:
- `spatial_prior_tests` (36 tests) — SpatialPrior engine, config validation, compute, RMQ, PH
- `spatial_council_tests` (21 tests) — 4-agent council, Director deliberation, persistence modifiers
- `apex17_smoke_test` — end-to-end smoke test

### Expected Output

```
✅ Market Topology Engine — 6/6 passed (1597.1ms)
✅ Regime Memory — 5/5 passed (608.1ms)
✅ Topological Guard — 4/4 passed (5.0ms)

✅ ALL PASSED — 15/15 tests (0 skipped) in 2.21s
Digest: 2ea6bd56e8f2b2ba
```

---

## Benchmarks & Replay

### `benchmarks/run_canonical.py`
**Claim**: Apex17 evaluates 4-agent policy drafts at µs-scale latency with deterministic outputs.

| Metric | What It Measures |
|--------|------------------|
| `throughput_eval_per_sec` | Policy evaluations per second |
| `bypass_rate` | Fraction of decisions resolved without LLM |
| `latency.p50_us` / `p99_us` | Per-evaluation latency |
| `decision_digest` | SHA-256 of all policy hashes — proves determinism |

### `benchmarks/reproduce_published.py`
Runs `run_canonical.py` with the exact parameters used in signalbrain.ai claims,
then validates results against stated thresholds.

```bash
python proof-artifacts/benchmarks/reproduce_published.py --suite website_v1
```

### `replay/verify.py`
Three independent checks:
1. **USI HMAC Integrity** — Re-verifies HMAC-SHA256 signatures
2. **Merkle Anchor Roots** — Rebuilds tree, confirms root matches receipt
3. **Deterministic Replay** — Re-runs benchmark, compares digests

In demo mode, performs self-consistency (runs twice, verifies identical digests).

### `presets/blackwell_82gb.yml`
Full production hardware preset:
- VRAM budget: Director (69.1 GB) + Council (17.3 GB) = 81.6 GB
- Apex17 policy compiler configuration
- Dual trading loops: CIO heartbeat (macro) + Titan Trader (tactical)
- Cryptographic audit (USI + Merkle anchoring)

---

## Dual Trading Loop Architecture

A key differentiator: SignalBrain-OS runs two independent execution planes
sharing the same Brain gateway.

```
  ┌──────────────────────────────────────┐
  │  Pico Swarm (CIO Heartbeat)          │
  │  Macro strategy · Universe approval   │
  │  Output → Redis (cio_universe)        │
  └──────────────┬───────────────────────┘
                 │ approved symbols
  ┌──────────────▼───────────────────────┐
  │  Titan Trader (Tactical Loop)         │
  │  Sub-second execution · 50 symbols    │
  │  Apex17 policy gate · Alpaca exec     │
  │  Consumes CIO universe for alignment  │
  └──────────────────────────────────────┘
```

Mutual exclusion: the CIO publishes macro intelligence; Titan consumes it.
Neither can override the other — hierarchical decision authority.

---

## Expected Output

```
╔══════════════════════════════════════════════════════╗
║  SignalBrain-OS Canonical Benchmark                 ║
║  Preset: blackwell_82gb                             ║
║  Duration: 10s | Seed: 42                           ║
╚══════════════════════════════════════════════════════╝

  Evaluations:      48,000
  Throughput:       4,800.0 eval/sec
  Bypass Rate:      62.4%
  Latency p50:      3.2 µs
  Latency p99:      18.7 µs
  Decision Digest:  a3f8d1b60b3b4b1a...
```

*Numbers are illustrative; actual results depend on hardware.*

---

## Directory Structure

```
proof-artifacts/
├── README.md
├── benchmarks/
│   ├── run_canonical.py               ← Core benchmark
│   ├── run_robotics_proof.py          ← Apex17 Robotics proof (15 tests)
│   └── reproduce_published.py         ← Validate website claims
├── replay/
│   └── verify.py                      ← USI + Merkle + replay checks
├── policy/
│   ├── run_all.py                     ← Orchestrator (all 6 suites)
│   ├── rejection_suite.py             ← Core enforcement (5 tests)
│   ├── adversarial_suite.py           ← Schema robustness (6 tests)
│   ├── policy_matrix_suite.py         ← Golden scenarios (11 tests)
│   ├── temporal_suite.py              ← Anti-thrash (3 tests)
│   ├── stress_suite.py                ← Latency & overload (4 tests)
│   └── invariance_suite.py            ← Cross-entropy (2 tests)
├── presets/
│   └── blackwell_82gb.yml             ← Production hardware config
├── signalbrain/
│   ├── __init__.py
│   ├── compiler.py                    ← Policy compiler shim
│   ├── audit.py                       ← USI audit interface
│   └── anchor.py                      ← Merkle anchor interface
└── results/                           ← Generated by test runs
```

---

*SignalBrain-OS · 15 U.S. patent applications · Built on Blackwell*

