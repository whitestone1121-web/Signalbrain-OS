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

## Quick Start

```bash
# 1. Clone and enter
git clone https://github.com/whitestone1121-web/Signalbrain-OS.git
cd Signalbrain-OS

# 2. Run a 10-second benchmark
python proof-artifacts/benchmarks/run_canonical.py --duration 10

# 3. Verify deterministic replay
python proof-artifacts/replay/verify.py \
    --run results/<run_id>/report.json

# 4. Run Apex17 policy enforcement tests
python proof-artifacts/policy/rejection_suite.py --verbose
```

---

## What Each Script Proves

### `benchmarks/run_canonical.py`
**Claim**: Apex17 evaluates 4-agent policy drafts at µs-scale latency with deterministic outputs.

| Metric | What It Measures |
|--------|------------------|
| `throughput_eval_per_sec` | Policy evaluations per second |
| `bypass_rate` | Fraction of decisions resolved without LLM |
| `latency.p50_us` / `p99_us` | Per-evaluation latency |
| `decision_digest` | SHA-256 of all policy hashes — proves determinism |
| `gpu.vram_used_mb` | GPU memory under load |

**Output**: `results/<run_id>/report.json`

### `benchmarks/reproduce_published.py`
**Claim**: Published website metrics are not aspirational — they're reproducible.

Runs `run_canonical.py` with the exact parameters used in signalbrain.ai claims,
then validates results against stated thresholds.

```bash
python proof-artifacts/benchmarks/reproduce_published.py --suite website_v1
```

### `replay/verify.py`
**Claim**: Every decision in the system is tamper-evident and deterministically replayable.

Three independent checks:
1. **USI HMAC Integrity** — Re-verifies HMAC-SHA256 signatures on the last N archived USI records
2. **Merkle Anchor Roots** — Rebuilds Merkle tree from leaf hashes, confirms root matches archived receipt
3. **Deterministic Replay** — Re-runs a previous benchmark, compares decision digests bit-for-bit

```bash
python proof-artifacts/replay/verify.py \
    --usi-db data/usi_audit.db \
    --run results/canonical.json \
    --assert deterministic_actions merkle_anchor usi_integrity
```

### `policy/rejection_suite.py`
**Claim**: The Apex17 compiler enforces policy at the kernel level — before any capital-facing action.

5 enforcement tests:
| Test | What It Proves |
|------|----------------|
| `flat_market_neutral` | No false positives in flat markets |
| `confidence_floor` | Ambiguous signals never breach confidence threshold |
| `deterministic_policy` | 1,000 iterations → identical output |
| `missing_fields` | Incomplete data → safe fallback (no crash) |
| `extreme_rsi` | Handles market extremes without policy violation |

### `presets/blackwell_82gb.yml`
Full production hardware preset documenting:
- VRAM budget: Director (69.1 GB) + Council (17.3 GB) = 81.6 GB
- Apex17 policy compiler configuration
- PicoAgent swarm runtime with sentinel pipeline
- Dual trading loops: CIO heartbeat (macro) + Titan Trader (tactical)
- Cryptographic audit (USI + Merkle anchoring)
- Observability stack (Prometheus + Grafana + DCGM)

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
├── README.md                          ← You are here
├── benchmarks/
│   ├── run_canonical.py               ← Core benchmark
│   └── reproduce_published.py         ← Validate website claims
├── replay/
│   └── verify.py                      ← USI + Merkle + replay checks
├── policy/
│   └── rejection_suite.py             ← Apex17 enforcement tests
├── presets/
│   └── blackwell_82gb.yml             ← Production hardware config
└── results/                           ← .gitignored, generated by runs
```

---

*SignalBrain-OS · 15 U.S. patent applications · Built on Blackwell*
