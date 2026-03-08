<div align="center">

# SignalBrain-OS

### Deterministic Autonomy on Local Silicon

**The first AI kernel that uses topology instead of neural networks for O(1) structural recognition.**

[![Website](https://img.shields.io/badge/🌐_Website-signalbrain.ai-0A84FF?style=for-the-badge)](https://signalbrain.ai)
[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Try_It_Now-28C840?style=for-the-badge)](https://signalbrain.ai/demo/)
[![Patents](https://img.shields.io/badge/📄_Patents-15_Filed-BFA165?style=for-the-badge)](https://signalbrain.ai/patents/)
[![Tests](https://img.shields.io/badge/✅_Tests-42/42_Passing-28C840?style=for-the-badge)](https://signalbrain.ai/performance/)

**5 Domains** · **42 Tests** · **15 Patents** · **Sub-5ms Edge Latency** · **Zero Cloud Dependency**

---

</div>

## What is SignalBrain-OS?

SignalBrain-OS is a **deterministic autonomy kernel** that uses **H₀ persistent homology** (topological data analysis) to create structural fingerprints from raw sensor data — not neural network inference.

This gives it three properties no other AI framework has:

| Property | What It Means |
|----------|---------------|
| **O(1) Recall** | Recognizes any previously seen pattern in constant time via hash lookup, regardless of database size |
| **Deterministic** | Same input → same fingerprint → same decision. Every time. Auditable. |
| **Edge-Native** | Runs on local silicon. Zero cloud dependency. Works air-gapped (DDIL) |

> **The core insight:** Topology is invariant to noise, rotation, and scale. A LiDAR point cloud, CT scan, radar return, or network flow all have measurable topological structure. SignalBrain-OS exploits this to achieve structural recognition faster than any neural network.

---

## 5 Proven Domains

| Domain | Engine | Latency | Key Capability | Proof |
|--------|--------|---------|----------------|-------|
| 🤖 **[Robotics](https://signalbrain.ai/robotics/)** | Apex17 Spatial | 35ms CUDA | 1M point cloud → topological scene identity | [run_robotics_proof.py](proof-artifacts/benchmarks/run_robotics_proof.py) |
| 💊 **[Healthcare](https://signalbrain.ai/healthcare/)** | Apex17 Clinical | 4.1ms | CT/MRI/ECG/Labs → 3-agent council → acuity score | [run_clinical_proof.py](proof-artifacts/benchmarks/run_clinical_proof.py) |
| 🎯 **[Defense & ISR](https://signalbrain.ai/defense/)** | Apex17 ISR | 4.5ms edge | SAR/SIGINT/IMINT → emitter fingerprint → ROE-auditable | [run_isr_proof.py](proof-artifacts/benchmarks/run_isr_proof.py) |
| 🔐 **[Cybersecurity](https://signalbrain.ai/cyber/)** | Apex17 Cyber | 2.5ms edge | NetFlow/DNS/EDR → zero-day topology → NIST-auditable | [run_cyber_proof.py](proof-artifacts/benchmarks/run_cyber_proof.py) |
| 📈 **Markets** | Regime Detector | 0.16ms CPU | OHLCV → regime fingerprint → O(1) regime recall | [run_canonical.py](proof-artifacts/benchmarks/run_canonical.py) |

**Every claim is backed by a runnable proof script.** See [`proof-artifacts/`](proof-artifacts/) for the complete test suite.

---

## How It Works

```
Raw Sensor Data → H₀ Persistent Homology → 64-bit Fingerprint → O(1) Hash Recall → Council Consensus → Policy Gate → Decision
     │                    │                       │                    │                    │               │            │
  LiDAR/CT/SAR     Topological features    Deterministic hash    SceneMemory/        3-agent debate    Apex17       Auditable
  NetFlow/OHLCV    Birth-death pairs       Locality-sensitive    RegimeMemory/       Consensus ≥2/3    Policy       trace
                                                                 EmitterMemory                        Compiler
```

### The Pipeline (Sub-5ms)

1. **Signal Ingest** — Raw sensor data (LiDAR, CT scan, radar, NetFlow, market data)
2. **H₀ Topology Extraction** — Persistent homology computes connected components and persistence diagrams
3. **Fingerprint** — Topology compressed into deterministic 64-bit hash via locality-sensitive hashing
4. **O(1) Memory Recall** — Hash lookup in domain-specific memory store (constant time, regardless of DB size)
5. **Council Consensus** — 3-agent modality-gated council debates and votes (≥2/3 required)
6. **Policy Gate** — Apex17 compiler enforces domain-specific rules (ROE/FDA/NIST)
7. **Decision + Audit** — Deterministic output with full provenance chain

---

## SignalBrain-OS vs. Everything Else

| Capability | **SignalBrain-OS** | LangChain | AutoGPT | CrewAI | Anduril Lattice |
|------------|:------------------:|:---------:|:-------:|:------:|:---------------:|
| **Deterministic Replay** | ✅ Merkle-anchored | ❌ | ❌ | ❌ | ❌ |
| **O(1) Recall** | ✅ Hash table | ❌ O(n) | ❌ O(n) | ❌ O(n) | ❌ |
| **Edge-Native (No Cloud)** | ✅ Local silicon | ❌ Cloud API | ❌ Cloud API | ❌ Cloud API | ✅ |
| **Policy Compiler** | ✅ Apex17 | ❌ | ❌ | ❌ | Partial |
| **Sub-5ms Latency** | ✅ All domains | ❌ seconds | ❌ seconds | ❌ seconds | Partial |
| **Topology-Based** | ✅ H₀ PH | ❌ Neural | ❌ Neural | ❌ Neural | ❌ Neural |
| **Patents Filed** | **15** | 0 | 0 | 0 | Classified |
| **Cross-Domain (5+)** | ✅ | ❌ | ❌ | ❌ | Defense only |
| **Auditable (ROE/FDA/NIST)** | ✅ | ❌ | ❌ | ❌ | ROE only |

---

## Run the Proof Suite

```bash
# Clone the repo
git clone https://github.com/whitestone1121-web/Signalbrain-OS.git
cd Signalbrain-OS

# Run all domain proofs
python3 proof-artifacts/benchmarks/run_robotics_proof.py
python3 proof-artifacts/benchmarks/run_clinical_proof.py
python3 proof-artifacts/benchmarks/run_isr_proof.py
python3 proof-artifacts/benchmarks/run_cyber_proof.py
python3 proof-artifacts/benchmarks/run_canonical.py

# Expected: 42/42 tests passing, 100% success rate
```

---

## Patent Portfolio

15 patent claims across 4 categories. Every patent maps to a production-deployed subsystem.

| Category | Count | Coverage |
|----------|-------|----------|
| GPU Infrastructure | 5 | O(1) scheduling, VRAM state management, persistent kernel |
| Data & Signal | 5 | Topological encoding, Merkle audit, replay tokens |
| Governance | 3 | Apex17 policy compiler, council consensus, sentinel validation |
| World Index | 2 | Cross-domain structural memory, O(1) recall architecture |

**[View full patent portfolio →](https://signalbrain.ai/patents/)**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     SignalBrain-OS Kernel                     │
├──────────────┬──────────────┬──────────────┬────────────────┤
│  Apex17      │  H₀ Topology │  O(1) World  │  GPU Council   │
│  Policy      │  Engine       │  Index       │  Consensus     │
│  Compiler    │  (PH + LSH)  │  (Hash Table)│  (3-agent)     │
├──────────────┴──────────────┴──────────────┴────────────────┤
│                    Domain Adapters                            │
├──────────┬──────────┬──────────┬──────────┬─────────────────┤
│ Robotics │Healthcare│ Defense  │  Cyber   │    Markets      │
│ (LiDAR)  │(CT/MRI)  │(SAR/SIG) │(NetFlow) │   (OHLCV)      │
└──────────┴──────────┴──────────┴──────────┴─────────────────┘
```

---

## Links

| Resource | URL |
|----------|-----|
| 🌐 **Website** | [signalbrain.ai](https://signalbrain.ai) |
| 🚀 **Live Demo** | [signalbrain.ai/demo](https://signalbrain.ai/demo/) |
| 📊 **Performance** | [signalbrain.ai/performance](https://signalbrain.ai/performance/) |
| 🔬 **Technology** | [signalbrain.ai/technology](https://signalbrain.ai/technology/) |
| 📄 **Whitepaper** | [signalbrain.ai/whitepaper](https://signalbrain.ai/whitepaper/) |
| 📋 **Patents** | [signalbrain.ai/patents](https://signalbrain.ai/patents/) |

---

<div align="center">

**Built by [Alan Samaha](https://www.linkedin.com/in/alansamaha/)** · **[SignalBrain, Inc.](https://signalbrain.ai)** · **© 2026**

*Sovereign intelligence infrastructure. Not another wrapper.*

</div>
