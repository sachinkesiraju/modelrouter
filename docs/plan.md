# Plan: `modelrouter` — Dynamic Adapter Dispatch for Multi-Agent Systems

Turning the Phase 1 CPU results (PorTAL cross-size refit + per-query routing on Qwen3 0.6B/1.7B) into (1) a verified open-source repo and (2) a publishable research contribution.

---

## 1. Repo name and scope

**Name:** `modelrouter` (working alternates: `adaptive-dispatch`, `latent-router`). Tagline: *"Route each query to the cheapest base model that can do the job — materializing the task adapter at runtime from a shared PorTAL latent."*

**In scope**
- Multi-base PorTAL training, refit, and adapter materialization (builds on `portallib`, which stays a dependency, not a fork).
- Routers: per-query score router, prompt-embedding router, task-latent `z` router.
- Dispatch policies: per-task floor, pre-routing, cascade/escalation, cost/latency/quality constrained policy engine.
- Runtime: HF Transformers backend (reference) and vLLM backend (performance), adapter hot-swap.
- Serving: OpenAI-compatible `/v1/completions` and `/v1/chat/completions` gateway, plus optional prefix-caching and request batching (inspired by Ramp Router's 100+ micro-decisions per request).
- Evaluation harness: task-suite eval, Pareto curves, kill-criteria checks, cost/latency oracle.
- Reproduction of all Phase 1 CPU numbers as smoke tests, plus GPU-scale experiments.

**Out of scope (v1)**
- Cross-family transfer (Qwen ↔ Llama) — stretch goal, documented as future work.
- Full agent frameworks (LangChain/AutoGen integration) — provide a thin adapter example only.
- Serving infrastructure (autoscaling, multi-tenant). One-node vLLM only.
- Training new PorTAL cores from scratch on novel modalities.

**Target audience:** ML engineers building LLM routing/cascades; researchers on LoRA hypernetworks and model routing; cost-optimization practitioners.

### 1.1 Inspiration: Ramp Router launch

Ramp recently launched **Ramp Router**, an OpenAI-compatible endpoint that tests new models on real work, routes each request to the cheapest model that clears the bar, and makes 100+ calls per request on what to cache, when to batch, and when to reach for something stronger. Their alpha group routes 2.75T tokens/month at 99.99% uptime, cutting costs 30%+ while keeping output quality the same or better. This validates the market need but the public product routes across already-deployed frontier/commercial models (GPT, Claude, Gemini, etc.).

`modelrouter` takes the same cost-quality routing principle and applies it at the **task-adapter level**: instead of picking among fixed API models, it materializes the right LoRA for the task on the chosen base model on demand. The same ideas — per-request routing, dynamic escalation, caching, batching, and an OpenAI-compatible gateway — are folded into the runtime and research stack.

### 1.2 What Ramp Router reveals about how to train/build our own router

From Ramp's public docs and benchmark work, the high-level training/operations loop is reasonably clear:

1. **Ground quality in a private, production-derived benchmark, not public leaderboards.**
   - Ramp built **Ramp SWE-Bench** from real merged PRs in its own codebase to avoid contamination and metric saturation.
   - For `modelrouter`, the equivalent is a fixed, versioned task suite (e.g., `portallib-tasks`) plus user-contributed JSONL tasks, evaluated by an automated judge (MC accuracy / EM / F1) rather than vibes.
   - Clue: train the router on *real task outcomes*, not model perplexity or generic benchmarks.

2. **Decision loop is: ingest → evaluate candidates → apply a user-set quality bar → rank survivors by cost/latency → select → fallback if needed.**
   - The "quality bar" is a per-route/workspace policy, not a hard-coded model.
   - A router is therefore trained to **predict, for each (task, query, base), whether the base clears the bar**.
   - Clue: frame router training as a binary (or ordinal) correctness predictor per base, then wrap it with a policy engine.

3. **Approved-model allowlist with tiers.**
   - Ramp docs list tiers: frontier, mid-tier, lightweight.
   - For `modelrouter`, the allowlist is the set of bases we have `PortalAlignment` for, and the tiers map to model families/sizes.
   - Clue: the router should never invent a base; it only chooses among registered, aligned bases.

4. **Policy is config, not code.**
   - Ramp lets users set `production`/`draft`/numeric quality bars, approved models, and fallback chains in a dashboard/config update without redeploying the app.
   - For `modelrouter`, keep policies as YAML (`quality_bar`, `max_latency_ms`, `allowed_bases`, `fallback_order`) so the same binary can serve multiple SLOs.

5. **Heavy observability and continuous re-evaluation.**
   - Ramp exposes decision traces (candidates, pass/fail, reason, fallback), spend views, and alerts.
   - Engineering post: per-product tagging + token-level visibility via Kafka/ClickHouse event pipeline.
   - Clue: emit JSONL traces on every request (`query`, `task`, `candidates`, `scores`, `chosen_base`, `cost`, `latency`, `outcome`) so the router can be retrained and policies audited.

6. **Routing is more than model selection.**
   - Ramp's launch thread mentions "100+ calls per request on what to cache, when to batch, when to reach for something stronger."
   - For `modelrouter`, extend the policy engine with cache-hit decisions, batching windows, and escalation triggers — but make them pluggable.

**Bottom line for training our own router:**
- Offline phase: evaluate every registered base on the private task suite to get per-(task, query, base) correctness scores.
- Train three router families on those labels:
  - **Score router:** logistic/MLP on per-base correctness logits (most accurate, most expensive at inference).
  - **Prompt-embedding router:** MLP on `sentence-transformers` embeddings (cheap, no base forward pass).
  - **Task-latent router:** MLP on PorTAL `z` (base-agnostic routing signal).
- Online phase: route, log traces, and periodically refresh the correctness labels as new bases/alignments are added.

### 1.3 Alignment with longhaul/TTR architecture

The user has a related project, `sachinkesiraju/longhaul`, whose execution runtime already separates an **inference plane** (`src/ttr/serve`) from the agent control plane. We can reuse several concepts:

- **Inference plane:** `longhaul` manages LLM cost, deadlines, and prefix caching. `modelrouter` runtime can be the same layer: it accepts a completion request, classifies task, routes to a base, materializes the adapter, and returns the result, while reporting cost/latency back to the caller.
- **Completion windows / latency tiers:** `longhaul` uses tiers like `asap`, `priority`, `standard`, `flex`. `modelrouter` policies can expose `max_latency_ms` and quality-bar presets (`production`, `draft`, `interactive`, `batch`) that map to those tiers.
- **Prefix amortization / prefix-cache:** `longhaul` reuses KV-cache across parallel shards. `modelrouter` vLLM backend should keep an LRU of materialized `(base, task)` LoRAs and prefix-cache hits so repeated queries for the same task avoid decoder and adapter-materialization cost.
- **Seam abstraction:** `longhaul` abstracts LLM/VM backends. In `modelrouter` this is the `Backend` seam: HF Transformers (reference), vLLM (performance), and a stub/mock backend for unit tests.
- **Durable execution / EntityStore:** `longhaul` journals steps and stores business entities outside agent state. For `modelrouter`, the equivalent is durable routing traces and a registry of tasks, bases, alignments, and artifacts in a SQLite/JSONL store, so the router can be retrained and policies audited without re-running the whole stack.
- **Cost-aware scheduling:** `longhaul` schedules inference by cost and deadline. `modelrouter` policy engine is the scheduler: `min cost` s.t. `quality >= bar` and `latency <= SLO`.

**Execution implication:** the first `modelrouter` runtime prototype can target the same architecture — an OpenAI-compatible server with a durable trace journal, adapter LRU, and policy-driven routing — so it is composable with `longhaul` agents later.

---

## 2. Repo structure

```
modelrouter/
├── src/modelrouter/
│   ├── training/        # PortalCore source training + per-base PortalAlignment training
│   ├── refit/           # Refit published source artifacts to new target bases
│   ├── routing/         # Score, prompt-embedding, and task-latent z routers
│   ├── dispatch/        # Policy engine: floor, pre-route, cascade, constrained optimization
│   ├── runtime/         # Backends (HF, vLLM), adapter materialization + injection
│   ├── eval/            # Task-suite eval, Pareto curves, kill-criteria, oracles
│   └── data/            # Task JSONL loaders, portallib-tasks integration, splits
├── experiments/         # Numbered, self-contained experiment configs + runners (exp01_cpu_smoke, exp02_gpu_refit, ...)
├── configs/             # YAML configs: models, tasks, routers, policies, hardware profiles
├── scripts/             # One-command entrypoints: train, refit, route, dispatch, benchmark, release
├── tests/               # Unit tests + CPU smoke test (tiny model, 2 tasks, <5 min)
├── docs/                # Architecture, API reference, experiment protocol, results log
├── examples/            # Quickstart, add-a-task, add-a-base, tune-a-policy notebooks/scripts
├── docker/              # CPU and CUDA Dockerfiles + compose for vLLM
└── .github/workflows/   # CI: lint, typecheck, unit, CPU smoke, docs build
```

One-line purposes:
| Dir | Purpose |
|---|---|
| `src/modelrouter/` | Installable library; everything importable and typed |
| `experiments/` | Frozen, reproducible experiment definitions producing the paper's tables/figures |
| `configs/` | All hyperparameters/hardware/model lists as declarative YAML — no magic numbers in code |
| `scripts/` | Thin CLI wrappers (`python scripts/dispatch.py --config ...`) |
| `tests/` | Correctness + a deterministic CPU smoke test gating CI |
| `docs/` | mkdocs site: quickstart, concepts, reproducing-the-paper page |
| `examples/` | Copy-paste-able extension recipes |
| `docker/` | Pinned images for CPU repro and GPU/vLLM runs |

---

## 3. Core modules and APIs

### 3.1 Training (`modelrouter.training`)
```python
core = PortalCoreTrainer(config).train(task_suite)          # source-train canonical hypernetwork + task_latents
align = AlignmentTrainer(core, base="Qwen/Qwen3-4B").train() # thin per-base decoder
artifact = PortalArtifact(core, task_latents, alignments={...})
artifact.save_pretrained("org/portal-qwen3-4b")             # HF-compatible
```
- Wraps `portallib` `PortalModel`; adds multi-base alignment management, checkpointing, W&B logging.
- Config-driven: rank, target modules (q/v default), epochs, examples/task, dtype.

### 3.2 Refit (`modelrouter.refit`)
```python
refitter = PortalRefitter.from_pretrained("RampPublic/portal-qwen3-1.7b")
new_artifact = refitter.refit(target_base="Qwen/Qwen3-0.6B", examples_per_task=200, epochs=3)
```
- Direct productization of `PortalAdapterRefitter` + the Phase 1 `phase1_qwen_cpu_*.py` refit script.
- Memory-aware: gradient accumulation + streaming dataset to respect the documented 8 GB CPU ceiling; auto-selects batch size from available RAM/VRAM.

### 3.3 Routers (`modelrouter.routing`)
Common interface:
```python
class Router(Protocol):
    def fit(self, features, labels) -> None
    def route(self, query, candidates: list[BaseSpec]) -> RoutingDecision  # scores + chosen base
    def save/load(...)
```
- `ScoreRouter`: logistic regression on per-query score features (top/second/margin/entropy/gold-diff) — the "22.6% savings, accuracy-improving" router. Requires running candidates → offline/oracle use.
- `PromptEmbeddingRouter`: per-base correctness MLPs on sentence-transformer embeddings (default `all-MiniLM-L6-v2`), with `delta` bias knob. Production path (prompt-only).
- `LatentRouter`: predicts best base from PorTAL task latent `z`; v1 per-task, v2 with a prompt→`z` mapper for per-query latent routing.
- `TaskClassifier`: prompt → task-id (MiniLM/mpnet), used by the end-to-end pipeline; pluggable.

**Training a router from scratch (Ramp-style loop):**
```python
# 1. Build an offline oracle dataset
oracle = Oracle(suite, bases).run()  # per-(query, base) correctness + cost/latency
# 2. Fit any router on the oracle labels
PromptEmbeddingRouter().fit(oracle.embeddings, oracle.correctness)
# 3. Online: route, log JSONL traces, refresh oracle as new bases/tasks arrive
```
- The `Oracle` is the private benchmark equivalent to Ramp SWE-Bench: it evaluates every registered base on every query in the task suite and returns pass/fail or metric labels.
- All routers are trained to predict those labels; the dispatch policy then applies the user-configured quality bar and cost/latency ranking.
- Traces mirror Ramp's observability: each request records candidates, per-base scores, pass/fail, chosen base, fallback, cost, and latency.

### 3.4 Dispatch (`modelrouter.dispatch`)
```python
policy = DispatchPolicy(router=..., constraint=Constraint(max_quality_drop=0.03) | Constraint(max_cost=...))
decision = policy.decide(query)          # pre-routing
cascade = CascadePolicy(order=[cheap, capable], escalation=LearnedEscalation(threshold=0.45))
```
- Policy engine takes a **cost/latency/quality model per base** (from the oracle, §7) and solves the constrained choice; supports per-task floor, per-query threshold, cascade with learned escalation, and "always-X" baselines as degenerate policies.
- All Phase 1 policies (floor 1.0–5.0, thresholds 0.2–0.9, delta sweeps, learned escalation 0.45/0.5) reproduce as config files.

### 3.5 Runtime (`modelrouter.runtime`)
```python
rt = Runtime(backend="vllm" | "hf", bases=[...], artifact=artifact)
out = rt.run(query, task=None)   # classify task if None → route → materialize LoRA → infer
```
- `HFBackend`: loads base, `PortalModel.generate(task)` → LoRA factors → `PortalInjector.activate` (reference path; already validated on CPU, ~50–60 ms materialization + <3 ms/forward).
- `VLLMBackend`: exports materialized factors to PEFT-format LoRA dir, serves via vLLM `LoRARequest` hot-swap; caches materialized adapters per (base, task) with LRU eviction.
- `Gateway`: OpenAI-compatible `/v1/completions` and `/v1/chat/completions` server that classifies the task, routes, materializes the adapter, and forwards to the selected backend. Supports optional prefix-cache hit optimization and small-batch request coalescing for throughput.
- Instrumentation: per-stage timing (classify, route, materialize, forward, cache hit, batch size) emitted as JSONL.

### 3.6 Evaluation harness (`modelrouter.eval`)
```python
report = Evaluator(suite, policies=[...]).run()   # accuracy, latency, cost per policy
report.pareto_curve(x="cost_savings", y="rel_accuracy").save("figs/pareto.png")
report.check_kill_criteria(min_savings=0.15, max_quality_drop=0.03)
```
- Task-suite eval (MC scoring + generative EM/F1), bootstrap CIs, per-task breakdown, Pareto AUC, and machine-checkable Go/No-Go gates.

---

## 4. Reproducibility requirements

- [ ] **Packaging:** `pyproject.toml` with `uv` lock (`uv.lock` committed); extras: `[gpu]` (vllm, flash-attn), `[dev]`, `[docs]`. Pin `portallib`, `transformers`, `torch`, `sentence-transformers`.
- [ ] **Docker:** `docker/cpu.Dockerfile` (repro of Phase 1 smoke) and `docker/cuda.Dockerfile` (vLLM, CUDA 12.x); published to GHCR with digest-pinned bases.
- [ ] **Data pipeline:** `portallib-tasks` pulled from HF with a pinned revision; `data/` module validates a documented custom-task JSONL schema (`{"task": ..., "prompt": ..., "choices": [...], "answer": ...}`); deterministic train/val/test splits with committed seed + split manifests (fixes the 15-example noise problem by construction at GPU scale).
- [ ] **Artifact versioning:** all artifacts (`PortalArtifact`, routers, policies) implement `save_pretrained`/`from_pretrained` with a `config.json` recording git SHA, config hash, dataset revision. Routers saved as safetensors/joblib + metadata.
- [ ] **CI (GitHub Actions):** ruff + ruff-format, mypy (strict on `src/`), pytest unit, and a **CPU smoke test** (≤10 min: refit a tiny random-init model on 2 tasks × 8 examples × 1 epoch, run all three routers + cascade, assert pipeline completes and metrics are within loose bounds). Nightly optional job runs the full Phase 1 CPU repro.
- [ ] **Benchmark scripts:** `scripts/benchmark.py --hardware {cpu,a100,h100}` producing the latency/cost oracle table; results committed to `docs/results/`.
- [ ] **Determinism:** global seed plumbing; document nondeterminism sources (vLLM batching, bf16).

---

## 5. GPU experimental plan

**Models (same family, three tiers):** Qwen3-1.7B (cheap), Qwen3-4B (mid), Qwen3-8B (capable). Rationale: keeps the validated Qwen3 tokenizer/architecture path; three tiers make the routing problem non-trivial and the Pareto frontier interesting. (Fallback if 8B is too costly per gate: 0.6B/1.7B/4B.)

**Setup**
- 1× A100 80GB (or H100) on Lambda/RunPod, ~$1.8–3/hr. All three bases fit simultaneously in bf16 (~28 GB weights) for vLLM multi-model serving; training runs one base at a time.
- vLLM ≥0.6 with `enable_lora=True`, `max_loras=8`, adapters exported per (base, task) on demand.
- **Cost/latency oracle:** measured per-base $/1K-token from vLLM throughput at batch sizes {1, 8, 32} × prompt lengths {128, 512}; plus a "API-price" synthetic oracle for the paper's cost columns. Stored as `configs/oracle_a100.yaml`.

**Scale:** full `portallib-tasks` suite (14 tasks, extend to ~20 with MMLU-subsets/GSM8K-MC if time allows); **300 train / 100 val / 100 test examples per task** (kills the 15-example val noise flagged in the report); refit epochs 3–5, batch 16–32, rank 8 (ablate 16).

**Budget estimate**
| Item | Est. GPU-hours | Est. cost |
|---|---:|---:|
| Refit/align 3 bases (or refit 2 from the 1.7B artifact) | 12–20 | $40–60 |
| Per-(base,task) independent LoRA baseline (3 bases × 14 tasks) | 20–30 | $60–90 |
| Router training + full eval sweeps | 6–10 | $20–30 |
| vLLM latency/cost benchmarking | 4 | $12 |
| Re-runs / 3 seeds on headline configs | 15 | $45 |
| **Total** | **~60–80** | **~$180–240** |

**Kill criteria (per gate, machine-checked by `check_kill_criteria`)**
1. GPU refit of 4B→1.7B alignment must beat random by ≥10 pp macro accuracy after epoch 1 — else stop and debug before spending on 8B.
2. Best router policy must achieve **≥15% cost savings at ≤3 pp accuracy drop** vs always-8B on ≥100-example test sets — else pivot (see §11 decision tree).
3. vLLM adapter swap overhead must be <10% of median request latency — else invest in adapter caching before continuing.

---

## 6. Baselines and ablations

| # | Study | Question answered |
|---|---|---|
| B1 | **Independent per-(base, task) LoRAs** (required baseline) | Does the shared PorTAL latent cost accuracy vs task-specific adapters? Compare accuracy and total artifact size/training cost. |
| B2 | Always-largest / always-smallest / **oracle router** (route by per-query ground-truth correctness) | Bounds of the Pareto frontier; oracle gives the routing-headroom ceiling. |
| B3 | Prompt-embedding vs per-query score vs task-latent `z` router | How much accuracy do you give up for prompt-only routing? Does `z` add signal over raw embeddings? |
| B4 | Cascade/escalation vs pre-routing | Which deployment pattern wins at equal quality drop? Include cascade's double-inference cost honestly. |
| A1 | Router training-set size sweep (50 → 3000 prompts) | Was the CPU overfitting (train acc ~98%) fixed by data? |
| A2 | Encoder ablation (MiniLM vs mpnet vs Qwen3-embedding) at GPU data scale | Report flagged mpnet losing on 210 prompts — does it flip with more data? |
| A3 | Known-task vs classified-task end-to-end | Quantify the task-classifier tax (Phase 1: dominant error source at 71–76%). |
| A4 | LoRA rank 8 vs 16; q,v vs q,k,v,o modules | Capacity vs materialization cost. |
| A5 | Refit vs from-scratch alignment for the cheapest base | Value of the published source artifact. |

---

## 7. Metrics and Pareto criteria

- **Quality:** accuracy (MC), EM/F1 (generative), macro over tasks, bootstrap 95% CIs; *relative accuracy vs always-largest*.
- **Cost:** measured vLLM latency (p50/p95), GPU-seconds/query, and oracle $/query; *cost savings %* vs always-largest.
- **Routing:** % routed per base, router AUROC vs oracle labels, task-classifier accuracy.
- **Overhead:** adapter materialization ms (decoder + injection), vLLM swap ms, cache hit rate.
- **Frontier:** Pareto curve (savings vs relative accuracy) per policy family; **normalized Pareto AUC** as the single-figure comparison; headline criterion **≥15% savings at ≤3 pp drop** (already met on CPU by the score router: 22.6% at +0.9 pp).
- **Wall-clock:** end-to-end request latency broken down by stage.

---

## 8. Artifact and model release plan

**Published on Hugging Face (org: e.g. `modelrouter`):**
- `portal-qwen3-{1.7b,4b,8b}` PorTAL artifacts (core + latents + alignment) — `save_pretrained` format, Apache-2.0 (subject to Qwen license compatibility; adapters carry Qwen3's Apache-2.0 fine).
- `router-prompt-minilm`, `router-latent-z` trained routers + task classifier.
- `portallib-tasks-splits` dataset revision with the committed split manifests.
- Model cards: training data, per-task metrics, intended use (routing research), limitations (same-family only, MC-heavy tasks), CO2/compute estimate.

**Stays in repo:** all code, configs, experiment definitions, results JSON, figures, oracle tables, CPU smoke artifacts (tiny random-init test fixtures only — no large binaries in git; use HF for anything >10 MB).

**Licenses:** code Apache-2.0; docs CC-BY-4.0; datasets inherit upstream licenses (documented per task).

---

## 9. README and examples

**Quickstart (one command):**
```bash
pip install modelrouter && modelrouter demo --config configs/demo_cpu.yaml
# downloads the 0.6B/1.7B artifacts + routers, dispatches 20 sample queries,
# prints the accuracy/cost table and saves a Pareto plot
```

**Examples (each a single runnable script + short doc):**
1. `examples/01_quickstart.py` — the demo above, annotated.
2. `examples/02_add_a_task.py` — write task JSONL → refit latents → eval → task appears in dispatch.
3. `examples/03_add_a_base.py` — refit a published artifact to a new base (e.g. Qwen3-4B) and register it with the runtime.
4. `examples/04_tune_policy.py` — sweep `delta`/threshold, plot the frontier, pick a policy under a quality constraint.
5. `examples/05_vllm_serving.py` — serve the full pipeline behind an OpenAI-compatible endpoint.
6. `examples/06_gateway_caching_batching.py` — run the gateway with prefix-cache hit logging and small-request batching, and compare end-to-end latency.

README sections: badges (CI, PyPI, HF), 60-second pitch + headline Pareto figure, quickstart, architecture diagram (prompt → task classifier → router → materialize → base), results table, reproducing-the-paper pointer, citation.

---

## 10. Publication outline

**Title:** *"One Latent, Many Bases: Per-Query Adapter Dispatch Across Model Sizes with Portable Task Representations"*

**Abstract sketch:** Multi-agent LLM systems waste compute by sending every query to the most capable model. Commercial routers such as Ramp Router now show that per-request model selection with caching, batching, and escalation can cut costs 30%+ at production scale. We take the same routing principle one level deeper: a shared portable task latent (PorTAL) lets heterogeneous base models act as interchangeable agents. A lightweight router picks the cheapest capable base per query and a LoRA adapter is materialized on it at runtime (~50 ms, <1% forward overhead). On N tasks across Qwen3 1.7B/4B/8B, prompt-only routing saves X% cost at ≤3 pp quality drop, and a score-based router *improves* accuracy while saving Y%, because differently-sized models make different errors. We release code, artifacts, and routers.

**Contributions:** (1) first per-query dispatch system over runtime-materialized cross-size adapters from a shared latent; (2) systematic comparison of three routing signal families (candidate scores, prompt embeddings, task latent `z`) and two deployment patterns (pre-route vs cascade); (3) evidence that task latents encode difficulty usable for base selection (78.6% LOO on CPU, to be confirmed at scale); (4) open, reproducible stack.

**Figures:** (F1) system diagram; (F2) headline Pareto frontier — all policies + oracle + always-X points; (F3) router-family comparison at equal savings; (F4) materialization overhead breakdown bar; (F5) per-task heatmap of chosen base; (F6) router accuracy vs training-set size (A1).

**Tables:** (T1) main results — policy × {accuracy, rel-acc, savings, %-routed} at GPU scale; (T2) B1 shared-latent vs independent LoRAs; (T3) ablations A2–A4; (T4) end-to-end known-task vs classified-task.

**Related work:** PorTAL and UAL (portable task representations, hypernetwork-generated LoRA); LoRA/PEFT and LoRA-hub-style composition; model routing & cascades (FrugalGPT, RouteLLM, LLM-Blender, AutoMix); commercial routers (Ramp Router — 2.75T tokens/month, 30%+ savings, OpenAI-compatible endpoint with caching/batching/escalation); speculative/early-exit execution; mixture-of-experts as intra-model routing (we route across models and materialize adapters at runtime).

**Go/No-Go conclusion section:** explicit statement against the pre-registered 15%/3pp criterion, including negative results (per-task floor weakness, mpnet regression, 8 GB refit ceiling).

**Venue:** arXiv + blog first; target workshop (ENLSP/ES-FoMo @ NeurIPS/ICML) or systems venue (MLSys) depending on how strong the vLLM numbers are.

---

## 11. Milestone roadmap with gates (2-week sprints)

| Sprint | Deliverable | Gate (Go/No-Go) |
|---|---|---|
| **S1** | Repo scaffold, packaging, CI, port Phase 1 scripts into `src/` modules, CPU smoke test green | G1: CPU smoke reproduces report's floor-1.0 and threshold-0.6 numbers within noise → Go |
| **S2** | HF runtime + eval harness + all routers/policies as library code; docs skeleton; Docker images | G2: full CPU pipeline runs from one config; kill-criteria checker works → Go |
| **S3** | GPU refits (1.7B/4B/8B), 300-ex data pipeline, independent-LoRA baseline (B1) | G3: kill criterion 1 (refit non-random) AND B1 gap ≤5 pp → Go; if latent costs >5 pp vs independent LoRAs, pivot to "routing over independent LoRAs" framing |
| **S4** | GPU router training + full sweeps (B2–B4, A1–A5), vLLM backend + oracle | G4: kill criterion 2 (≥15% savings at ≤3 pp) met by any policy → Go to paper; met only at 10–15% → Go with reframed claims; <10% → No-Go, publish negative-result blog |
| **S5** | Paper draft, figures from `experiments/`, HF artifact release + model cards, README/examples polish | G5: every number in the paper regenerable by one script → Go |
| **S6** | arXiv/blog release, community launch (HN/Twitter), issue triage, v0.1.0 tag | — |

**Decision tree at G3/G4:**
- Shared latent underperforms independent LoRAs badly → pivot: keep the dispatch/router contribution, swap the adapter source (still novel: routing + runtime LoRA swap study).
- Router savings collapse at scale (models' error sets converge) → pivot to cascade-only framing or add a 0.6B tier to widen the cost spread.
- vLLM swap overhead dominates → contribution becomes the materialization-caching design; report it honestly.
- Everything fails the 10% bar → stop; publish a rigorous negative-result write-up (the CPU→GPU non-transfer is itself informative).

---

## 12. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| GPU cost overrun | Med | Hard budget ($300); gates before each spend tier; refit (cheap) before source-train (expensive); spot instances + checkpointing |
| vLLM LoRA swap overhead erodes savings | Med | Pre-materialize + LRU-cache adapters per (base, task); measure swap cost first (kill criterion 3); fall back to HF backend numbers for the paper if needed |
| Small per-task datasets → noisy routing labels | High (proven on CPU) | 300/100/100 splits, bootstrap CIs, 3 seeds on headline rows; A1 sweep quantifies data need |
| Task classifier remains the bottleneck | Med | Report both known-task and classified-task numbers (A3); try a stronger/fine-tuned encoder; classifier improvement is orthogonal future work |
| Cross-family transfer expectations | High | Explicitly out of scope in README + paper limitations; one exploratory appendix run at most |
| Larger models converge (no routing headroom) | Med | Oracle-router bound (B2) measured early in S4 — if oracle headroom <15%, pivot immediately per decision tree |
| Qwen/dataset license issues | Low | Qwen3 is Apache-2.0; audit each task dataset license before HF release |
| 8 GB-style OOMs recur on GPU | Low | Memory-aware refitter (auto batch size), gradient accumulation, documented hardware minimums |

---

## 13. How the plan uses the attached CPU results

**Directly reusable (port into `src/` in S1):**
- Refit code path (`PortalAdapterRefitter` usage, 20-ex/3-epoch config) → `modelrouter.refit` + CPU smoke fixture.
- Router feature extraction + logistic/MLP training from `phase1_qwen_cpu_extras.py` and `prompt_router_*` → `modelrouter.routing`.
- End-to-end pipeline (`phase1_qwen_cpu_end_to_end*.py`) → `modelrouter.runtime` HF backend.
- Policy sweeps + result JSONs (`eval_results.json`, `dispatch_results.json`, `perquery_dispatch.json`, `cascade_policy.json`, etc.) → committed as `experiments/exp01_cpu_smoke/expected/` regression baselines.
- `materialization_timing.json` methodology → the overhead benchmark script.

**Becomes the CPU smoke test:** the entire Phase 1 pipeline at reduced scale (2–4 tasks, 8 examples, 1 epoch, 0.6B or tiny random-init) gates every CI run; the full 14-task version runs nightly.

**Must be re-run at GPU scale:** all headline numbers (Pareto tables, router comparisons, cascade results, LOO generalization, `z`-routing accuracy) — the CPU numbers rest on 15-example test slices and are directional only; the 50-example refit (OOM-killed) becomes trivially feasible; latency/cost columns must switch from raw CPU forward ms to the vLLM oracle.

**Already-answered questions we won't re-litigate:** materialization overhead is negligible (<1% per forward); margin-based escalation is too blunt (learned rule only); mpnet needs more data to beat MiniLM (folded into A1/A2 rather than assumed).

---

## 14. Execution log

**2026-07-18 — S1 scaffold + first modules landed:**
- Created `modelrouter` repo with `pyproject.toml` (hatchling + uv-ready), `README.md`, declarative `configs/smoke_cpu.yaml`, and `src/modelrouter/{data,routing,dispatch,runtime,eval,tracing}` packages.
- Ported and generalized Phase 1 CPU logic into typed modules:
  - `modelrouter.routing`: `ScoreRouter`, `PromptEmbeddingRouter`, `TaskClassifier`.
  - `modelrouter.dispatch`: `FloorPolicy`, `CascadePolicy`, `BaseSpec`, `RoutingDecision`.
  - `modelrouter.runtime`: `HFBackend`, `DispatchRuntime`, OpenAI-compatible `Gateway` (FastAPI), `TraceJournal`.
  - `modelrouter.eval`: `Oracle` dataset builder for per-(query, base) correctness labels.
  - `modelrouter.config`: YAML config loader via Pydantic.
- Added CLI `modelrouter smoke` and a CPU smoke test (`scripts/smoke_test.py`) that trains a `ScoreRouter` from the pre-computed Phase 1 score bundle and evaluates floor/cascade policies.
- Smoke test results (208 examples, Qwen3-0.6B vs Qwen3-1.7B, floor=1.2):
  - `always_cheap`: 56.7% accuracy
  - `always_capable`: 66.8% accuracy
  - `floor_policy`: 67.3% accuracy, 15.6% cost savings
  - `cascade_policy`: 62.0% accuracy, 31.5% cost savings
- Added `tests/test_dispatch.py` (4 unit tests, all passing) and `examples/01_quickstart.py` (prompt router + task classifier demo, passing).
- README references Ramp Router and `longhaul`/`TTR` inference-plane concepts.

**2026-07-18 — S2 CPU end-to-end + eval harness + Docker/CI:**
- Ported the live HF materialization pipeline into `modelrouter.runtime.HFBackend` (`load`, `score_task`, `score_dataset`, `close`).
- Implemented `modelrouter.eval` with `accuracy_from_scores`, `policy_stats`, `check_kill_criteria`, and `plot_pareto`.
- Created and ran `experiments/exp01_cpu_smoke/run.py`: full prompt → task classifier → `ScoreRouter` → dispatch policy → base model materialization → scoring, on `rte` and `cb` tasks (8 examples each).
- Results on CPU (Qwen3-0.6B refit vs Qwen3-1.7B source):
  - Task classifier: 81.2% test accuracy.
  - `always_cheap`: 68.8% accuracy, 50% savings.
  - `always_capable`: 87.5% accuracy, 0% savings.
  - `floor` (floor=1.2): **93.8% accuracy**, **12.5% savings** (quality *improved* over always-capable on this tiny slice).
  - `cascade` (threshold=0.5): 68.8% accuracy, 50% savings.
  - Kill-criteria check (≥15% savings, ≤3 pp quality drop vs capable): `floor` fails the savings bar by 2.5 pp; `cascade` fails the quality bar. This is expected on a 16-example test set; the larger GPU run is needed for a definitive verdict.
- Added `configs/smoke_gpu.yaml`, `docker/cpu.Dockerfile`, `docker/cuda.Dockerfile`, `.github/workflows/ci.yml`, and `docs/{architecture.md,reproducing.md}`.
- Added placeholder `examples/02_add_a_task.py` and `examples/03_add_a_base.py`.
- Updated README with quickstart and `experiments/exp01_cpu_smoke/run.py` instructions.

**2026-07-18 — S3 larger CPU end-to-end (4 tasks × 8 examples):**
- Optimized `experiments/exp01_cpu_smoke/run.py` to load each base once and score combined val+test rows, then split results by offset.
- Added `configs/smoke_cpu_4tasks.yaml` with `rte`, `cb`, `copa`, `wsc`.
- Live end-to-end results (32 test examples, Qwen3-0.6B refit vs Qwen3-1.7B source, floor=1.2):
  - Task classifier: 84.4% test accuracy.
  - `always_cheap`: 59.4% accuracy, 50% savings.
  - `always_capable`: 78.1% accuracy, 0% savings.
  - `floor` (floor=1.2): **75.0% accuracy**, **14.1% savings**, quality drop vs capable = 3.1 pp.
  - `cascade` (threshold=0.5): 59.4% accuracy, 50% savings.
  - Kill-criteria check: `floor` just misses both bars (savings 14.1% < 15%, drop 3.1 pp > 3 pp); `cascade` misses quality bar. This is near the Go/No-Go threshold and confirms the approach is in the right ballpark on CPU, but the decisive test still needs GPU scale and more data.
- Added `modelrouter.refit` wrapper and `modelrouter.training` placeholder.

**2026-07-18 — S4 robust policy sweep on full 208-example bundle:**
- Created `experiments/exp02_cpu_policy_sweep/` with `run.py`, `plot_pareto.py`, `task_latent_router.py`, and `report.md`.
- Used the pre-computed Phase 1 score bundle (14 tasks, 205 val + 208 test examples) to train `ScoreRouter`, `PromptEmbeddingRouter`, and a task-latent `z` router on the validation split and evaluate dispatch policies on held-out test examples.
- Tracked bootstrap 95% CIs (200 resamples) and leave-one-task-out generalization.
- Results (Qwen3-0.6B refit vs Qwen3-1.7B source, cost 1.0 vs 2.0):
  - `always_capable`: 66.8% accuracy, 0% savings.
  - `always_cheap`: 56.7% accuracy, 50% savings.
  - Oracle per-query routing upper bound: 75.5% accuracy, 28.4% savings.
  - **Score-router `floor=1.2`**: **67.3% accuracy**, **15.6% savings**, +0.5 pp vs `always_capable` (bootstrap CI: accuracy 63.0–71.6%, savings 13.5–17.9%).
  - **Prompt-router `delta=0.0`**: 63.5% accuracy, 13.5% savings (no candidate forward pass needed).
  - **Task-latent `z` router**: 78.6% leave-one-task-out accuracy at predicting whether the cheap base is at least as good as the capable base for a task.
- Leave-one-task-out averages:
  - Score-router `floor=1.2`: 70.7% accuracy, 19.6% savings.
  - Prompt-router `delta=0.0`: 67.6% accuracy, 14.9% savings.
- This is the first CPU result that clears the kill test (≥15% savings at ≤3 pp drop) using the score router, and the first strong evidence that a prompt-only router and the shared task latent `z` are viable routing signals.
- Generated `results/results.json`, `results/pareto.png`, `results/task_latent_router.json`, and `report.md`.

**2026-07-18 — S5 prompt-router ablation and live end-to-end sweeps:**
- Added `experiments/exp02_cpu_policy_sweep/prompt_router_ablation.py` to compare MiniLM/mpnet encoders, MLP sizes, and prompt+`z` ensembles, with `delta` selected on a held-out validation split.
- Added `experiments/exp02_cpu_policy_sweep/task_latent_router.py` and confirmed the PorTAL task latent `z` predicts per-task cheap-is-good-enough with 78.6% LOO accuracy.
- Extended `experiments/exp01_cpu_smoke/run.py` with `--task-encoder`, `--oracle-tasks`, and validation-based floor tuning.
- Extended `DatasetConfig` and `run.py` to support explicit `train/val/test_per_task` counts.
- Ran live end-to-end on 14 tasks with 30/10/10 splits:
  - **Oracle task labels**: val-tuned floor (1.1) saves 17.1% at a 3.6 pp accuracy drop — right at the kill-test boundary, confirming the dispatch policy works when the skill is known.
  - **mpnet task classifier (75% test accuracy)**: val-tuned floor (1.2) saves 30.7% but drops 10.7 pp, because the 0.6B refit collapses when the wrong task LoRA is materialized.
  - Task classifier scales from 75% accuracy (30 ex/task) to 80% (200 ex/task), but the 0.6B refit remains the bottleneck for a fully task-agnostic pipeline.
- Added `experiments/exp01_cpu_smoke/report.md` and updated `experiments/exp02_cpu_policy_sweep/report.md` with ablation tables and bottleneck analysis.
- Updated `README.md` status to reflect the CPU kill-test result and the task-classifier / cheap-base bottleneck.

**Next:** GPU-scale Phase 1/2 — Qwen3 1.7B/4B/8B with 300/100/100 splits, vLLM latency/cost oracle, and independent-LoRA baseline. Prioritize (a) known-skill dispatch on 1.7B/4B/8B and (b) task-classifier robustness with a larger cheap base.
