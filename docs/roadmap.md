# portal-dispatch → Production Inference Router: Productization & Engineering Roadmap

*Prepared 2026-07-21. Based on `portal_dispatch_plan.md`, `experiments/exp01_cpu_smoke/report.md`, and `experiments/exp02_cpu_policy_sweep/report.md`.*

---

## 0. Where we are (grounding facts)

- **Dispatch works when the task is known.** Score-router floor=1.2 clears the kill test on the 208-example bundle: 67.3% acc (+0.5 pp vs always-capable), 15.6% savings; LOO generalization 70.7% / 19.6%. Oracle headroom is 75.5% acc / 28.4% savings, so real routing headroom exists.
- **Task-agnostic mode is not viable yet.** With a 75%-accurate task classifier, the 0.6B refit collapses (−10.7 pp at 30.7% savings). The bottleneck is task-classifier error × cheap-base fragility, not the base router.
- **Prompt-only routing is safe but conservative** (13.5% savings, −3.3 pp at δ=0; 1–5% savings at ≤2.5 pp in the ablation). The task latent `z` predicts per-task cheap-is-good-enough at 78.6% LOO — a genuine zero-shot routing signal nobody else has.
- **Materialization overhead is a non-issue** (~50–60 ms, <1% per forward), so the runtime-LoRA story is technically credible.
- **What exists as code:** typed `src/portal_dispatch/{routing,dispatch,runtime,eval,data,tracing,refit}` modules, HF backend, vLLM backend stub, FastAPI OpenAI-compatible gateway stub, Oracle builder, kill-criteria checker, CI + Docker skeletons, two reproducible experiments.

**Strategic implication:** the near-term product is a **known-skill router** (caller supplies or route implies the task), not a magic universal router. That is also exactly how real customers integrate (per-endpoint/per-route policies), so it is a feature, not a concession.

---

## 1. MVP definition

### 1.1 The smallest shippable version

**"Policy-driven OpenAI-compatible router for named routes."** A single self-hostable binary/container that:

1. Exposes `/v1/chat/completions` and `/v1/completions`, drop-in for the OpenAI SDK (change `base_url`, set `model: "auto"` or `model: "route/<name>"`).
2. Routes each request among an **approved candidate set** defined per route in YAML: commercial API models (via a LiteLLM-style adapter layer) **and/or** internal LoRA-served bases (vLLM + PorTAL materialization).
3. Applies a config-not-code **policy**: `quality_bar` (production/draft/numeric), `max_latency_ms`, `max_cost_per_1k`, `allowed_models`, `fallback_order`, `escalation` (cascade on low confidence).
4. Emits a **JSONL/OTLP decision trace per request** (candidates, scores, chosen model, reason, fallback, cost, latency, cache/batch decisions) and a spend/quality dashboard.
5. Ships with an **offline eval harness**: point it at your prompts+graders (or thumbs-up labels), it builds the private oracle benchmark, trains the prompt router, and reports the Pareto frontier before you flip routing on. "Shadow mode" (log the decision, don't act on it) is the default first week.

**Explicitly deferred from MVP:** task-agnostic auto-classification in the hot path (kill-test failure), multi-tenant control plane, autoscaling, cross-family PorTAL transfer, marketplace-style provider aggregation.

### 1.2 Differentiation vs Ramp Router

| Axis | Ramp Router | portal-dispatch MVP |
|---|---|---|
| Candidate pool | Frontier/commercial API models only | Commercial APIs **plus your own LoRA-served open models** — the cheap tier can be a model you own at marginal GPU cost |
| Quality grounding | Ramp's internal benchmark | **Your** private oracle built from your traffic + graders, versioned and re-runnable |
| Adapter story | None | Runtime LoRA materialization from a shared task latent — one artifact serves N tasks × M base sizes; new task = JSONL + refit, no per-task deployment |
| Deployment | Ramp's cloud | Self-hosted / VPC (data never leaves), OSS core |
| Routing signal | Proprietary | Open, inspectable routers (score / prompt-embedding / task-latent `z`), retrainable on your labels |
| Zero-shot task routing | — | `z`-latent predicts base suitability for *unseen tasks* (78.6% LOO) — unique research moat |

One-line positioning: **"Ramp Router routes across models you rent; portal-dispatch also routes into models you own — and materializes the right skill on them at runtime."**

---

## 2. Architecture

### 2.1 Current components → production mapping

```
                        ┌──────────────── control plane ────────────────┐
                        │ route/policy registry (YAML→DB)               │
                        │ model/base/alignment registry (allowlist)     │
                        │ oracle benchmark store + router training jobs │
                        │ trace warehouse (ClickHouse) + dashboards     │
                        └───────────────┬───────────────────────────────┘
                                        │ policies, router weights, allowlists
client ── OpenAI SDK ──► Gateway ──► Policy Engine ──► Router(s) ──► Backend pool
                          │            │  (floor / cascade /          ├─ vLLM (LoRA hot-swap, adapter LRU)
                          │            │   constrained min-cost)      ├─ HF reference backend
                          │            │                              ├─ LiteLLM adapter → OpenAI/Anthropic/…
                          │  prompt/prefix cache, request coalescing  └─ mock backend (tests)
                          └────────► Trace Journal (JSONL → Kafka/Redpanda → ClickHouse)
```

- **Gateway** (`runtime.Gateway`, exists as FastAPI stub): harden into the data-plane entry — authn (API keys), per-route config resolution, streaming, retries/fallbacks, prompt-cache lookup, small-batch coalescing window.
- **Policy engine** (`dispatch`): already has floor/cascade/constraint policies; extend to full constrained min-cost solve over the candidate set with per-route quality bars and fallback chains; hot-reload from control plane.
- **Routers** (`routing`): production hot path uses `PromptEmbeddingRouter` (no candidate forward pass) + `TaskClassifier` only where the route allows it; `ScoreRouter` runs offline/shadow to generate labels; `LatentRouter` prices new tasks before any traffic.
- **Backends** (`runtime`): finish `VLLMBackend` (PEFT export + `LoRARequest` hot-swap + (base,task) adapter LRU — kill criterion 3: swap <10% of median latency); add a **commercial-API backend** via LiteLLM so every candidate is uniform behind the `Backend` seam.
- **Oracle** (`eval.Oracle`): becomes the control-plane batch job — continuously re-evaluate registered candidates on the versioned private benchmark, refresh router weights, publish to gateway.
- **Trace journal** (`tracing`): JSONL locally → Kafka/Redpanda → ClickHouse; this table is simultaneously the observability store *and* the router-retraining dataset (Ramp's key loop).
- **Caches/queues:** prompt/prefix cache (vLLM native prefix caching + gateway-level exact/semantic response cache in Redis), adapter LRU per GPU, batching window per backend, retry queue for fallbacks.
- **Reuse from longhaul/TTR:** completion-window latency tiers (asap/priority/standard/flex) map directly to policy presets; durable trace journal mirrors the TTR journal pattern; the `Backend` seam is the same seam abstraction.

### 2.2 Data plane vs control plane split

- **Data plane** (latency-critical, stateless, horizontally scalable): gateway + policy engine + routers + backends + caches. No DB writes on the hot path except async trace emit.
- **Control plane** (Postgres/SQLite + object store): route/policy CRUD, model registry, oracle runs, router training, artifact versioning (git SHA + config hash + dataset revision, already specced), dashboard API.

---

## 3. Phase-by-phase milestones, kill criteria, effort

Assume a core team of **2 engineers** (1 infra/serving, 1 ML/routing) + fractional PM/design. "w" = calendar weeks.

| Phase | Scope | Duration / effort | Kill / gate criteria |
|---|---|---|---|
| **P0 — GPU validation** (the decisive experiment, already planned as S3–S4) | Qwen3 1.7B/4B/8B refits, 300/100/100 splits, independent-LoRA baseline (B1), router sweeps (B2–B4), vLLM oracle + swap benchmark. Budget ~$300 GPU. | 3–4 w, 1.5 eng | G-A: refit non-random (≥10 pp over random after epoch 1). G-B: any policy ≥15% savings at ≤3 pp drop on ≥100-ex tests — else no product, publish negative result. G-C: vLLM adapter swap <10% median latency — else invest in caching before proceeding. |
| **P1 — Router MVP (commercial-models-only mode)** | Harden gateway (streaming, keys, retries, fallbacks), LiteLLM backend, YAML policies, shadow mode, JSONL→ClickHouse traces, offline eval harness + private-oracle builder, basic Grafana dashboard. *This works even if P0 partially fails* — it's a Ramp-class router without the LoRA tier. | 4–6 w, 2 eng | Ship gate: route 100% of an internal/design-partner workload in shadow mode for 1 week with 0 correctness regressions in traces; live-routing shows ≥15% cost reduction at quality bar on that workload. Kill: if no design partner shows ≥10% addressable savings in shadow mode, reposition as pure observability/eval tool. |
| **P2 — LoRA tier integration** | vLLM backend GA (hot-swap + LRU + prefix cache), PorTAL artifact registry, `add-a-task` / `add-a-base` flows productized, known-skill routes (caller supplies task or route implies it), cost model incl. self-hosted GPU $/token. | 4–5 w, 2 eng | Gate: on ≥1 real workload, a route mixing {internal 1.7B/4B LoRA, commercial fallback} beats commercial-only routing by ≥20% cost at equal quality bar. Kill: if self-hosted tier never wins on cost-at-quality vs cheap commercial models (e.g. Haiku/Flash-class), drop LoRA tier to "research mode" and keep P1 product. |
| **P3 — Task-agnostic + continuous learning** | Stronger task classifier (fine-tuned encoder, retrieval-assisted), abstain-to-capable default (classifier uncertainty ⇒ never route cheap — fixes the exp01 failure mode by construction), online label collection from traces + graders, scheduled oracle refresh + router retraining, canary/rollback for router weights. | 5–6 w, 2 eng | Gate: task-agnostic route within 2 pp of known-skill route on the same traffic. Kill for auto-mode only (product keeps known-skill mode): classifier+abstain still loses >3 pp at <10% savings. |
| **P4 — Multi-tenant / GA hardening** | Control-plane UI, per-tenant keys/quotas/spend limits, SOC2-track logging, autoscaling (K8s + vLLM pools), SLA/alerting, semantic cache. | 6–8 w, 3 eng (add 1) | Business gate: ≥3 paying design partners or ≥1 large committed deployment before spending here. |

Cumulative to a sellable P2 product: **~4 months, 2 engineers** (plus ~$1–2k GPU). P0 is cheap and decisive — do it first, in parallel with P1 scaffolding since P1 doesn't depend on P0's outcome.

---

## 4. Infrastructure, integrations, observability

**Serving**
- **vLLM** (primary): `enable_lora`, `max_loras≥8`, prefix caching, multi-model on one A100/H100 for the internal tier. TGI as a documented alternative, not a build target.
- **LiteLLM** (library, not proxy) as the uniform commercial-API adapter behind the `Backend` seam — gets OpenAI/Anthropic/Gemini/Bedrock/Together for free, with cost tables.
- **OpenRouter** as an optional *upstream backend* (one candidate among many), not a competitor to integrate deeply — useful for long-tail model access in evals.

**Data & observability**
- Traces: structured events → **Kafka/Redpanda → ClickHouse** (Ramp's own stack; also what the plan's §1.2.5 calls for). SQLite/JSONL single-node mode for OSS self-hosters.
- **Langfuse** (or OTel GenAI conventions + Grafana Tempo) for per-request LLM traces/spans that customers already understand; emit both.
- **Grafana + Prometheus**: p50/p95 per stage (classify, route, materialize, swap, forward), cache hit rate, %-routed per model, spend per route, quality-bar violation alerts.
- **Eval/oracle**: our own `eval.Oracle` + optional graders (exact-match, LLM-judge via a pinned judge model); results versioned in the control plane; every router weight traceable to (oracle run, git SHA, dataset revision).

**Control plane**
- Postgres (multi-tenant) / SQLite (single-node), S3-compatible object store for artifacts, HF Hub for public PorTAL artifacts/routers.
- CI: existing GH Actions (ruff, mypy strict, pytest, ≤10-min CPU smoke) + nightly full CPU repro + weekly GPU regression on a spot instance.

**Deployment targets:** Docker Compose (single node, OSS default) → Helm chart (K8s, vLLM GPU pool + CPU gateway pool) at P4.

---

## 5. Open questions & risks

| # | Question / risk | Why it matters | Mitigation |
|---|---|---|---|
| R1 | **Does routing headroom survive at GPU scale?** CPU headroom (oracle 28.4%) may shrink if 4B/8B error sets converge. | Whole thesis | P0 measures oracle bound first (B2) before any product spend; pivot to cascade-only or add a cheaper tier if headroom <15%. |
| R2 | **Task-classifier fragility** (proven failure at 75% acc). | Blocks auto mode | Ship known-skill first; in auto mode, *abstain-to-capable* on low classifier confidence so mistakes cost money, never quality; report both modes honestly. |
| R3 | **Self-hosted tier may lose to cheap commercial models** (Flash/Haiku are very cheap). | P2 value prop | Cost model must include real GPU amortization; target segments where self-hosting wins structurally: data-residency/VPC, high-volume narrow tasks, fine-tuned-quality needs. Kill LoRA tier per P2 gate if it never wins. |
| R4 | **Quality measurement is the product.** Customers without graders can't set a quality bar. | Adoption | Bundle grader templates (EM/F1, rubric LLM-judge, thumbs-feedback ingestion); shadow mode + oracle report is the free on-ramp. |
| R5 | vLLM LoRA swap overhead / max_loras limits erode savings. | P2 latency | Kill criterion 3 measured in P0; adapter LRU + pre-materialization; cap concurrent tasks per GPU. |
| R6 | **Ramp/OpenRouter/Cloudflare move down-market** into self-hosted routing. | Moat | Our moat is the adapter layer (one artifact → N tasks × M bases) and open, retrainable routers; publish the research (existing paper plan) to own the category narrative. |
| R7 | Small routing-label datasets → noisy routers (proven on CPU). | Router quality | 300/100/100 splits, bootstrap CIs, 3 seeds (already planned); continuous label refresh from production traces. |
| R8 | Who is the buyer — platform team or app team? | GTM | Design-partner interviews in P1; MVP config UX (YAML per route) targets platform/infra teams first. |
| R9 | License/compliance of routing customer data through our judge models. | Enterprise sales | Self-hosted-first, judge model pluggable/self-hosted, no data leaves VPC. |

---

## 6. Go-to-market & positioning

**Category:** "Cost-aware quality-barred inference router" — the control loop, not the marketplace.

- **vs Ramp Router:** Ramp is a hosted endpoint over commercial frontier models, closed routing logic, Ramp's benchmark. We are **self-hosted/VPC, open routing logic, your benchmark, and we can route into models you own** — including runtime-materialized task adapters. Ramp validated the market (2.75T tok/mo, 30%+ savings); we sell the same outcome to teams that can't or won't send traffic through a third party.
- **vs OpenRouter:** OpenRouter is a *marketplace/aggregator* — it gives you access and uniform billing, but you pick the model. We sit **above** it (OpenRouter can be one of our backends) and make the per-request choice against your quality bar. Not competitive; complementary.
- **vs Cloudflare AI Gateway / generic gateways (Portkey, Kong AI):** those are proxies with caching, rate-limiting, and fallback *rules*. They don't measure quality or train routers on your outcomes. Our wedge is the **quality-grounded decision loop** (oracle → router → policy → trace → retrain).
- **vs academic/OSS model routers (RouteLLM, FrugalGPT, Martian, NotDiamond):** closest competitors on the routing brain. Differentiators: (a) the adapter-materialization tier (unique), (b) production gateway + policy + observability as one artifact, (c) the task-latent zero-shot routing signal, (d) reproducible open benchmark methodology.

**Motion:** OSS core (Apache-2.0) + paper/blog launch (the existing §10 publication plan doubles as content marketing) → design partners from high-volume LLM teams (agents, support automation, doc processing) → paid tiers: hosted control plane, enterprise features (SSO, audit, SLA), and "managed adapter tier" (we train/refit task LoRAs from your JSONL).

**Pricing anchor:** % of verified savings or per-routed-token platform fee; shadow-mode savings report is the free funnel ("here's the 23% you're leaving on the table").

---

## 7. Concrete next 3 actions for the parent session

1. **Run the P0/G-B decisive experiment now** (it gates everything): provision 1× A100-80GB spot (~$2/hr, ≤$300 budget), refit/align Qwen3 1.7B/4B/8B with 300/100/100 splits, run B1 (independent-LoRA baseline) and B2 (oracle headroom) *first*, then the router sweeps, and produce `experiments/exp03_gpu_scale/report.md` with the machine-checked kill-criteria verdict (≥15% savings at ≤3 pp drop; refit non-random; vLLM swap <10% latency).
2. **Finish the P1 data-plane skeleton in parallel** (independent of P0's outcome): wire `LiteLLMBackend` behind the `Backend` seam, harden the gateway (streaming, API keys, fallback chains, per-route YAML policies with quality_bar/max_latency/allowed_models), and land shadow-mode routing + full JSONL decision traces so any workload can be measured without risk. Ship as `portal-dispatch serve --config routes.yaml`.
3. **Recruit 1–2 design partners and build their private oracle**: pick a real high-volume workload (internal agent traffic — e.g. longhaul's inference plane is a natural first tenant — or an external partner), capture 1–2k prompts with graders, run the shadow-mode savings report, and use it to validate both the quality-bar UX and the R3/R8 questions (does the self-hosted tier win; who is the buyer).

---

*Appendix — reused assets:* score/prompt/z routers and floor/cascade policies (validated), Oracle builder, kill-criteria checker, trace journal, HF backend, CI/Docker scaffolds, and the S1–S5 execution log all carry forward unchanged into P0–P2.
