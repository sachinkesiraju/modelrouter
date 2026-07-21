# portal-dispatch

*Route each query to the cheapest base model that can do the job — materializing the task adapter at runtime from a shared PorTAL latent.*

`portal-dispatch` is a research prototype of a cost-aware inference router. Instead of picking among fixed API models, it (1) predicts which registered base model can answer a query correctly, (2) picks the cheapest one that clears a quality bar, and (3) materializes the task-specific LoRA adapter for that base on demand (~50 ms) from a shared [PorTAL](https://pypi.org/project/portallib/) hypernetwork latent.

## How it works

```
prompt ──► TaskClassifier ──► Router (score / prompt-embedding / task-latent z)
                                   │
                                   ▼
                         DispatchPolicy (floor / cascade)
                                   │
                       cheapest capable base chosen
                                   │
                                   ▼
              PortalModel.generate(task) ──► PortalInjector.activate
                                   │
                                   ▼
                            base forward pass
```

- **Routers** (`portal_dispatch.routing`): `ScoreRouter` (per-base score-distribution features; offline/oracle-grade), `PromptEmbeddingRouter` (MiniLM prompt embeddings; production path — no candidate forward pass), `LatentRouter` (predicts per-task base suitability from the PorTAL task latent `z`; zero-shot task routing), `TaskClassifier`.
- **Policies** (`portal_dispatch.dispatch`): `FloorPolicy` (pre-route: cheapest base with `p * floor >= max p`), `CascadePolicy` (run cheap, escalate on low confidence).
- **Runtime** (`portal_dispatch.runtime`): `HFBackend` loads a base once and hot-swaps task LoRAs via `PortalInjector` with an adapter cache.
- **Gateway** (`portal_dispatch.gateway`): OpenAI-compatible `/v1/chat/completions` FastAPI app with JSONL decision traces.
- **Eval** (`portal_dispatch.eval`): policy stats, bootstrap CIs, Pareto plots, and a machine-checkable kill-criteria gate (default: ≥15% savings at ≤3 pp accuracy drop).

## Production gateway

Beyond the research pipeline, `portal_dispatch.serve` is a production-oriented OpenAI-compatible gateway with config-not-code routing:

```bash
portal-dispatch serve --config configs/routes.example.yaml
```

- **Per-route YAML policies** — candidate models, floor, quality/fallback order, default model.
- **Commercial-model tier** via LiteLLM (`portal_dispatch.backends.LiteLLMBackend`): OpenAI, Anthropic, Together, Gemini, ... with real $/token cost from price tables. Local PorTAL-served bases plug into the same seam (`PortalLocalBackend`).
- **Shadow mode** — log every routing decision but always serve the default model; the zero-risk on-ramp: measure would-be savings before flipping live.
- **Fallback chains** — on backend failure the request walks `fallback_order`; failures are recorded in the trace.
- **API keys** — optional bearer-token auth.
- **Abstain-to-capable** — in task-agnostic mode, low task-classifier confidence routes to the default model, so classifier mistakes cost money, never quality.
- **Continuous learning** (`portal_dispatch.learning`) — retrain the prompt router from labeled production traces into versioned, canary-able artifacts (`retrain_from_traces`).

Every request emits a JSONL decision trace: route, decision + reason + scores, served model, fallbacks tried, cost (USD), latency, token counts.

Validated live against Together AI serverless models (Qwen2.5-7B as cheap, Llama-3.3-70B as capable): shadow mode logs `decision=cheap, served=capable`; live mode serves the cheap model at ~3.4× lower measured cost per request.

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or your CUDA build
pip install -e ".[serve,plots,dev]"
```

## Reproduce the CPU validation

Everything below runs on a CPU box (~8 GB RAM). Artifacts come from the HF Hub (`RampPublic/portal-qwen3-1.7b`, `RampPublic/portallib-tasks`).

```bash
# 1. Refit the published Qwen3-1.7B artifact to Qwen3-0.6B (the cheap tier)
python experiments/exp01_cpu_refit/run.py

# 2. Score val+test rows on both bases with the correct task LoRA
python experiments/exp02_policy_sweep/build_bundle.py \
  --cheap-artifact experiments/exp01_cpu_refit/artifacts/refit-qwen3-0.6b

# 3. Train routers on val, sweep dispatch policies on test, check kill criteria
python experiments/exp02_policy_sweep/run.py

# 4. Live end-to-end with a task classifier (measures the task-classifier tax)
python experiments/exp03_live_e2e/run.py \
  --cheap-artifact experiments/exp01_cpu_refit/artifacts/refit-qwen3-0.6b
```

Results, reports, and the Pareto plot land in each experiment's `results/` directory. See `experiments/*/report.md` for the validated numbers and discussion.

## Status and findings (CPU validation, 2026-07-21)

All numbers below were produced by the three experiments in this repo, end to end from public artifacts, on an 8 GB / 2-core CPU box:

| Result | Value | Where |
|---|---|---|
| 0.6B refit beats pre-refit baseline | +10.0 pp macro (48.1% → 58.1%) | `exp01_cpu_refit/report.md` |
| **Kill test: ≥15% savings at ≤3 pp drop** | **PASSED** — val-tuned score-floor: 19.0% savings, −2.9 pp (CI: 16.2–22.6% savings) | `exp02_policy_sweep/report.md` |
| Oracle routing headroom | +5.7 pp accuracy at 40% savings | `exp02_policy_sweep/report.md` |
| Task-latent `z` zero-shot task routing | 78.6% leave-one-task-out | `exp02_policy_sweep/report.md` |
| Task classifier (live e2e) | 81.2% accuracy; ~3 pp tax on this slice | `exp03_live_e2e/report.md` |

Dispatch works when the task is known; prompt-only routing is safe but conservative; known-skill deployment is the near-term sweet spot. The decisive next experiment is GPU-scale (Qwen3 1.7B/4B/8B + vLLM), tracked in `docs/roadmap.md`.

## License

Apache-2.0.
