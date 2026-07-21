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

## Status and findings

See `experiments/exp02_policy_sweep/report.md` for the current CPU-scale validation. Headline: dispatch works when the task is known; the score-router floor policy clears the ≥15%-savings / ≤3 pp-drop kill test, prompt-only routing is close behind, and the task latent `z` predicts per-task base suitability leave-one-task-out. The 0.6B refit is fragile to task-classifier mistakes, so known-skill deployment is the near-term sweet spot; the decisive experiment is GPU-scale (Qwen3 1.7B/4B/8B + vLLM), tracked in `docs/roadmap.md`.

## License

Apache-2.0.
