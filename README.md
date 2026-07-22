# modelrouter

Route each query to the cheapest model that can do the job. Load the task-specific LoRA adapter for that model at request time, generated from a single [PorTAL](https://github.com/ramp-public/portallib) artifact.

Most requests don't need the largest model. A router trained on per-model correctness can send each request to the cheapest model that will get it right, and apply the task-specific LoRA adapter at runtime.

Inspired by [Ramp Router](https://ramp.com/router) and Ramp's work on [cost-efficient LLM routing](https://builders.ramp.com/post/thompson-sampling-model-routing).

![How modelrouter works](docs/assets/how-it-works.svg)

## Headline result

> **Learned routing cut real OpenAI spend by 41% (at a 3.6 point drop) versus always calling gpt-5.6, and cut cost 45% with essentially no accuracy loss (0.2 points) on a self-hosted Qwen3 ladder. A prompt-only router, deciding before any model runs, still saved 47%.**

The self-hosted ladder is Qwen3 0.6B / 1.7B / 4B; allowing a 2.8 point drop raises its savings to 58%, or 51.5% when costs are GPU-amortized from measured tokens/sec on an A10G. The OpenAI ladder is gpt-5.4-nano / gpt-5.4-mini / gpt-5.6, priced at real per-request API cost. The same router also cut spend by 40.7% at a 3.3 point drop on free-form GSM8K over a Together AI ladder.

Measured on 14 tasks / 1,230 held-out rows:

| Result | Value |
|---|---|
| Cheapest-capable routing: send each request to the cheapest of 0.6B/1.7B/4B predicted to answer correctly | 58.4% cost savings at −2.8 pp accuracy (CI: 56.9–59.6%) |
| Same policy, re-priced with GPU-amortized costs (measured tok/s on A10G) | 51.5% savings at −2.8 pp (CI: 50.2–52.5%) |
| Same policy tuned for near-zero quality loss | 44.7% savings at −0.2 pp |
| Prompt-only router: decides from the prompt alone, before any model runs | 47.0% savings at −1.1 pp (1.7B vs 4B) |
| Oracle upper bound: a perfect router picking the best tier per request | +12.3 pp accuracy *above* always-largest at 59.2% savings |
| Predicting the right tier for a never-seen task from its PorTAL latent `z` | 100% leave-one-task-out accuracy |
| Cost of swapping a task LoRA adapter on a live vLLM server | 15.4 ms = 2.2% of a request |
| Same routing on an OpenAI ladder (gpt-5.4-nano/mini → gpt-5.6) at real API prices | 40.7% spend cut at −3.6 pp (CI: 38.4–43.0%) |
| Oracle upper bound on the OpenAI ladder | +6.0 pp accuracy *above* always-gpt-5.6 at 86% savings |
| Free-form generation (GSM8K exact match) on a Together AI ladder (7B/20B/70B) at real API prices | 40.7% spend cut at −3.3 pp (CI: 34.9–45.6%) |
| Task-agnostic mode: classify the task from the prompt, abstain to the largest model when unsure | matches known-task routing at zero accuracy tax |

Full reproduction (Modal GPU runs, OpenAI scoring) is documented in each experiment's report under [`experiments`](experiments/).

## Architecture

```
client (OpenAI SDK)
   │
   ▼
Gateway (/v1/chat/completions)        modelrouter.gateway
   │  task = caller-supplied or TaskClassifier(prompt)
   ▼
Router                                 modelrouter.routing
   │  p_correct per registered model (prompt embeddings on the hot path)
   ▼
DispatchPolicy                         modelrouter.dispatch
   │  cheapest model with p * floor >= max p   (or cascade + escalation)
   ▼
Backend                                modelrouter.runtime
   │  PortalModel.generate(task) → PortalInjector.activate → forward
   ▼
TraceJournal                           modelrouter.tracing
      JSONL: prompt, task, candidates, chosen, reason, scores
```

- **Routers** (`routing`): trained offline on per-(query, model) correctness. Prompt-embedding router on the hot path; score-distribution and task-latent `z` routers for oracle labels and unseen tasks.
- **Policies** (`dispatch`): one knob — floor (cheapest model within a quality floor) or cascade (run cheap, escalate on low confidence).
- **Backends** (`runtime`, `backends`): one seam hiding HF with PorTAL LoRA hot-swap and a LiteLLM commercial tier with real $/token costs.
- **Gateway** (`serve`): OpenAI-compatible server — YAML route policies, shadow mode, fallbacks, API keys, JSONL traces, retraining from traces.
- **Eval** (`eval`): policy stats, bootstrap CIs, and a machine-checkable quality/cost acceptance gate.

## Quickstart

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or your CUDA build
pip install -e ".[serve,plots,dev]"

# serve the gateway
modelrouter serve --config configs/routes.example.yaml

# reproduce the headline table from the committed GPU score bundles (no GPU needed)
python experiments/exp01_gpu_scale/run_sweep.py

# or run the gateway in Docker (health checks at /healthz and /readyz)
docker compose up
```

## Limitations

A validated research artifact plus a working single-node router, not a production service:

- Local ladder tops out at Qwen3-4B.
- Not production-hardened: no multi-tenant control plane or K8s packaging (see the [roadmap](docs/roadmap.md)).

## License

Apache-2.0.
