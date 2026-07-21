# modelrouter

[![CI](https://github.com/sachinkesiraju/modelrouter/actions/workflows/ci.yml/badge.svg)](https://github.com/sachinkesiraju/modelrouter/actions/workflows/ci.yml)

*Route each query to the cheapest model that can do the job. Load the task-specific LoRA adapter for that model at request time, generated from a single PorTAL artifact.*

Most requests don't need the largest model. A router trained on per-model correctness can send each request to the cheapest model that will get it right, and apply the task-specific LoRA adapter at runtime.

Inspired by [Ramp Router](https://ramp.com/router) and Ramp's work on [cost-efficient LLM routing](https://builders.ramp.com/post/thompson-sampling-model-routing).

![How modelrouter works](docs/assets/how-it-works.svg)

## Headline result

> **Learned routing cut inference cost by 58% while giving up only 2.8 accuracy points versus always running the largest model in the ladder (Qwen3-4B). A prompt-only router, which decides before any model runs, still cut cost by 47% at a 1.1 point drop.**

The same router applied to a commercial frontier ladder (gpt-5.4-nano / gpt-5.4-mini / gpt-5.6, real per-request API costs) cut spend by 40.7% at a 3.6 point drop versus always calling gpt-5.6 ([exp03](experiments/exp03_commercial_api/report.md)).

Measured on 14 tasks / 1,230 held-out rows (local ladder: GPU-trained refits on Modal A100/A10G; [exp01](experiments/exp01_gpu_scale/report.md), [exp02](experiments/exp02_vllm_bench/report.md)):

| Result | Value |
|---|---|
| **Cheapest-capable routing (3-tier ladder)** | **58.4% cost savings at −2.8 pp accuracy** (CI: 56.9–59.6%) |
| Near-zero-loss operating point | 44.7% savings at −0.2 pp |
| **Prompt-only router (usable live, pre-inference)** | **47.0% savings at −1.1 pp** (1.7B vs 4B) |
| Routing headroom (oracle, 3-tier) | +12.3 pp accuracy *above* always-largest at 59.2% savings |
| Task-latent `z` tier prediction for unseen tasks | 100% leave-one-task-out |
| vLLM task-adapter hot-swap overhead | 15.4 ms = 2.2% of a request |
| **Commercial ladder (gpt-5.4-nano/mini → gpt-5.6, real $)** | **40.7% spend cut at −3.6 pp** (CI: 38.4–43.0%) |
| Commercial oracle headroom | +6.0 pp accuracy *above* always-gpt-5.6 at 86% savings |

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
```

Full reproduction (Modal GPU runs, OpenAI scoring) is documented in each experiment's report under [`experiments/`](experiments/). See the [roadmap](docs/roadmap.md) for the productization plan.

## Limitations

A validated research artifact plus a working single-node router, not a production service:

- **Benchmark scope**: 14 multiple-choice tasks with programmatic graders; free-form generation quality is not measured.
- **Model scope**: local GPU ladder tops out at Qwen3-4B; commercial routing validated on one provider (OpenAI, gpt-5.4-nano/mini/gpt-5.6) plus a live Together AI gateway test.
- **Cost model**: local-tier savings use parameter-proportional costs; GPU amortization not modeled.
- **Serving**: single-request benchmarks only; vLLM's LoRA path adds a steady ~23% latency vs the bare base (shrinks with batching). No streaming, batching, or load testing.
- **Operations**: no multi-tenant control plane, quotas, health checks, or K8s packaging (see the [roadmap](docs/roadmap.md)).
- **Task-agnostic mode**: lightly validated; guarded by abstain-to-capable.

## License

Apache-2.0.
