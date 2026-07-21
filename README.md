# portal-dispatch

*Route each query to the cheapest model that can do the job — materializing the task adapter at runtime from a shared PorTAL latent.*

Inspired by [Ramp Router](https://ramp.com/router) and Ramp's work on [cost-efficient LLM routing](https://builders.ramp.com/post/thompson-sampling-model-routing).

![How portal-dispatch works](docs/assets/how-it-works.svg)

A cost-aware inference router: predict which model can answer a query, pick the cheapest one that clears a quality bar, and generate the task-specific LoRA for that model on demand from a single [PorTAL](https://pypi.org/project/portallib/) artifact (~15 ms swap on vLLM).

## Results

GPU-scale validation (Modal A100/A10G; Qwen3-0.6B/1.7B/4B ladder, 14 tasks, 1,230 held-out rows — `experiments/exp04_gpu_scale/`, `exp05_vllm_bench/`):

| Result | Value |
|---|---|
| **Kill test: ≥15% savings at ≤3 pp drop** | **PASSED — 58.4% savings at −2.8 pp** (CI: 56.9–59.6%) |
| Zero-drop operating point | 44.7% savings at −0.2 pp |
| **Prompt-only router (live-deployable)** | **47.0% savings at −1.1 pp** (1.7B vs 4B) |
| Oracle headroom (3-tier) | +12.3 pp accuracy over always-capable at 59.2% savings |
| Task-latent `z` tier prediction | 100% leave-one-task-out |
| vLLM adapter hot-swap | **15.4 ms = 2.2% of request — gate PASSED** |

CPU-scale numbers (reproducible on an 8 GB laptop) are in `experiments/exp01–03/*/report.md`.

## Components

- **Routers** (`routing`) — score-distribution, prompt-embedding (live path), and task-latent `z` routers, plus a task classifier.
- **Policies** (`dispatch`) — floor (cheapest model within a quality floor) and cascade (run cheap, escalate on low confidence).
- **Runtime** (`runtime`) — HF backend that hot-swaps PorTAL task LoRAs with an adapter cache.
- **Gateway** (`serve`) — production OpenAI-compatible gateway: per-route YAML policies, LiteLLM commercial tier with real $/token costs, shadow mode, fallback chains, API keys, abstain-to-capable, JSONL decision traces, and router retraining from traces (`learning`). Validated live against Together AI.
- **Eval** (`eval`) — policy stats, bootstrap CIs, Pareto plots, machine-checkable kill criteria.

## Quickstart

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or your CUDA build
pip install -e ".[serve,plots,dev]"

# serve the gateway
portal-dispatch serve --config configs/routes.example.yaml

# reproduce the CPU validation (all public artifacts)
python experiments/exp01_cpu_refit/run.py
python experiments/exp02_policy_sweep/build_bundle.py \
  --cheap-artifact experiments/exp01_cpu_refit/artifacts/refit-qwen3-0.6b
python experiments/exp02_policy_sweep/run.py
```

See `docs/architecture.md` and `docs/roadmap.md` for design and the productization plan.

## Limitations

A validated research artifact plus a working single-node router — not a production service:

- **Benchmark scope** — 14 multiple-choice tasks with programmatic graders; free-form generation quality is not measured.
- **Model scope** — GPU ladder tops out at Qwen3-4B; commercial routing validated on one provider pair (Together AI).
- **Cost model** — local-tier savings use parameter-proportional costs; GPU amortization not modeled.
- **Serving** — single-request benchmarks only; vLLM's LoRA path adds a steady ~23% latency vs the bare base (shrinks with batching). No streaming, batching, or load testing.
- **Operations** — no multi-tenant control plane, quotas, health checks, or K8s packaging (see `docs/roadmap.md`).
- **Task-agnostic mode** — lightly validated; guarded by abstain-to-capable.

## License

Apache-2.0.
