# exp02_vllm_bench: vLLM adapter hot-swap benchmark (Modal, A10G)

The serving-substrate gate from `docs/roadmap.md` P0: is swapping PorTAL task LoRAs on
vLLM cheap enough for production (<10% of request latency)?

## Setup

- PorTAL task LoRAs from `RampPublic/portal-qwen3-1.7b` exported to standard PEFT adapter
  dirs via `PortalModel.get_peft_model` (rank 8, q/v_proj) — ~0.5 s/task, one-time.
- vLLM 0.25.1 serving Qwen3-1.7B (`enable_lora`, `max_loras=4`), A10G, bf16,
  FlashAttention. 8 task adapters staged on container-local disk (as production NVMe
  would); greedy 64-token completions.

## Results (median over tasks)

| Measurement | Value |
|---|---:|
| Base request (no LoRA) | 573 ms |
| LoRA request, adapter warm | 704 ms |
| LoRA request, first use (cold swap) | 719 ms |
| **Adapter swap cost** | **15.4 ms = 2.2% of request** |
| **Gate: swap < 10% of latency** | **PASSED** |

- Swap overhead is negligible — consistent with the ~50 ms HF-backend materialization
  measured in the CPU-scale pilots. Task LoRA hot-swap is operationally free.
- Caveat worth knowing: vLLM's LoRA execution path (punica kernels) adds a steady ~23%
  latency vs the bare base regardless of swapping. That is a serving-stack constant, not a
  PorTAL cost, and shrinks with batching; it should be priced into the internal tier's
  cost model.
- Reading adapters straight off a network volume instead of local disk pushed the first
  swap to ~595 ms — adapters must live on local NVMe (or be pre-fetched) in production.

## Load benchmark: streaming + batched concurrent throughput (`load_bench`)

Per ladder model, against a live OpenAI-compatible `vllm serve` on the same A10G:
sequential streamed requests (TTFT, inter-token latency) and concurrent load at
1/4/16/32 simultaneous clients (4 requests per client, 128 output tokens each,
`ignore_eos` for stable token counts). Results in `results/load_bench.json`.

| Model | TTFT p50 | Inter-token p50 | tok/s @ c=1 | tok/s @ c=16 | **peak tok/s** | p95 latency @ c=16 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 11.6 ms | 3.7 ms | 995 | 8680 | **8707** | 935 ms |
| Qwen3-1.7B | 14.6 ms | 8.9 ms | 439 | 4690 | **4825** | 1738 ms |
| Qwen3-4B | 26.3 ms | 18.8 ms | 204 | 2345 | **2345** | 3482 ms |

- Throughput saturates around 16 concurrent clients on all three models (c=32 adds
  little throughput and mostly queueing latency).
- Batched throughput scales ~8-10x from c=1 to saturation; the ratio between models is
  much flatter than parameter count (3.7x from 0.6B to 4B, not 6.7x). These peak tok/s
  numbers are the input to the GPU-amortized cost model
  (`modelrouter.costs.GpuCostModel`) used to re-price the exp01 sweep.
- Streaming works end-to-end through the OpenAI-compatible path with single-digit to
  low-double-digit ms TTFT at c=1 on all tiers.
- Caveat: measured on the bare bases (no LoRA). The LoRA execution path adds the ~23%
  single-request overhead measured above; it shrinks with batching but was not
  re-measured under load.

## Cost

Two A10G runs ≈ 0.4 GPU-hours ≈ **$1.30**; the 3-model load bench ≈ 0.5 A10G-hours ≈ **$0.55**.

## Reproduce

```bash
modal run experiments/exp02_vllm_bench/modal_app.py::bench
# results land in /vol/results/vllm_bench.json on the "modelrouter" Modal volume:
modal volume get modelrouter results/vllm_bench.json \
  experiments/exp02_vllm_bench/results/vllm_bench.json

# streaming + concurrent load bench for the full ladder (writes results/load_bench.json):
modal run experiments/exp02_vllm_bench/modal_app.py::load_bench_ladder
```
