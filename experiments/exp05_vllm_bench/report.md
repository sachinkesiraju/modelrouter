# exp05_vllm_bench: vLLM adapter hot-swap benchmark (Modal, A10G)

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
  measured at CPU scale. Task LoRA hot-swap is operationally free.
- Caveat worth knowing: vLLM's LoRA execution path (punica kernels) adds a steady ~23%
  latency vs the bare base regardless of swapping. That is a serving-stack constant, not a
  PorTAL cost, and shrinks with batching; it should be priced into the internal tier's
  cost model.
- Reading adapters straight off a network volume instead of local disk pushed the first
  swap to ~595 ms — adapters must live on local NVMe (or be pre-fetched) in production.

## Cost

Two A10G runs ≈ 0.4 GPU-hours ≈ **$1.30**.

## Reproduce

```bash
modal run experiments/exp05_vllm_bench/modal_app.py::bench
# results land in /vol/results/vllm_bench.json on the "modelrouter" Modal volume:
modal volume get modelrouter results/vllm_bench.json \
  experiments/exp05_vllm_bench/results/vllm_bench.json
```
