"""exp02: vLLM adapter-swap benchmark on Modal (the serving-substrate gate).

Exports PorTAL task LoRAs to standard PEFT directories, serves the base on vLLM
with LoRA hot-swap, and measures per-task swap overhead vs the no-LoRA baseline.

Gate (roadmap P0): median adapter-swap overhead < 10% of request latency.

  modal run experiments/exp02_vllm_bench/modal_app.py::bench
"""

from __future__ import annotations

import json

import modal

app = modal.App("modelrouter-vllm")

volume = modal.Volume.from_name("modelrouter", create_if_missing=True)
VOL = "/vol"

image = (
    # CUDA devel base: flashinfer JIT-compiles kernels and needs nvcc.
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .pip_install("vllm==0.25.1", "portallib==0.1.2", "peft>=0.14", "hf_transfer")
    .env({"HF_HOME": f"{VOL}/hf", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "1",
          "VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
)


@app.function(image=image, gpu="A10G", volumes={VOL: volume}, timeout=2400)
def bench(model_id: str = "Qwen/Qwen3-1.7B", artifact: str = "RampPublic/portal-qwen3-1.7b",
          n_tasks: int = 8, n_warm: int = 20, max_tokens: int = 64) -> dict:
    import os
    import statistics
    import time

    import torch
    from portallib import PortalModel
    from transformers import AutoModelForCausalLM

    os.makedirs(f"{VOL}/peft", exist_ok=True)
    os.makedirs(f"{VOL}/results", exist_ok=True)

    # 1. Export PorTAL task LoRAs as standard PEFT adapter dirs (CPU, once).
    portal = PortalModel.from_pretrained(artifact)
    tasks = list(portal.config.tasks)[:n_tasks]
    export_ms = {}
    base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
    for task in tasks:
        out = f"{VOL}/peft/{task}"
        if not os.path.exists(f"{out}/adapter_model.safetensors"):
            start = time.perf_counter()
            peft_model = portal.get_peft_model(task, base_model=base, adapter_name="default")
            peft_model.save_pretrained(out)
            export_ms[task] = (time.perf_counter() - start) * 1000
            peft_model.unload()
        print("exported", task, flush=True)
    del base, portal
    volume.commit()

    # 2. Serve on vLLM with LoRA hot-swap.
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    rank = json.load(open(f"{VOL}/peft/{tasks[0]}/adapter_config.json"))["r"]
    llm = LLM(model=model_id, enable_lora=True, max_loras=4, max_lora_rank=rank,
              gpu_memory_utilization=0.85, enforce_eager=False)
    params = SamplingParams(max_tokens=max_tokens, temperature=0)
    prompt = "Question: Is the sky blue on a clear day? Answer:"

    def run(lora: LoRARequest | None) -> float:
        start = time.perf_counter()
        llm.generate([prompt], params, lora_request=lora, use_tqdm=False)
        return (time.perf_counter() - start) * 1000

    # Warm up the engine itself.
    for _ in range(3):
        run(None)
    base_lat = [run(None) for _ in range(n_warm)]

    # Stage adapters on container-local disk: production serving keeps adapters on
    # local NVMe; reading from the network volume would dominate the swap time.
    import shutil

    for task in tasks:
        shutil.copytree(f"{VOL}/peft/{task}", f"/tmp/peft/{task}", dirs_exist_ok=True)

    cold_ms, warm_ms = {}, {}
    for i, task in enumerate(tasks):
        req = LoRARequest(task, i + 1, f"/tmp/peft/{task}")
        cold_ms[task] = run(req)  # includes adapter load from volume
        warm_ms[task] = statistics.median(run(req) for _ in range(5))

    base_med = statistics.median(base_lat)
    warm_med = statistics.median(warm_ms.values())
    cold_med = statistics.median(cold_ms.values())
    result = {
        "model": model_id,
        "tasks": tasks,
        "base_median_ms": base_med,
        "lora_warm_median_ms": warm_med,
        "lora_cold_median_ms": cold_med,
        "warm_overhead_frac": (warm_med - base_med) / base_med,
        "cold_swap_ms": cold_med - warm_med,
        "cold_swap_frac_of_request": (cold_med - warm_med) / warm_med,
        "per_task_cold_ms": cold_ms,
        "per_task_warm_ms": warm_ms,
        "gate_swap_lt_10pct": (cold_med - warm_med) / warm_med < 0.10,
    }
    with open(f"{VOL}/results/vllm_bench.json", "w") as fh:
        json.dump(result, fh, indent=2)
    volume.commit()
    print(json.dumps({k: v for k, v in result.items() if not k.startswith("per_")}, indent=2))
    return result
