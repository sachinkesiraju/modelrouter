"""exp02: vLLM adapter-swap benchmark on Modal (the serving-substrate gate).

Exports PorTAL task LoRAs to standard PEFT directories, serves the base on vLLM
with LoRA hot-swap, and measures per-task swap overhead vs the no-LoRA baseline.

Gate (roadmap P0): median adapter-swap overhead < 10% of request latency.

  modal run experiments/exp02_vllm_bench/modal_app.py::bench

``load_bench`` measures streaming (TTFT / inter-token latency) plus batched
concurrent throughput against a live OpenAI-compatible vLLM server, per ladder
model. Its saturated tokens/sec also feeds the GPU-amortized cost model used by
exp01 (`modelrouter.costs`).

  modal run experiments/exp02_vllm_bench/modal_app.py::load_bench_ladder
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
    base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
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


@app.function(image=image, gpu="A10G", volumes={VOL: volume}, timeout=3600)
def load_bench(model_id: str = "Qwen/Qwen3-1.7B", max_tokens: int = 128,
               concurrencies: str = "1,4,16,32", n_stream: int = 10) -> dict:
    """Streaming + concurrent-load benchmark against a live OpenAI-compatible vLLM server.

    Measures per-request TTFT / inter-token latency (streaming) and aggregate
    decode throughput + latency percentiles under concurrent load. The peak
    aggregate tokens/sec is the input to the GPU-amortized cost model
    (``modelrouter.costs``).
    """
    import asyncio
    import statistics
    import subprocess
    import time

    import httpx

    port = 8321
    proc = subprocess.Popen(
        ["vllm", "serve", model_id, "--port", str(port), "--gpu-memory-utilization", "0.85"],
    )
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 1200
    while time.time() < deadline:
        try:
            if httpx.get(f"{base_url}/health", timeout=2).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(3)
    else:
        proc.kill()
        raise RuntimeError("vLLM server failed to start")

    prompt = ("You are a helpful assistant. Summarize the trade-offs between routing every "
              "request to the largest available language model versus routing each request "
              "to the cheapest model predicted to answer it correctly. Cover cost, accuracy, "
              "latency, and operational complexity, with concrete examples.")
    body = {"model": model_id, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0.0, "ignore_eos": True}

    async def one_stream(client: httpx.AsyncClient) -> tuple[float, float, int]:
        start = time.perf_counter()
        ttft, n_chunks, last = None, 0, start
        async with client.stream("POST", f"{base_url}/v1/completions",
                                 json=dict(body, stream=True), timeout=300) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - start
                n_chunks += 1
                last = now
        itl = (last - start - ttft) / max(n_chunks - 1, 1)
        return ttft * 1000, itl * 1000, n_chunks

    async def one_request(client: httpx.AsyncClient) -> tuple[float, int]:
        start = time.perf_counter()
        resp = await client.post(f"{base_url}/v1/completions", json=body, timeout=600)
        resp.raise_for_status()
        return (time.perf_counter() - start) * 1000, resp.json()["usage"]["completion_tokens"]

    async def run_all() -> dict:
        async with httpx.AsyncClient() as client:
            # Warmup.
            await one_request(client)
            # Streaming: sequential streamed requests -> TTFT and inter-token latency.
            streams = [await one_stream(client) for _ in range(n_stream)]
            stream_stats = {
                "ttft_ms_median": statistics.median(s[0] for s in streams),
                "ttft_ms_p95": sorted(s[0] for s in streams)[int(0.95 * (n_stream - 1))],
                "inter_token_ms_median": statistics.median(s[1] for s in streams),
            }
            # Concurrent load: c simultaneous clients, 4 requests each.
            load = {}
            for c in (int(x) for x in concurrencies.split(",")):
                n_req = 4 * c
                start = time.perf_counter()
                results = await asyncio.gather(
                    *(one_request(client) for _ in range(n_req))
                )
                wall = time.perf_counter() - start
                lats = sorted(r[0] for r in results)
                out_tokens = sum(r[1] for r in results)
                load[str(c)] = {
                    "n_requests": n_req,
                    "wall_s": wall,
                    "output_tok_per_s": out_tokens / wall,
                    "req_per_s": n_req / wall,
                    "latency_ms_p50": lats[len(lats) // 2],
                    "latency_ms_p95": lats[int(0.95 * (len(lats) - 1))],
                }
            return {"streaming": stream_stats, "load": load}

    try:
        stats = asyncio.run(run_all())
    finally:
        proc.kill()

    peak = max(v["output_tok_per_s"] for v in stats["load"].values())
    result = {"model": model_id, "gpu": "A10G", "max_tokens": max_tokens,
              "peak_output_tok_per_s": peak, **stats}
    import os
    os.makedirs(f"{VOL}/results", exist_ok=True)
    with open(f"{VOL}/results/load_bench_{model_id.split('/')[-1]}.json", "w") as fh:
        json.dump(result, fh, indent=2)
    volume.commit()
    print(json.dumps(result, indent=2))
    return result


@app.local_entrypoint()
def load_bench_ladder(concurrencies: str = "1,4,16,32") -> None:
    """Run load_bench for the full local ladder and write a combined results file."""
    from pathlib import Path

    models = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"]
    combined = {m: r for m, r in zip(models, load_bench.map(models, kwargs={"concurrencies": concurrencies}))}
    out = Path(__file__).parent / "results" / "load_bench.json"
    out.write_text(json.dumps(combined, indent=2))
    print("wrote", out)
