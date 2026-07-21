import json

import pytest

from modelrouter.costs import GpuCostModel


def test_usd_per_request() -> None:
    model = GpuCostModel(gpu_hourly_usd=3.6, throughput_tok_s={"a": 100.0}, gpu="A10G")
    # 100 tokens at 100 tok/s = 1 s of GPU at $3.6/h = $0.001.
    assert model.usd_per_request("a", 100.0) == pytest.approx(0.001)


def test_relative_costs_are_inverse_throughput_ratios() -> None:
    model = GpuCostModel(gpu_hourly_usd=1.1, throughput_tok_s={"small": 400.0, "big": 100.0})
    rel = model.relative_costs()
    assert rel["small"] == pytest.approx(1.0)
    assert rel["big"] == pytest.approx(4.0)


def test_from_load_bench(tmp_path) -> None:
    combined = {
        "Qwen/Qwen3-0.6B": {"gpu": "A10G", "peak_output_tok_per_s": 900.0},
        "Qwen/Qwen3-4B": {"gpu": "A10G", "peak_output_tok_per_s": 300.0},
    }
    path = tmp_path / "load_bench.json"
    path.write_text(json.dumps(combined))
    model = GpuCostModel.from_load_bench(path, gpu_hourly_usd=1.1)
    assert model.gpu == "A10G"
    assert model.relative_costs()["Qwen/Qwen3-4B"] == pytest.approx(3.0)


def test_from_load_bench_rejects_mixed_gpus(tmp_path) -> None:
    combined = {
        "a": {"gpu": "A10G", "peak_output_tok_per_s": 900.0},
        "b": {"gpu": "A100", "peak_output_tok_per_s": 300.0},
    }
    path = tmp_path / "load_bench.json"
    path.write_text(json.dumps(combined))
    with pytest.raises(ValueError):
        GpuCostModel.from_load_bench(path, gpu_hourly_usd=1.1)
