"""Refit a published PorTAL artifact to a new (cheaper) target base."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch
from portallib import (
    ChoiceDataset,
    PortalAdapterRefitter,
    PortalBase,
    PortalModel,
    PortalTrainingConfig,
    RefitResult,
)


def refit_artifact(
    source_artifact: str,
    target_model_id: str,
    dataset: ChoiceDataset,
    *,
    epochs: int = 3,
    batch_size: int = 2,
    refit_max_examples: int = 2000,
    dtype: str = "bfloat16",
    device: str | None = None,
    output_dir: str | Path | None = None,
    on_epoch=None,
) -> RefitResult:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    source = PortalModel.from_pretrained(source_artifact, device=device)
    tokenizer = AutoTokenizer.from_pretrained(target_model_id)
    auto_cls = cast(Any, AutoModelForCausalLM)
    model = auto_cls.from_pretrained(
        target_model_id, torch_dtype=getattr(torch, dtype)
    ).to(device)
    target = PortalBase(model_id=target_model_id, model=model, tokenizer=tokenizer)
    target.freeze()
    config = PortalTrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        refit_max_examples=refit_max_examples,
        eval_max_examples=64,
        eval_batch_size=2,
        gradient_checkpointing=True,
    )
    refitter = PortalAdapterRefitter(source, target, dataset, config=config)
    result = refitter.refit(on_epoch=on_epoch)
    if output_dir is not None:
        result.artifact.save_pretrained(str(output_dir))
    return result
