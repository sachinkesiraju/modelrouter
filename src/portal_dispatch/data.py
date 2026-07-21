"""Deterministic splits over the portallib task suite."""

from __future__ import annotations

import random
from dataclasses import dataclass

from portallib import ChoiceDataset, ChoiceExample


@dataclass(frozen=True)
class SuiteSplits:
    train: dict[str, list[ChoiceExample]]
    val: dict[str, list[ChoiceExample]]
    test: dict[str, list[ChoiceExample]]

    @property
    def tasks(self) -> tuple[str, ...]:
        return tuple(self.train.keys())


def make_splits(
    dataset: ChoiceDataset,
    *,
    train_per_task: int = 20,
    val_per_task: int = 15,
    test_per_task: int = 15,
    seed: int = 0,
) -> SuiteSplits:
    """Carve deterministic train/val/test slices per task.

    Train comes from the upstream train split; val/test are disjoint slices of
    the upstream validation split, shuffled with a fixed seed.
    """
    rng = random.Random(seed)
    train: dict[str, list[ChoiceExample]] = {}
    val: dict[str, list[ChoiceExample]] = {}
    test: dict[str, list[ChoiceExample]] = {}
    for task in dataset.tasks:
        train_rows = list(dataset.rows("train", task))
        rng.shuffle(train_rows)
        train[task] = train_rows[:train_per_task]
        heldout = list(dataset.rows("validation", task))
        rng.shuffle(heldout)
        val[task] = heldout[:val_per_task]
        test[task] = heldout[val_per_task : val_per_task + test_per_task]
    return SuiteSplits(train=train, val=val, test=test)
