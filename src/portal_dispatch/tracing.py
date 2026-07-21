"""JSONL decision-trace journal: the observability store and router-retraining dataset."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TraceJournal:
    path: str | Path

    def emit(self, **event: Any) -> None:
        record = {"ts": time.time(), **event}
        with open(self.path, "a") as fh:
            fh.write(json.dumps(record) + "\n")

    def read(self) -> list[dict[str, Any]]:
        p = Path(self.path)
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
