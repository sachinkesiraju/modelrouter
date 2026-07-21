"""Score the exp04 held-out suite on commercial OpenAI models with real $ costs.

Writes results/openai_scores.json: per-row correctness and measured API cost for
each model on the same seed-0 val/test splits used in exp04.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import litellm
from portallib import ChoiceDataset

from modelrouter.data import make_splits

HERE = Path(__file__).parent

MODELS = ["gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.6-sol"]
LETTERS = "ABCDEFGHIJKLMNOP"


def ask(model: str, row) -> dict:
    options = "\n".join(f"{LETTERS[i]}.{c}" for i, c in enumerate(row.choices))
    messages = [{
        "role": "user",
        "content": (f"{row.prompt.strip()}\n\n{options}\n\n"
                    "Reply with only the letter of the best answer."),
    }]
    for attempt in range(5):
        try:
            resp = litellm.completion(model=model, messages=messages,
                                      reasoning_effort="none", max_tokens=128)
            break
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    text = (resp.choices[0].message.content or "").strip()
    m = re.search(r"[A-P]", text.upper())
    pred = LETTERS.index(m.group(0)) if m else -1
    usage = resp.usage
    cost = (litellm.completion_cost(completion_response=resp) or 0.0)
    return {"task": row.task, "correct": int(pred == row.gold_idx), "pred": pred,
            "cost_usd": cost, "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-per-task", type=int, default=100)
    parser.add_argument("--test-per-task", type=int, default=100)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    dataset = ChoiceDataset.from_hub("RampPublic/portallib-tasks")
    splits = make_splits(dataset, train_per_task=1, val_per_task=args.val_per_task,
                         test_per_task=args.test_per_task)
    out: dict = {"models": MODELS, "splits": {}}
    for split_name in ("val", "test"):
        rows = [r for task in splits.tasks for r in getattr(splits, split_name)[task]]
        out["splits"][split_name] = {
            "prompts": [r.prompt for r in rows],
            "tasks": [r.task for r in rows],
        }
        for model in MODELS:
            with ThreadPoolExecutor(args.workers) as pool:
                results = list(pool.map(lambda r: ask(model, r), rows))
            acc = sum(r["correct"] for r in results) / len(results)
            spend = sum(r["cost_usd"] for r in results)
            print(f"{split_name} {model}: acc={acc:.3f} spend=${spend:.2f}", flush=True)
            out["splits"][split_name][model] = results

    (HERE / "results").mkdir(exist_ok=True)
    with open(HERE / "results" / "openai_scores.json", "w") as fh:
        json.dump(out, fh)


if __name__ == "__main__":
    main()
