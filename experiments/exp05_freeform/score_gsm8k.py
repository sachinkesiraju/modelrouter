"""Score free-form GSM8K generation on a Together AI ladder with real $ costs.

Programmatic exact-match grading (no LLM judge): the model writes its reasoning
and a final ``#### <number>`` line; a row is correct iff the extracted number
equals the reference. Writes results/gsm8k_scores.json with per-row correctness
and measured API cost per model.

  export TOGETHERAI_API_KEY=...
  pip install datasets
  python experiments/exp05_freeform/score_gsm8k.py
"""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import litellm

HERE = Path(__file__).parent

MODELS = [
    "together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo",
    "together_ai/openai/gpt-oss-20b",
    "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
]

PROMPT = ("Solve the following math word problem. Think step by step, then give the final "
          "numeric answer on its own line in the form '#### <number>'.\n\n{question}")


def extract_answer(text: str) -> str | None:
    m = re.findall(r"####\s*\$?(-?[\d,]+(?:\.\d+)?)", text)
    if not m:
        # Fall back to the last number in the response.
        m = re.findall(r"(-?[\d,]+(?:\.\d+)?)", text)
    if not m:
        return None
    return m[-1].replace(",", "").rstrip(".")


def grade(text: str, reference: str) -> int:
    pred = extract_answer(text)
    gold = extract_answer(reference)
    if pred is None or gold is None:
        return 0
    try:
        return int(float(pred) == float(gold))
    except ValueError:
        return int(pred == gold)


def ask(model: str, question: str, reference: str, max_tokens: int) -> dict:
    for attempt in range(5):
        try:
            resp = litellm.completion(
                model=model, max_tokens=max_tokens, temperature=0.0, timeout=120,
                messages=[{"role": "user", "content": PROMPT.format(question=question)}],
            )
            break
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    text = resp.choices[0].message.content or ""
    usage = resp.usage
    cost = litellm.completion_cost(completion_response=resp) or 0.0
    return {"correct": grade(text, reference), "pred": extract_answer(text), "cost_usd": cost,
            "prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-val", type=int, default=300)
    parser.add_argument("--n-test", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    from datasets import load_dataset

    train = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=0)
    test = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=0)
    splits = {"val": train.select(range(args.n_val)), "test": test.select(range(args.n_test))}

    out: dict = {"models": MODELS, "splits": {}}
    for split_name, rows in splits.items():
        out["splits"][split_name] = {
            "prompts": [r["question"] for r in rows],
            "tasks": ["gsm8k"] * len(rows),
        }
        for model in MODELS:
            with ThreadPoolExecutor(args.workers) as pool:
                futures = [pool.submit(ask, model, r["question"], r["answer"], args.max_tokens)
                           for r in rows]
                results = []
                for i, fut in enumerate(futures):
                    results.append(fut.result())
                    if (i + 1) % 50 == 0:
                        print(f"  {split_name} {model}: {i + 1}/{len(rows)}", flush=True)
            acc = sum(r["correct"] for r in results) / len(results)
            spend = sum(r["cost_usd"] for r in results)
            print(f"{split_name} {model}: acc={acc:.3f} spend=${spend:.2f}", flush=True)
            out["splits"][split_name][model] = results

    (HERE / "results").mkdir(exist_ok=True)
    with open(HERE / "results" / "gsm8k_scores.json", "w") as fh:
        json.dump(out, fh)


if __name__ == "__main__":
    main()
