# exp05_freeform: free-form generation routing (GSM8K, Together AI ladder)

Does cheapest-capable routing hold on free-form generation, graded programmatically
(exact-match, no LLM judge), on a second commercial provider?

## Setup

- **Task**: GSM8K math word problems, free-form chain-of-thought ending in
  `#### <number>`; correctness = numeric exact-match against the reference
  (programmatic grader in `score_gsm8k.py`, no LLM judge).
- **Ladder** (Together AI serverless, measured average $ per request from token
  usage): Qwen2.5-7B-Instruct-Turbo ($7.4e-5) → gpt-oss-20b ($1.18e-4) →
  Llama-3.3-70B-Instruct-Turbo ($2.72e-4).
- **Data**: 150 val (train split, seed 0) + 150 test (test split, seed 0);
  greedy decoding, 640 max tokens.
- **Router**: same recipe as exp03 — prompt-embedding router (MiniLM) fit on 70% of
  val, floor tuned on the remaining 30%, frozen (router, floor) evaluated once on test.

## Results (150 test rows)

| Policy | Accuracy | Savings vs always-70B | Drop |
|---|---:|---:|---:|
| always Qwen2.5-7B | 91.3% | 72.7% | −6.7 pp |
| always gpt-oss-20b | 91.3% | 56.4% | −6.7 pp |
| always Llama-3.3-70B | 98.0% | 0% | 0 |
| oracle (cheapest correct) | 98.7% | 68.7% | +0.7 pp |
| prompt router, floor 1.03 | 94.7% | 40.7% | −3.3 pp |
| **prompt router, val-tuned floor 1.03** | **94.7%** | **40.7%** (CI 34.9–45.6%) | **−3.3 pp** |

- Routing transfers to free-form generation: the val-tuned operating point cut spend
  by **40.7% at a 3.3 pp drop** versus always calling the 70B tier, routing 56% of
  prompts to the 7B tier.
- The oracle again shows complementary errors (+0.7 pp above always-70B at 68.7%
  savings), though headroom is smaller than on the multiple-choice suite because the
  cheap tier is already at 91% on GSM8K.
- The exact same `PromptEmbeddingRouter` + `FloorPolicy` code path used in exp01/exp03
  runs unmodified; only the grader and provider changed. This also extends commercial
  validation to a second provider (Together AI) with real measured per-request costs.

## Limitations

- One free-form task with a verifiable answer; open-ended quality (summarization,
  writing) needs an LLM-as-judge grader, which requires a judge API key and is not
  measured here.
- 150 test rows per model (the scoring key was heavily rate-limited; ~30 s/request
  under concurrency), hence the wide CI.

## Cost

900 API calls ≈ **$0.14** total.

## Reproduce

```bash
export TOGETHERAI_API_KEY=...   # ~$0.14 of API spend
pip install datasets
python experiments/exp05_freeform/score_gsm8k.py --workers 8 --n-val 150 --n-test 150 --max-tokens 640
python experiments/exp05_freeform/run.py   # analysis only, reruns free from committed scores
```
