# exp06: Routing across commercial frontier APIs (OpenAI)

Does cheapest-capable routing hold when the ladder is made of real commercial API
models with real per-request dollar costs, topped by a frontier model?

## Setup

- **Ladder**: gpt-5.4-nano ($0.20/$1.25 per M tokens) → gpt-5.4-mini ($0.75/$4.50)
  → gpt-5.6 (`gpt-5.6-sol`, $5/$30), all via the OpenAI API with
  `reasoning_effort="none"`.
- **Data**: the same seed-0 splits as exp04 — 14 tasks, 1,230 val + 1,230 test rows,
  multiple-choice with letter-answer prompting.
- **Costs**: measured average API cost per request from token usage (not synthetic
  ratios): nano $2.3e-5, mini $8.6e-5, gpt-5.6 $5.66e-4 per request.
- **Router**: prompt-embedding router (MiniLM) trained on per-model correctness from
  70% of val; the floor is tuned on the remaining 30%; the frozen (router, floor) is
  evaluated once on test.

## Results (1,230 test rows)

| Policy | Accuracy | Savings vs always-gpt-5.6 | Drop |
|---|---|---|---|
| always gpt-5.4-nano | 74.3% | 95.9% | −16.3 pp |
| always gpt-5.4-mini | 83.4% | 84.7% | −7.2 pp |
| always gpt-5.6 | 90.7% | 0% | 0 |
| oracle (cheapest correct) | 96.7% | 86.0% | **+6.0 pp** |
| prompt router, floor 1.0 | 89.7% | 7.9% | −1.0 pp |
| prompt router, floor 1.03 | 88.1% | 25.8% | −2.5 pp |
| **prompt router, val-tuned floor 1.05** | **87.1%** | **40.7%** (CI 38.4–43.0%) | **−3.6 pp** |

- The val-tuned operating point cut real API spend by **40.7%** at a 3.6 pp drop
  versus always calling gpt-5.6; a more conservative floor (1.03) keeps the drop
  within 3 pp at 25.8% savings.
- The oracle shows the same complementary-errors effect as the local ladder: perfect
  routing would be **6 pp more accurate than always-frontier while spending 86% less**.
- The router transfers unchanged from the local Qwen ladder setup: same
  `PromptEmbeddingRouter` + `FloorPolicy`, only the correctness labels and dollar
  costs differ.

## Cost

~7,400 API calls ≈ **$1.70** total.

## Limitations

- Same 14 multiple-choice tasks as exp04; free-form generation is not measured.
- One provider (OpenAI); letter-answer prompting with no reasoning effort.
- Savings depend on the price gap between tiers (25x nano-to-frontier here).

## Reproduce

```bash
export OPENAI_API_KEY=...   # ~$1.70 of API spend
python experiments/exp06_commercial_api/score_openai.py
python experiments/exp06_commercial_api/run.py   # analysis only, reruns free from committed scores
```
