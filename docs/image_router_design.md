# Image model routing design

## Problem

Image generation has the same model-proliferation problem LLMs have: many open and closed source models (Stable Diffusion XL, Flux, Ideogram, Recraft, DALL-E 3, Midjourney, Firefly, Imagen, ...) with different strengths, costs and latencies. There is no good way to route a prompt to the cheapest model that can deliver the desired *ability* (prompt following, text rendering, anatomy, photorealism) and *vibe* (anime, oil painting, 3D render, logo, editorial).

## Claim

The same routing machinery we built for LLMs works for image generators once the right signal is used. For text we route on *correctness*. For images the analogous signal is *human preference* conditioned on the prompt. A lightweight prompt-embedding classifier can predict which image model is most likely to produce the preferred output for a given prompt, letting us send each request to the cheapest capable model and saving cost with bounded quality loss.

## Design

```
client (OpenAI-compatible image endpoint)
   │
   ▼
ImageGateway (/v1/images/generations)
   │  task/vibe = caller-supplied or VibeClassifier(prompt, optional reference image)
   ▼
ImageVibeRouter                          modelrouter.image_routing
   │  p_win per registered image model (prompt embeddings on the hot path)
   ▼
FloorPolicy / CascadePolicy              modelrouter.dispatch (reused)
   │  cheapest model whose p_win is within floor of the best
   ▼
ImageBackend                              modelrouter.image_backends (new seam)
   │  closed:  DALL-E 3, Ideogram, Recraft, Midjourney, Firefly, Imagen via provider SDKs
   │  open:    local diffusers / ComfyUI / SDXL / Flux via HTTP or pipeline
   ▼
TraceJournal                              modelrouter.tracing (reused)
      JSONL: prompt, vibe, candidates, chosen, reason, preference score
```

### Router

`ImageVibeRouter` is a multi-class classifier over a shared multi-modal prompt encoder. It predicts, for each registered image model, the probability that the model produces the *most preferred* image for the prompt. We train it on pairwise or ranked human preference data:

1. **Encoder**: a pre-trained text/image encoder (`openai/clip-vit-base-patch32`, `timm/ViT-SigLIP`, or `sentence-transformers` for a text-only fast path). The prompt text is embedded into a fixed-size vector. If a reference image is supplied it is embedded with the same model and concatenated.
2. **Label construction**: preference datasets give pairwise comparisons for the same prompt. We convert these to a per-prompt *best model* label using a Bradley-Terry/Elo or simple win-count aggregation.
3. **Classifier**: a small `LogisticRegression` or `MLPClassifier` per model (mirroring `PromptEmbeddingRouter`) that predicts `P(model m wins for this prompt)`.
4. **Policy**: reuse `FloorPolicy` — sort models by cost, pick the cheapest whose `p_win * floor >= max_p_win`. `CascadePolicy` can also be used: generate with the cheap model, run a learned preference/vibe validator, and escalate if confidence is low.

### Training data

Public datasets:

- **HPDv2 / HPSv2** (`ymhao/HPDv2` on HuggingFace): 798k pairwise comparisons over 107k prompts and many models. Best for training the general router.
- **Pick-a-Pic**: 500k pairs.
- **ImageRewardDB**: 137k expert comparisons.
- **Open Image Preferences** (`data-is-better-together/open-image-preferences-v1`): community preference pairs.

For the initial prototype we synthesize a small benchmark that mimics these labels, because downloading the full 35GB HPDv2 image payload is unnecessary to validate the routing mechanics — the annotations are enough.

### Vibe classification

`ImageVibeClassifier` is an optional prompt-only classifier that maps a request to a coarse vibe bucket (`photo`, `anime`, `illustration`, `3d`, `logo`, `painting`, `abstract`). It is trained on prompt text and used either:

- to label traces for retraining, or
- as an abstain signal: when vibe confidence is low, route to the capable (largest) model.

### Cost model

Image model costs are per-image, not per-token. `ImageBackendSpec` extends `BaseSpec` with `cost_per_image` and `latency_ms`. The dispatch policies already optimize `cost`, so no policy change is needed.

## Validation plan

1. Build `ImageVibeRouter` and a synthetic image-preference benchmark where prompts contain style cues that correlate with known model strengths.
2. Train the router on sentence-transformer (and optionally CLIP) embeddings.
3. Run a policy sweep with `FloorPolicy` and compute accuracy, cost, savings, drop vs. always-capable, and routed-cheap fraction, reusing `eval.policy_stats` and `eval.bootstrap_ci`.
4. Check the kill criteria from `eval.check_kill_criteria` (≥15% savings at ≤3 pp quality drop).
5. Produce a Pareto plot and a results JSON under `experiments/exp06_image_vibe_routing/`.
6. Add unit tests covering router training, dispatch, and the end-to-end sweep.

## Why this is the right seam

- It reuses the existing `BaseSpec`, `FloorPolicy`, `CascadePolicy`, and `eval` machinery.
- It swaps the *label* (correctness → preference) and the *encoder* (sentence-transformer/CLIP), but the routing math is identical.
- It generalizes to real preference data by replacing the synthetic dataset loader with `datasets.load_dataset("ymhao/HPDv2")`.
- It explicitly separates *ability* (which model wins for a prompt) from *vibe* (coarse style classes) while letting the policy trade them off against cost.
