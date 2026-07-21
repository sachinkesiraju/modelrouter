# Architecture

```
client (OpenAI SDK)
   │
   ▼
Gateway (/v1/chat/completions)        modelrouter.gateway
   │  task = caller-supplied or TaskClassifier(prompt)
   ▼
Router                                 modelrouter.routing
   │  p_correct per registered base (prompt embeddings on the hot path)
   ▼
DispatchPolicy                         modelrouter.dispatch
   │  cheapest base with p * floor >= max p   (or cascade + escalation)
   ▼
Backend                                modelrouter.runtime
   │  PortalModel.generate(task) → PortalInjector.activate → forward
   ▼
TraceJournal                           modelrouter.tracing
      JSONL: prompt, task, candidates, chosen, reason, scores
```

## Components

- **Routers** are trained offline against the *oracle*: per-(query, base) correctness labels
  obtained by scoring every registered base on the private task suite
  (`experiments/exp02_policy_sweep/build_bundle.py`). Three signal families:
  - `ScoreRouter` — features of the candidate base's per-choice score distribution
    (top, second, margin, entropy). Most accurate; requires the candidate forward pass.
  - `PromptEmbeddingRouter` — per-base correctness classifiers on MiniLM prompt
    embeddings. Prompt-only, production hot path.
  - `LatentRouter` — predicts per-task base suitability from the PorTAL task latent `z`;
    prices *unseen tasks* before any query arrives.
- **Policies** are config, not code: `FloorPolicy(floor)` trades savings vs quality with one
  knob tuned on validation under a quality constraint; `CascadePolicy(threshold)` runs the
  cheap base and escalates on low confidence (double-inference cost accounted).
- **Backends** hide the serving substrate behind one seam. `HFBackend` (reference) loads a
  base once and hot-swaps task LoRAs via `PortalInjector` with an adapter cache. A vLLM
  backend (PEFT export + `LoRARequest` hot-swap) is the performance path (see roadmap).
- **Kill criteria** are machine-checked (`eval.check_kill_criteria`): a policy ships only if
  it achieves ≥15% cost savings at ≤3 pp accuracy drop vs always-capable.
