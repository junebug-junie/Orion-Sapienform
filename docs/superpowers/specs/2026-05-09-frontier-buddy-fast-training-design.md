# Frontier Buddy Fast Training Design (Approach B)

## Context
Frontier Buddy training readiness exists across two rails in this repository:
- Topic Foundry (`services/orion-topic-foundry`) supports topic clustering model runs and artifacts.
- ChatGPT QLoRA SFT (`scripts/run_chatgpt_qlora_sft.py`, `orion/training/chatgpt_qlora/*`) supports adapter training/eval flows.

This design targets the fastest realistic path to train Frontier Buddy with real data while minimizing setup and runtime risk.

## Goal
Deliver a minimal, real (non-simulated) training run for Frontier Buddy that:
- Uses a small sample from Postgres `chat_gpt_derived_example` rows.
- Produces adapter artifacts and training/eval manifests.
- Is visible through the Hub-facing flow as a completed run with lineage.

## Scope
### In scope
- Small bounded extraction from Postgres using `import_run_ids` plus a strict record limit.
- Foundry dataset/partition build with existing routing policy enforcement.
- Tiny real QLoRA training run (small step budget).
- Eval pass that writes base vs adapter output comparisons.
- Hub visibility verification for run/artifact presence.

### Out of scope
- Hyperparameter search and tuning.
- Large-corpus training.
- Model quality benchmarking beyond smoke-level sanity checks.
- Refactoring unrelated services or training architecture.

## Architecture
The runtime path is:
1. Extract a bounded real sample from Postgres.
2. Build dataset and foundry partitions.
3. Run real QLoRA training with minimal step budget.
4. Run eval on held-out prompts.
5. Verify Hub-visible run/artifact lineage.

This keeps the path close to production behavior while constraining runtime and risk.

## Components and Responsibilities
### Data sampler
- Reads from `chat_gpt_derived_example` via configured `postgres_uri`.
- Restricts rows by `import_run_ids` and hard row limit.
- Emits deterministic run metadata (source IDs, run name, timestamps).

### Foundry partitioner
- Uses `orion/training/chatgpt_qlora/foundry.py`.
- Preserves policy that `frontier_oracle` requires rewrite before direct SFT.
- Produces partitioned artifacts and manifests.

### QLoRA trainer
- Uses real path in `orion/training/chatgpt_qlora/trainer.py` (not `--simulate`).
- Runs with minimal compute settings (small batch, low `max_steps`, frequent checkpoints).
- Writes adapter and training manifest artifacts.

### Evaluator
- Uses `orion/training/chatgpt_qlora/eval.py`.
- Produces base vs adapter output snapshots and eval manifest.

### Hub visibility checker
- Uses existing Hub-facing path for topic/training visibility.
- Confirms latest run completion state and artifact lineage are observable.

## Data Flow
1. **Extract** a bounded Postgres sample.
2. **Build** canonical dataset and foundry outputs.
3. **Train** a tiny real QLoRA adapter run.
4. **Evaluate** against held-out prompts.
5. **Expose and verify** run + artifact lineage in Hub.

## Constraints and Success Criteria
### Constraints
- Must be real training mode (no simulation).
- Must keep `frontier_oracle` rewrite-first routing invariant.
- Must finish in short runtime window suitable for smoke execution.

### Success criteria
- `run-all` command completes in real mode.
- Training manifest status is real completion (not simulated).
- Adapter artifact directory exists and is non-empty.
- Eval manifest exists with base and adapter outputs.
- Hub-visible path confirms new run/artifact lineage presence.

## Error Handling Strategy
### Preflight (fail fast)
- Abort early on missing GPU/runtime training dependencies.
- Abort on invalid Postgres connectivity or empty extraction window.
- Abort on missing required config keys for source, limits, and run identity.

### Runtime (preserve evidence)
- If dataset/foundry stage fails, keep partial manifests and phase-specific error.
- If training fails, keep generated upstream artifacts and mark run failed.
- If eval fails post-training, preserve training artifacts and mark eval-specific failure state.

## Verification Plan
1. Run focused command for real-mode `run-all` with tiny config.
2. Confirm training manifest and adapter artifact outputs exist.
3. Confirm eval manifest includes side-by-side outputs.
4. Confirm Hub-facing visibility check sees latest run/artifact lineage.
5. Keep routing invariant checks green for `frontier_oracle` rewrite-before-SFT behavior.

## Risks and Mitigations
- **Risk:** Tiny data and low step budget produce weak quality signals.
  - **Mitigation:** Treat this phase as operational readiness, not quality validation.
- **Risk:** Environment drift (GPU/deps) causes flaky first run.
  - **Mitigation:** enforce preflight checks and explicit dependency failures.
- **Risk:** Hub visibility path lags or caches stale run data.
  - **Mitigation:** verify with explicit latest-run query/check after training completes.

## Follow-on Plan Boundary
After this design is accepted and implemented, the next planning cycle can cover:
- Data window expansion.
- Controlled hyperparameter sweeps.
- Frontier Buddy-specific eval rubric and quality gates.
