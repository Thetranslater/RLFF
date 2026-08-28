# RLFF implementation contract

This is the authoritative implementation brief for every subagent working in
`src/rlff`. It supersedes the TRL and slime-specific parts of older planning
documents. Do not silently broaden the scope or substitute another framework.

## Fixed decisions

- One shared policy and one existing SFT LoRA adapter generate every character.
- Continue training that exact adapter in BF16 LoRA mode. Do not create a new
  empty adapter, merge it into the base model, or treat the base model as the
  reference policy.
- AReaL provides distributed training/checkpoint infrastructure; SGLang
  provides rollout. Both integrations must use lazy imports.
- An episode contains an ordered, non-empty `characters` list. Generation is
  direct round-robin in that order. There is intentionally no scheduler
  abstraction in the first implementation.
- `max_rounds` is an upper bound on complete round-robin cycles. The production
  horizon is derived from character count: 7 rounds for one or two characters,
  6 for three characters, and 5 for four or more characters. Every character
  therefore receives the same number of completions. Natural conversation
  termination is outside scope. EOS ends only the current completion.
- Generated trajectories contain character messages only. There is no
  Environment/narrator role, reward, completion, or loss mask.
- Every generation uses the target character's system prompt, profile, plot,
  shared tasks, private tasks, and visible dialogue history. Previous messages
  by that character are assistant history; other characters are serialized as
  named user history.
- All trajectories in one `group_id` use identical episode content, character
  order, prompt template choices, and renderer randomness. Only model sampling
  differs.
- Generated token IDs, completion boundaries, and rollout log probabilities
  must come directly from SGLang. Never reconstruct training tokens by
  tokenizing returned text.
- First-version rewards contain only completion-local character rewards and a
  trajectory-global reward. The old initialization/instant state verifier is
  not part of the new training path.
- Both reward scopes ultimately call Qwen3.7-Flash through native DashScope HTTP. Until reward
  prompts/schemas are finalized, an explicitly enabled zero-reward provider
  keeps the pipeline runnable.
- Invalid RM output is never silently converted to a zero reward. After bounded
  retries, mark the affected trajectory/group invalid.
- Role rewards are normalized across trajectories by `(group_id, character)`.
  The resulting advantage is applied only to that character completion's
  tokens. Do not sum all character rewards into one undifferentiated trajectory
  scalar.
- A trajectory normally ends with `termination_reason=max_rounds`. Generation,
  context-limit, or infrastructure failures produce explicit invalid/truncated
  metadata and are not disguised as successful early termination.

## Coarse implementation phases

### Phase A — foundation and episode preparation

Owns `contracts.py`, `config.py`, and `episodes.py` plus their tests.

Deliver a strict, versioned protocol; JSONL loading; deterministic IDs and
fingerprints; validation; group sampling; target-character prompt projection;
and configuration validation. This phase does not call SGLang, DashScope, or
AReaL.

### Phase B — multi-character rollout

Owns `rollout.py` plus its tests.

Deliver the direct round-robin loop, target-role prompt rendering, SGLang
request/response conversion, exact token/logprob capture, fixed policy-version
checks, context-limit handling, and trajectory assembly. Do not add a scheduler
interface, tool calls, external environment, narrator, or natural-stop model.

### Phase C — DashScope reward pipeline

Owns `rewards.py`, `prompts/`, and their tests.

Deliver strict local/global response schemas, placeholder zero rewards,
native DashScope HTTP calls, bounded retry/concurrency, LangSmith tracing, raw response
audit records, scalar aggregation, and invalid-result propagation. Do not add
state validation or LangChain/LangGraph.

### Phase D — role GRPO and training assembly

Owns `grpo.py`, `runtime.py`, `cli.py`, and their tests.

Deliver role subgroups, numerical advantage computation, completion/token
masks, AReaL batch conversion, trainable SFT adapter loading, frozen initial
adapter reference semantics, checkpoint metadata, metrics, dry-run validation,
and the executable training entry point. Reuse AReaL training/checkpoint
machinery; do not reimplement a distributed trainer.

## Cross-phase data flow

```text
EpisodeRecord
  -> G identical EpisodeSample objects with distinct seeds and one group_id
  -> round-robin SGLang rollout
  -> RolloutGroup[Trajectory[CompletionTrace]]
  -> local completion rewards + global trajectory rewards
  -> role effective rewards
  -> (group_id, character) normalization
  -> token-masked AReaL training batch
  -> BF16 update of the existing SFT LoRA adapter
```

The protocol must distinguish `episode_id`, `group_id`, `trajectory_id`, and
`completion_id`. Persist prompt/render versions, tokenizer fingerprint, policy
version, adapter identity, termination reason, and raw RM results for audit and
resume validation.

## Local versus cloud execution

The local Windows/AMD environment must run imports, schema tests, prompt tests,
reward tests with fake HTTP, advantage tests, and fake rollout tests. AReaL,
SGLang, NCCL, and CUDA execution are cloud-only. Never place their imports at
module top level.

## Required working style

- Preserve user data and unrelated legacy extraction scripts.
- Add focused tests with each phase; do not leave untested placeholder logic
  presented as complete.
- Use Pydantic for boundary schemas and plain tensor/numeric structures inside
  hot training paths.
- Prefer explicit failures over fallback retokenization, fabricated replies,
  or implicit zero rewards.
- If AReaL cannot load and continue the exact LLaMA-Factory SFT adapter, stop
  and report the blocker. Do not silently change the training semantics.
