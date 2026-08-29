# RLFF deploy source layout

`deploy/rlff_phase_d/src/rlff` is the only maintained RLFF training source tree.
The repository-level `src/rlff` directory is legacy material and is neither
imported nor synchronized by this deployment package.

The package keeps the historical public imports stable while splitting the
implementation by responsibility:

- `rewards/`
  - `protocol.py`: schemas, constants, and reward-boundary errors.
  - `prompts.py`: prompt loading/rendering and response parsing.
  - `payloads.py`: canonical and proxy reward payload construction.
  - `transport.py` and `transport_types.py`: HTTP/tracing boundary and interfaces.
  - `provider_base.py` and `providers.py`: placeholder and production providers.
  - `response_scoring.py` and `aggregation.py`: scalar conversion and role rewards.
- `proxy/`
  - `types.py`: proxy records and provider protocol.
  - `grouping.py`: character order and reward-weight schedule.
  - `normalization.py`: role-level GRPO normalization.
  - `agent.py`: grouped rollout/reward barrier and OpenAI-proxy workflow.
- `runtime/`
  - `types.py`: pinned runtime records, constants, and compatibility errors.
  - `adapter.py`: SFT LoRA inspection and compatibility checks.
  - `preflight.py`: AReaL YAML validation and runtime planning.
  - `integration.py`: lazy AReaL classes, dataset, and workflow wiring.
  - `training.py`: the real training entrypoint and dry-run report.
- `rollout/`
  - `types.py`: rollout errors and tokenized-prompt record.
  - `tokenizer.py`: tokenizer adapter and backend protocol.
  - `generation.py`: generation request/result and prompt projection.
  - `policy.py`: round policy and rollout setting validation.
  - `engine.py`: framework-neutral trajectory state machine.

The four package `__init__.py` files are compatibility facades. Existing code
may continue importing from `rlff.rewards`, `rlff.proxy`, `rlff.runtime`, and
`rlff.rollout`; new internal code should import from the owning submodule.

