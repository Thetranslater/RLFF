# Reward prompt placeholders

Phase C owns this directory. The two text files are deliberately non-final
placeholders for the configured remote reward model. Production training must refuse placeholder
prompts unless the resolved configuration explicitly opts into development
zero rewards.

- `completion_reward_system_v2.txt`: scores one target-character completion using
  its role prompt and visible trajectory context.
- `trajectory_reward_system.txt`: scores task progress and quality over the
  complete character-only trajectory.

Do not add the legacy state checklist to either prompt.
