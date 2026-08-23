"""RLFF shared-policy, multi-character reinforcement-learning package.

SUBAGENT BRIEF
==============
This package is being implemented in four coarse phases described in
``src/rlff/IMPLEMENTATION.md``. Read that document before changing any RLFF
module. The implementation uses AReaL with SGLang, continues the existing SFT
LoRA adapter in BF16, generates character-only trajectories with direct
round-robin scheduling, and computes role-level GRPO advantages.

Do not restore the removed TRL single-turn trainer, the legacy state verifier,
an Environment/narrator actor, or a generic scheduler/plugin framework. Keep
imports from CUDA-only frameworks lazy so this package remains importable in
the local CPU development environment.
"""

__all__: list[str] = []
