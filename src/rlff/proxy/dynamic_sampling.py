"""AReaL group wrapper for RLFF trajectory-reward dynamic sampling.

The pinned AReaL ``dynamic_filter_fn`` rejects a group and then advances the
dataloader.  RLFF instead needs to regenerate the *same* episode until at
least one character has non-constant trajectory rewards across the group.
This module installs a narrow runtime replacement for AReaL's grouping wrapper
without modifying the vendored/installed AReaL package.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

_ATTEMPT_FIELD = "_rlff_resample_attempt"
_WRAPPER_MARKER = "_rlff_same_episode_dynamic_sampling"


def _agent_from_workflow(workflow: Any) -> Any | None:
    """Return the inline proxy agent owned by an OpenAIProxyWorkflow."""

    return getattr(workflow, "agent", None)


def install_same_episode_group_wrapper() -> bool:
    """Install RLFF's same-episode group wrapper in the current AReaL worker.

    The agent is instantiated inside ``RemoteInfEngine._resolve_workflow``
    immediately before AReaL looks up ``GroupedRolloutWorkflow``.  Replacing
    that module global here therefore affects only runtime grouping; no AReaL
    source file is edited.  CPU-only unit tests may not have AReaL installed,
    in which case scoring logic remains directly testable and ``False`` is
    returned.
    """

    try:
        from areal.api import RolloutWorkflow
        from areal.experimental.openai import InteractionWithTokenLogpReward
        from areal.infra import remote_inf_engine
        from areal.utils.data import concat_padded_tensors
    except ModuleNotFoundError:
        return False

    current = remote_inf_engine.GroupedRolloutWorkflow
    if getattr(current, _WRAPPER_MARKER, False):
        return True

    class RLFFSameEpisodeGroupedRolloutWorkflow(RolloutWorkflow):
        """Run complete fresh groups until the RLFF agent accepts one."""

        _rlff_same_episode_dynamic_sampling = True

        def __init__(self, workflow: Any, group_size: int, logger: Any) -> None:
            if group_size < 1:
                raise ValueError(f"group_size must be >= 1, got {group_size}")
            self.workflow = workflow
            self.group_size = group_size
            self.logger = logger

        def _merge(self, results: list[Any]) -> dict[str, Any] | None:
            valid_results = [result for result in results if result is not None]
            if not valid_results:
                return None
            if len(valid_results) != len(results):
                self.logger.warning(
                    "RLFF grouped rollout returned %d/%d valid trajectories; "
                    "rejecting the incomplete group",
                    len(valid_results),
                    len(results),
                )
                return None
            first = valid_results[0]
            if (
                isinstance(first, dict)
                and first
                and all(
                    isinstance(value, InteractionWithTokenLogpReward)
                    for value in first.values()
                )
            ):
                merged: dict[str, Any] = {}
                for result in valid_results:
                    merged.update(result)
                return merged or None
            concatenated = concat_padded_tensors(valid_results)
            return concatenated or None

        async def arun_episode(
            self,
            engine: Any,
            data: dict[str, Any],
        ) -> dict[str, Any] | None:
            agent = _agent_from_workflow(self.workflow)
            dynamic = bool(
                getattr(agent, "dynamic_trajectory_resampling", False)
            )
            attempt = 0
            while True:
                attempt_data = dict(data)
                attempt_data[_ATTEMPT_FIELD] = attempt
                results = await asyncio.gather(
                    *(
                        # Keep each proxy session's input isolated even if a
                        # future AReaL/OpenAI workflow mutates its payload.
                        self.workflow.arun_episode(engine, dict(attempt_data))
                        for _ in range(self.group_size)
                    )
                )
                if not dynamic:
                    return self._merge(list(results))

                consume = getattr(agent, "consume_group_sampling_decision", None)
                if not callable(consume):
                    raise RuntimeError(
                        "RLFF dynamic sampling agent does not expose its group decision"
                    )
                accepted = await consume(attempt_data)
                if accepted:
                    self.logger.info(
                        "RLFF accepted episode group after %d attempt(s)",
                        attempt + 1,
                    )
                    return self._merge(list(results))

                # Every result belongs to a completed, independent proxy
                # session.  Dropping them here guarantees rejected token IDs,
                # logprobs, and rewards never reach AReaL's training batch.
                self.logger.info(
                    "RLFF rejected constant-trajectory-reward group; "
                    "resampling the same episode (next attempt=%d)",
                    attempt + 2,
                )
                attempt += 1

    remote_inf_engine.GroupedRolloutWorkflow = RLFFSameEpisodeGroupedRolloutWorkflow
    return True


def attempt_index(data: Mapping[str, Any]) -> int:
    """Validate and return the internal same-episode attempt index."""

    value = data.get(_ATTEMPT_FIELD, 0)
    if type(value) is not int or value < 0:
        raise ValueError(f"{_ATTEMPT_FIELD} must be a non-negative integer")
    return value


__all__ = ["attempt_index", "install_same_episode_group_wrapper"]
