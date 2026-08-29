"""Completion and trajectory reward payload construction.

This file is part of the deploy-only RLFF implementation. Keep changes here;
the repository-level src/rlff tree is intentionally not synchronized.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from ..contracts import CharacterSpec, CompletionTrace, EpisodeRecord, Trajectory
from .protocol import REWARD_REQUEST_SCHEMA_VERSION, RewardResponseError


def _character(episode: EpisodeRecord, name: str) -> CharacterSpec:
    for character in episode.characters:
        if character.name == name:
            return character
    raise ValueError(f"character {name!r} is not registered in episode {episode.episode_id}")


def _turn_payload(turn: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "character": turn.character,
        "content": turn.content,
    }
    if getattr(turn, "turn_id", None) is not None:
        result["turn_id"] = turn.turn_id
    return result


def _trajectory_character_completions(
    trajectory: Trajectory,
    character: str,
) -> tuple[CompletionTrace, ...]:
    """Return one character's completions in their stable trajectory order."""

    result = tuple(
        sorted(
            (
                completion
                for completion in trajectory.completions
                if completion.character == character
            ),
            key=lambda completion: completion.turn_index,
        )
    )
    if not result:
        raise ValueError(
            f"trajectory {trajectory.trajectory_id!r} has no completion for {character!r}"
        )
    return result


def _full_trajectory_history(
    episode: EpisodeRecord,
    trajectory: Trajectory,
) -> list[dict[str, Any]]:
    """Return initial dialogue plus every generated turn in trajectory order."""

    history = [_turn_payload(turn) for turn in episode.dialogue]
    history.extend(_turn_payload(turn) for turn in trajectory.turns)
    return history


def _history_text(turns: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(
        f"{turn.get('character', turn.get('speaker', ''))!s}:{turn.get('content', '')!s}"
        for turn in turns
    )


def _indexed_utters(texts: Sequence[str]) -> list[dict[str, Any]]:
    """Label target-role replies so a repair request cannot invent coverage."""

    return [{"index": index, "content": text} for index, text in enumerate(texts)]


def _merged_tasks(shared_tasks: Sequence[Any], private_tasks: Sequence[Any]) -> tuple[str, ...]:
    """Merge shared and target-private tasks once, preserving their source order."""

    result: list[str] = []
    seen: set[str] = set()
    for value in (*shared_tasks, *private_tasks):
        task = str(value).strip()
        if task and task not in seen:
            result.append(task)
            seen.add(task)
    return tuple(result)


def build_completion_reward_payload(
    episode: EpisodeRecord,
    trajectory: Trajectory,
    character: str,
) -> dict[str, Any]:
    """Build one request for all replies by a character in a trajectory."""

    target = _character(episode, character)
    completions = _trajectory_character_completions(trajectory, character)
    return {
        "schema_version": REWARD_REQUEST_SCHEMA_VERSION,
        "scope": "completion_local",
        "ids": {
            "episode_id": episode.episode_id,
            "group_id": trajectory.group_id,
            "trajectory_id": trajectory.trajectory_id,
            "character": target.name,
            "completion_ids": [completion.completion_id for completion in completions],
        },
        "template_args": {
            "character": target.name,
            "profile": target.profile,
            "plot": episode.plot,
        },
        "completion_texts": [completion.text for completion in completions],
        "utters": _indexed_utters([completion.text for completion in completions]),
        "input": {
            "utterances": _history_text(_full_trajectory_history(episode, trajectory)),
        },
    }


def build_trajectory_reward_payload(
    episode: EpisodeRecord,
    trajectory: Trajectory,
    character: str,
) -> dict[str, Any]:
    """Build one target-character task payload over the complete trajectory."""

    target = _character(episode, character)
    tasks = _merged_tasks(episode.shared_tasks, target.private_tasks)
    full_history = [*(_turn_payload(turn) for turn in episode.dialogue)]
    full_history.extend(_turn_payload(turn) for turn in trajectory.turns)
    return {
        "schema_version": REWARD_REQUEST_SCHEMA_VERSION,
        "scope": "trajectory_role",
        "ids": {
            "episode_id": episode.episode_id,
            "group_id": trajectory.group_id,
            "trajectory_id": trajectory.trajectory_id,
            "character": target.name,
        },
        "template_args": {
            "character": target.name,
            "plot": episode.plot,
            "tasks": json.dumps(tasks, ensure_ascii=False),
        },
        "tasks": list(tasks),
        "input": {
            "utterances": _history_text(full_history),
        },
    }


def _proxy_field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _proxy_episode_value(episode: Mapping[str, Any] | Any, name: str, default: Any = None) -> Any:
    if isinstance(episode, Mapping):
        return episode.get(name, default)
    return getattr(episode, name, default)


def _proxy_characters(episode: Mapping[str, Any] | Any) -> tuple[Mapping[str, Any], ...]:
    values = _proxy_episode_value(episode, "characters", ())
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise RewardResponseError("proxy episode must contain a character sequence")
    result: list[Mapping[str, Any]] = []
    for value in values:
        if isinstance(value, Mapping):
            result.append(value)
        else:
            dumped = getattr(value, "model_dump", None)
            if not callable(dumped):
                raise RewardResponseError("proxy character must be a mapping")
            raw = dumped(mode="json")
            if not isinstance(raw, Mapping):
                raise RewardResponseError("proxy character model_dump must be a mapping")
            result.append(raw)
    return tuple(result)


def build_proxy_completion_reward_payload(
    episode: Mapping[str, Any] | Any,
    trajectory: Any,
    character: str,
) -> dict[str, Any]:
    """Build one role-completion request from a text-only proxy view.

    This boundary intentionally accepts no token fields and never constructs a
    :class:`CompletionTrace`; exact token metadata remains the responsibility
    of AReaL's official export path.
    """

    characters = _proxy_characters(episode)
    character_name = str(character)
    target = next(
        (
            item
            for item in characters
            if str(item.get("name", item.get("character", ""))) == character_name
        ),
        None,
    )
    if target is None or character_name.casefold() == "environment":
        raise RewardResponseError(f"proxy completion character {character_name!r} is invalid")
    completions = tuple(_proxy_field(trajectory, "completions", ()))
    target_completions = tuple(
        item
        for item in sorted(
            completions,
            key=lambda value: int(_proxy_field(value, "turn_index", 0)),
        )
        if str(_proxy_field(item, "character", "")) == character_name
    )
    if not target_completions:
        raise RewardResponseError(
            f"proxy trajectory has no completion for character {character_name!r}"
        )
    history: list[dict[str, Any]] = []
    base_history = (
        _proxy_episode_value(episode, "dialogue", ())
        or _proxy_episode_value(episode, "history", ())
        or ()
    )
    for turn in base_history:
        history.append(
            {
                "character": str(
                    _proxy_field(
                        turn,
                        "character",
                        _proxy_field(turn, "speaker", ""),
                    )
                ),
                "content": str(_proxy_field(turn, "content", "")),
            }
        )
    for item in sorted(completions, key=lambda value: int(_proxy_field(value, "turn_index", 0))):
        history.append(
            {
                "character": str(_proxy_field(item, "character", "")),
                "content": str(_proxy_field(item, "text", "")),
                "completion_id": str(_proxy_field(item, "completion_id", "")),
            }
        )
    return {
        "schema_version": REWARD_REQUEST_SCHEMA_VERSION,
        "scope": "completion_local",
        "ids": {
            "episode_id": str(_proxy_episode_value(episode, "episode_id", "")),
            "group_id": str(_proxy_field(trajectory, "group_id", "")),
            "trajectory_id": str(_proxy_field(trajectory, "trajectory_id", "")),
            "character": character_name,
            "completion_ids": [
                str(_proxy_field(item, "completion_id", "")) for item in target_completions
            ],
        },
        "template_args": {
            "character": character_name,
            "profile": str(target.get("profile", "")),
            "plot": str(_proxy_episode_value(episode, "plot", "")),
        },
        "completion_texts": [str(_proxy_field(item, "text", "")) for item in target_completions],
        "utters": _indexed_utters(
            [str(_proxy_field(item, "text", "")) for item in target_completions]
        ),
        "input": {
            "utterances": _history_text(history),
        },
    }


def build_proxy_trajectory_reward_payload(
    episode: Mapping[str, Any] | Any,
    trajectory: Any,
    character: str,
) -> dict[str, Any]:
    """Build one role-task payload from a complete text-only proxy trajectory."""

    characters = _proxy_characters(episode)
    target = next(
        (
            item
            for item in characters
            if str(item.get("name", item.get("character", ""))) == character
        ),
        None,
    )
    if target is None or character.casefold() == "environment":
        raise RewardResponseError(f"proxy trajectory character {character!r} is invalid")
    shared_tasks = tuple(
        _proxy_episode_value(
            episode,
            "shared_tasks",
            _proxy_episode_value(episode, "tasks", ()),
        )
        or ()
    )
    tasks = _merged_tasks(shared_tasks, tuple(target.get("private_tasks", ())))
    history: list[dict[str, Any]] = []
    base_history = (
        _proxy_episode_value(episode, "dialogue", ())
        or _proxy_episode_value(episode, "history", ())
        or ()
    )
    for turn in (*base_history, *_proxy_field(trajectory, "turns", ())):
        history.append(
            {
                "character": str(
                    _proxy_field(
                        turn,
                        "character",
                        _proxy_field(turn, "speaker", ""),
                    )
                ),
                "content": str(_proxy_field(turn, "content", "")),
            }
        )
    return {
        "schema_version": REWARD_REQUEST_SCHEMA_VERSION,
        "scope": "trajectory_role",
        "ids": {
            "episode_id": str(_proxy_episode_value(episode, "episode_id", "")),
            "group_id": str(_proxy_field(trajectory, "group_id", "")),
            "trajectory_id": str(_proxy_field(trajectory, "trajectory_id", "")),
            "character": character,
        },
        "template_args": {
            "character": character,
            "plot": str(_proxy_episode_value(episode, "plot", "")),
            "tasks": json.dumps(tasks, ensure_ascii=False),
        },
        "tasks": list(tasks),
        "input": {
            "utterances": _history_text(history),
        },
    }
