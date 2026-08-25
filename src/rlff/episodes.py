"""Offline episode loading, deterministic grouping, and target-role prompts."""

from __future__ import annotations

import json
import random
import threading
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypeAlias, overload

from pydantic import ValidationError

from .config import DEFAULT_PROMPT_TEMPLATE
from .contracts import (
    RENDERER_VERSION,
    CharacterSpec,
    ChatMessage,
    EpisodeRecord,
    EpisodeSample,
    GroupedEpisodeSamples,
    PromptProjection,
    PromptRenderSpec,
    stable_fingerprint,
    stable_id,
)

_RENDER_LOCK = threading.RLock()
_DEFAULT_TEMPLATE_ID: Final = "default"
TemplateDefinition: TypeAlias = tuple[str, str]


class EpisodeLoadError(ValueError):
    """Line-aware error raised while loading canonical episode JSONL."""


@dataclass(frozen=True)
class EpisodeDataset(Sequence[EpisodeRecord]):
    """Validated records and their deterministic ordered dataset fingerprint."""

    records: tuple[EpisodeRecord, ...]
    fingerprint: str
    source_path: str | None = None

    def __len__(self) -> int:
        return len(self.records)

    @overload
    def __getitem__(self, index: int) -> EpisodeRecord: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[EpisodeRecord, ...]: ...

    def __getitem__(self, index: int | slice) -> EpisodeRecord | tuple[EpisodeRecord, ...]:
        result = self.records[index]
        return result

    def __iter__(self) -> Iterator[EpisodeRecord]:
        return iter(self.records)


def _dataset_fingerprint(records: Sequence[EpisodeRecord]) -> str:
    return stable_fingerprint(
        {
            "schema_version": "rlff.episode-dataset.v1",
            "episode_fingerprints": [record.fingerprint for record in records],
        }
    )


def load_episode_jsonl(
    path: str | Path,
    *,
    limit: int | None = None,
) -> EpisodeDataset:
    """Load and validate canonical UTF-8/UTF-8-SIG episode JSONL.

    Blank lines are rejected deliberately: a canonical dataset fingerprint
    must not depend on silently discarded source data.  Every parse and schema
    failure includes its source path and line number.
    """

    source = Path(path)
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be a positive integer when provided")
    try:
        lines = source.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise EpisodeLoadError(f"{source}: cannot read JSONL: {exc}") from exc

    records: list[EpisodeRecord] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise EpisodeLoadError(f"{source}:{line_number}: blank JSONL line is not allowed")
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EpisodeLoadError(
                f"{source}:{line_number}: invalid JSON at column {exc.colno}: {exc.msg}"
            ) from exc
        if not isinstance(raw, dict):
            raise EpisodeLoadError(f"{source}:{line_number}: record must be a JSON object")
        try:
            record = EpisodeRecord.model_validate(raw)
        except ValidationError as exc:
            raise EpisodeLoadError(f"{source}:{line_number}: invalid episode: {exc}") from exc
        if record.episode_id is None:  # defensive; EpisodeRecord derives it
            raise EpisodeLoadError(f"{source}:{line_number}: episode_id could not be derived")
        if record.episode_id in seen_ids:
            raise EpisodeLoadError(
                f"{source}:{line_number}: duplicate episode_id {record.episode_id!r}"
            )
        seen_ids.add(record.episode_id)
        records.append(record)
        if limit is not None and len(records) >= limit:
            break

    validated = tuple(records)
    if not validated:
        raise EpisodeLoadError(f"{source}: episode JSONL dataset is empty")
    return EpisodeDataset(
        records=validated,
        fingerprint=_dataset_fingerprint(validated),
        source_path=str(source),
    )


def load_episodes(path: str | Path, *, limit: int | None = None) -> EpisodeDataset:
    """Compatibility spelling for :func:`load_episode_jsonl`."""

    return load_episode_jsonl(path, limit=limit)


def _stable_integer(*parts: object) -> int:
    # Keep deterministic seeds inside the non-negative signed int64 domain.
    # Hugging Face Datasets/Arrow serializes these values through a C ``long``;
    # an unsigned 64-bit digest prefix can therefore overflow on Linux before
    # rollout starts.  Masking the sign bit retains 63 bits of entropy while
    # remaining valid for Arrow and downstream inference runtimes.
    return int(stable_fingerprint(parts)[:16], 16) & ((1 << 63) - 1)


def _normalise_templates(
    templates: Iterable[TemplateDefinition] | None,
) -> tuple[TemplateDefinition, ...]:
    if templates is None:
        return ((_DEFAULT_TEMPLATE_ID, DEFAULT_PROMPT_TEMPLATE),)
    result = tuple(templates)
    if not result:
        raise ValueError("at least one prompt template is required")
    ids: set[str] = set()
    for template_id, template in result:
        if not template_id.strip() or not template.strip():
            raise ValueError("prompt template IDs and contents must be non-empty")
        if template_id in ids:
            raise ValueError(f"duplicate prompt template ID {template_id!r}")
        ids.add(template_id)
    return result


def build_episode_group(
    record: EpisodeRecord,
    *,
    group_size: int,
    base_seed: int = 0,
    sampling_iteration: int = 0,
    templates: Iterable[TemplateDefinition] | None = None,
) -> GroupedEpisodeSamples:
    """Replicate one episode into a deterministic same-prompt group."""

    if type(group_size) is not int or group_size <= 0:
        raise ValueError("group_size must be a positive integer")
    if type(base_seed) is not int or base_seed < 0:
        raise ValueError("base_seed must be a non-negative integer")
    if type(sampling_iteration) is not int or sampling_iteration < 0:
        raise ValueError("sampling_iteration must be a non-negative integer")
    definitions = _normalise_templates(templates)
    group_id = stable_id(
        "group",
        record.fingerprint,
        group_size,
        base_seed,
        sampling_iteration,
    )
    template_index = _stable_integer("template", group_id) % len(definitions)
    template_id, template = definitions[template_index]
    render_seed = _stable_integer("render", group_id)
    render = PromptRenderSpec(
        template_id=template_id,
        template=template,
        renderer_version=RENDERER_VERSION,
        render_seed=render_seed,
    )

    episode_id = record.episode_id
    if episode_id is None:  # defensive; EpisodeRecord derives it
        raise ValueError("episode must have a deterministic episode_id")
    samples: list[EpisodeSample] = []
    for sample_index in range(group_size):
        seed = _stable_integer("sampling", group_id, sample_index)
        trajectory_id = stable_id(
            "trajectory",
            group_id,
            sampling_iteration,
            sample_index,
        )
        samples.append(
            EpisodeSample(
                episode_id=episode_id,
                group_id=group_id,
                trajectory_id=trajectory_id,
                sample_index=sample_index,
                sampling_iteration=sampling_iteration,
                seed=seed,
                render=render,
                episode=record,
            )
        )
    return GroupedEpisodeSamples(
        group_id=group_id,
        episode_id=episode_id,
        episode=record,
        samples=tuple(samples),
        group_size=group_size,
        sampling_iteration=sampling_iteration,
    )


def group_episode(
    record: EpisodeRecord,
    group_size: int,
    *,
    base_seed: int = 0,
    sampling_iteration: int = 0,
    templates: Iterable[TemplateDefinition] | None = None,
) -> GroupedEpisodeSamples:
    """Positional convenience wrapper for :func:`build_episode_group`."""

    return build_episode_group(
        record,
        group_size=group_size,
        base_seed=base_seed,
        sampling_iteration=sampling_iteration,
        templates=templates,
    )


def build_episode_groups(
    records: Iterable[EpisodeRecord],
    *,
    group_size: int,
    base_seed: int = 0,
    sampling_iteration: int = 0,
    templates: Iterable[TemplateDefinition] | None = None,
) -> tuple[GroupedEpisodeSamples, ...]:
    """Build groups in stable source order."""

    return tuple(
        build_episode_group(
            record,
            group_size=group_size,
            base_seed=base_seed,
            sampling_iteration=sampling_iteration,
            templates=templates,
        )
        for record in records
    )


@contextmanager
def _isolated_renderer_seed(seed: int) -> Iterator[None]:
    """Seed the legacy module-global renderer without leaking RNG state."""

    if type(seed) is not int or seed < 0:
        raise ValueError("render_seed must be a non-negative integer")
    with _RENDER_LOCK:
        state = random.getstate()
        random.seed(seed)
        try:
            yield
        finally:
            random.setstate(state)


def _render_template(template: str, values: Mapping[str, str], seed: int) -> str:
    # Importing the existing parser here keeps episode validation usable in
    # environments where only the RLFF package (not its extraction scripts) is
    # installed, while preserving exactly its grammar and semantics locally.
    try:
        from script.system_prompt_renderer import render_system_prompt
    except ImportError as exc:  # pragma: no cover - packaging failure
        raise RuntimeError("system prompt renderer is unavailable") from exc
    with _isolated_renderer_seed(seed):
        return render_system_prompt(template, values)


def _character_map(record: EpisodeRecord) -> dict[str, CharacterSpec]:
    return {character.name: character for character in record.characters}


def project_target_prompt(
    record: EpisodeRecord,
    target_character: str,
    *,
    render: PromptRenderSpec | None = None,
    template_id: str = _DEFAULT_TEMPLATE_ID,
    template: str = DEFAULT_PROMPT_TEMPLATE,
    render_seed: int = 0,
) -> PromptProjection:
    """Project one episode into the target character's SFT-compatible view."""

    characters = _character_map(record)
    target = characters.get(target_character)
    if target is None:
        raise ValueError(f"target character {target_character!r} is not registered")
    if render is None:
        render = PromptRenderSpec(
            template_id=template_id,
            template=template,
            renderer_version=RENDERER_VERSION,
            render_seed=render_seed,
        )
    else:
        template_id = render.template_id
        template = render.template
        render_seed = render.render_seed

    # CharacterSpec is deliberately a Pydantic model; this local attribute
    # access keeps the projection free of untyped dict protocols.
    private_tasks = target.private_tasks
    profile = target.profile
    tasks = ";".join((*record.shared_tasks, *private_tasks))
    system = _render_template(
        template,
        {
            "character": target.name,
            "profile": profile,
            "plot": record.plot,
            "tasks": tasks,
        },
        render_seed,
    )

    projected: list[ChatMessage] = [ChatMessage(role="system", content=system)]
    for turn in record.dialogue:
        if turn.character == target.name:
            role: Literal["user", "assistant"] = "assistant"
            content = turn.content
        else:
            role = "user"
            content = f"{turn.character}:{turn.content}"
        if projected and projected[-1].role == role:
            previous = projected[-1]
            projected[-1] = ChatMessage(
                role=role,
                content=f"{previous.content}\n{content}",
            )
        else:
            projected.append(ChatMessage(role=role, content=content))
    return PromptProjection(
        target_character=target.name,
        messages=tuple(projected),
        render=render,
    )


def project_sample_prompt(
    sample: EpisodeSample,
    target_character: str,
) -> PromptProjection:
    """Project a grouped sample using only its shared render specification."""

    return project_target_prompt(
        sample.episode,
        target_character,
        render=sample.render,
    )


def build_target_prompt(
    record: EpisodeRecord,
    target_character: str,
    *,
    render: PromptRenderSpec | None = None,
    template_id: str = _DEFAULT_TEMPLATE_ID,
    template: str = DEFAULT_PROMPT_TEMPLATE,
    render_seed: int = 0,
) -> PromptProjection:
    """Compatibility spelling for :func:`project_target_prompt`."""

    return project_target_prompt(
        record,
        target_character,
        render=render,
        template_id=template_id,
        template=template,
        render_seed=render_seed,
    )


__all__ = [
    "EpisodeDataset",
    "EpisodeLoadError",
    "TemplateDefinition",
    "build_episode_group",
    "build_episode_groups",
    "build_target_prompt",
    "group_episode",
    "load_episode_jsonl",
    "load_episodes",
    "project_sample_prompt",
    "project_target_prompt",
]
