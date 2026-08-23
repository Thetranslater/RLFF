"""Parse and render tree-structured RLFF system-prompt templates.

Grammar overview::

    document  := (text | random | rearrange | implicit_text)*
    text      := <text> raw_text </text>
    random    := <random [id=ID]> option (semicolon option)* </random>
    rearrange := <rearrange> item+ </rearrange>
    item      := <item> (text | random | rearrange | implicit_text)* </item>

Whitespace used only to format the markup is ignored. Non-whitespace raw text
at document/item level is accepted as an implicit ``text`` block so the current
templates can place variables such as ``{profile}`` directly inside an item.
Every block is trimmed at its boundaries and sibling results are joined with a
single newline.
"""

from __future__ import annotations

import random
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_BOUNDARY_WHITESPACE = " \t\r\n"
_VARIABLE_PATTERN = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_RANDOM_SEPARATOR = re.compile(r"[;；]")
_RANDOM_OPEN_PATTERN = re.compile(
    r"<random(?:\s+id\s*=\s*(?:\"([^\"]+)\"|'([^']+)'|([^\s>]+)))?\s*>"
)
_KNOWN_OPENINGS = ("<text>", "<random", "<rearrange>", "<item>")
_KNOWN_CLOSINGS = ("</text>", "</random>", "</rearrange>", "</item>")


class TemplateParseError(ValueError):
    """Raised when a prompt template does not satisfy the block grammar."""

    def __init__(self) -> None:
        super().__init__("无法解析template")


class MissingTemplateArgumentsError(ValueError):
    """Raised when args lacks one or more variables used by the template."""

    def __init__(self, missing: list[str]) -> None:
        self.missing = tuple(missing)
        super().__init__(f"args缺少参数: {', '.join(missing)}")


@dataclass(frozen=True)
class _TextNode:
    text: str


@dataclass(frozen=True)
class _RandomNode:
    options: tuple[str, ...]
    random_id: str | None


@dataclass(frozen=True)
class _ItemNode:
    children: tuple[_Node, ...]


@dataclass(frozen=True)
class _RearrangeNode:
    items: tuple[_ItemNode, ...]


@dataclass(frozen=True)
class _DocumentNode:
    children: tuple[_Node, ...]


_Node = _TextNode | _RandomNode | _ItemNode | _RearrangeNode | _DocumentNode


def _trim(value: str) -> str:
    return value.strip(_BOUNDARY_WHITESPACE)


class _Parser:
    def __init__(self, template: str) -> None:
        self.template = template
        self.position = 0

    def parse(self) -> _DocumentNode:
        children = self._parse_container(end_tag=None)
        if self.position != len(self.template):
            raise TemplateParseError()
        return _DocumentNode(tuple(children))

    def _parse_container(self, *, end_tag: str | None) -> list[_Node]:
        children: list[_Node] = []
        raw_start = self.position

        while self.position < len(self.template):
            if end_tag is not None and self.template.startswith(end_tag, self.position):
                self._append_implicit_text(children, raw_start, self.position)
                self.position += len(end_tag)
                return children

            if self.template.startswith("<text>", self.position):
                self._append_implicit_text(children, raw_start, self.position)
                children.append(self._parse_text())
                raw_start = self.position
                continue

            if self.template.startswith("<random", self.position):
                self._append_implicit_text(children, raw_start, self.position)
                children.append(self._parse_random())
                raw_start = self.position
                continue

            if self.template.startswith("<rearrange>", self.position):
                self._append_implicit_text(children, raw_start, self.position)
                children.append(self._parse_rearrange())
                raw_start = self.position
                continue

            if self.template.startswith("<item>", self.position):
                raise TemplateParseError()

            if any(
                self.template.startswith(closing, self.position)
                for closing in _KNOWN_CLOSINGS
            ):
                raise TemplateParseError()

            if self.template[self.position] == "<":
                raise TemplateParseError()

            self.position += 1

        if end_tag is not None:
            raise TemplateParseError()

        self._append_implicit_text(children, raw_start, self.position)
        return children

    def _append_implicit_text(
        self,
        children: list[_Node],
        start: int,
        end: int,
    ) -> None:
        content = _trim(self.template[start:end])
        if content:
            children.append(_TextNode(content))

    def _parse_text(self) -> _TextNode:
        self.position += len("<text>")
        closing_position = self.template.find("</text>", self.position)
        if closing_position < 0:
            raise TemplateParseError()

        content = _trim(self.template[self.position : closing_position])
        self.position = closing_position + len("</text>")
        if not content:
            raise TemplateParseError()
        return _TextNode(content)

    def _parse_random(self) -> _RandomNode:
        opening = _RANDOM_OPEN_PATTERN.match(self.template, self.position)
        if opening is None:
            raise TemplateParseError()

        random_id = next((value for value in opening.groups() if value is not None), None)
        self.position = opening.end()
        closing_position = self.template.find("</random>", self.position)
        if closing_position < 0:
            raise TemplateParseError()

        content = self.template[self.position : closing_position]
        if any(tag in content for tag in _KNOWN_OPENINGS + _KNOWN_CLOSINGS):
            raise TemplateParseError()

        options = tuple(_trim(option) for option in _RANDOM_SEPARATOR.split(content))
        if not options or any(not option for option in options):
            raise TemplateParseError()

        self.position = closing_position + len("</random>")
        return _RandomNode(options, random_id)

    def _parse_rearrange(self) -> _RearrangeNode:
        self.position += len("<rearrange>")
        items: list[_ItemNode] = []

        while self.position < len(self.template):
            self._skip_formatting_whitespace()
            if self.template.startswith("</rearrange>", self.position):
                self.position += len("</rearrange>")
                if not items:
                    raise TemplateParseError()
                return _RearrangeNode(tuple(items))
            if not self.template.startswith("<item>", self.position):
                raise TemplateParseError()
            items.append(self._parse_item())

        raise TemplateParseError()

    def _parse_item(self) -> _ItemNode:
        self.position += len("<item>")
        children = self._parse_container(end_tag="</item>")
        if not children:
            raise TemplateParseError()
        return _ItemNode(tuple(children))

    def _skip_formatting_whitespace(self) -> None:
        while (
            self.position < len(self.template)
            and self.template[self.position] in _BOUNDARY_WHITESPACE
        ):
            self.position += 1


def _walk(node: _Node):
    yield node
    if isinstance(node, (_DocumentNode, _ItemNode)):
        for child in node.children:
            yield from _walk(child)
    elif isinstance(node, _RearrangeNode):
        for item in node.items:
            yield from _walk(item)


def _get_argument(args: object, name: str) -> tuple[bool, Any]:
    if isinstance(args, Mapping):
        if name in args:
            return True, args[name]
        return False, None

    try:
        return True, getattr(args, name)
    except AttributeError:
        return False, None


def _collect_variable_names(tree: _DocumentNode) -> list[str]:
    names: set[str] = set()
    for node in _walk(tree):
        if isinstance(node, _TextNode):
            names.update(_VARIABLE_PATTERN.findall(node.text))
        elif isinstance(node, _RandomNode):
            for option in node.options:
                names.update(_VARIABLE_PATTERN.findall(option))
    return sorted(names)


def _validate_random_ids(tree: _DocumentNode) -> None:
    option_counts: dict[str, int] = {}
    for node in _walk(tree):
        if not isinstance(node, _RandomNode) or node.random_id is None:
            continue
        count = len(node.options)
        previous = option_counts.setdefault(node.random_id, count)
        if previous != count:
            raise TemplateParseError()


def _replace_variables(text: str, values: Mapping[str, str]) -> str:
    return _VARIABLE_PATTERN.sub(lambda match: values[match.group(1)], text)


def _join_block_results(results: list[str]) -> str:
    trimmed = [_trim(result) for result in results]
    return "\n".join(result for result in trimmed if result)


def _evaluate(
    node: _Node,
    values: Mapping[str, str],
    shared_choices: dict[str, int],
) -> str:
    if isinstance(node, _TextNode):
        return _trim(_replace_variables(node.text, values))

    if isinstance(node, _RandomNode):
        if node.random_id is None:
            index = random.randrange(len(node.options))
        elif node.random_id in shared_choices:
            index = shared_choices[node.random_id]
        else:
            index = random.randrange(len(node.options))
            shared_choices[node.random_id] = index
        return _trim(_replace_variables(node.options[index], values))

    if isinstance(node, (_DocumentNode, _ItemNode)):
        return _join_block_results(
            [_evaluate(child, values, shared_choices) for child in node.children]
        )

    if isinstance(node, _RearrangeNode):
        items = list(node.items)
        random.shuffle(items)
        return _join_block_results(
            [_evaluate(item, values, shared_choices) for item in items]
        )

    raise TypeError(f"Unsupported template node: {type(node)!r}")


def render_system_prompt(template: str, args: object) -> str:
    """Render a block template using variables from a mapping or object."""

    if not isinstance(template, str):
        raise TemplateParseError()

    tree = _Parser(template).parse()
    _validate_random_ids(tree)

    values: dict[str, str] = {}
    missing: list[str] = []
    for name in _collect_variable_names(tree):
        exists, value = _get_argument(args, name)
        if not exists:
            missing.append(name)
        else:
            values[name] = str(value)

    if missing:
        raise MissingTemplateArgumentsError(missing)

    return _trim(_evaluate(tree, values, shared_choices={}))


__all__ = [
    "MissingTemplateArgumentsError",
    "TemplateParseError",
    "render_system_prompt",
]
