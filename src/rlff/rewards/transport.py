"""HTTP transport, redaction, and optional LangSmith tracing.

This file is part of the deploy-only RLFF implementation. Keep changes here;
the repository-level src/rlff tree is intentionally not synchronized.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Mapping, Sequence
from typing import Any, cast

import aiohttp

from .transport_types import (
    RewardHTTPResponse,
    RewardTransport,
    RewardTransportCallable,
    RewardTransportResult,
)


def _json_payload(messages: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    return [dict(message) for message in messages]


def _redact(value: str, secrets: Sequence[str]) -> str:
    result = value
    for secret in secrets:
        if secret:
            result = result.replace(secret, "[REDACTED]")
    return result


def _response_from_transport(value: RewardTransportResult) -> RewardHTTPResponse:
    if isinstance(value, RewardHTTPResponse):
        return value
    if isinstance(value, bytes):
        return RewardHTTPResponse(status_code=200, text=value.decode("utf-8", "replace"))
    if isinstance(value, str):
        return RewardHTTPResponse(status_code=200, text=value)
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], int)
        and isinstance(value[1], str)
    ):
        return RewardHTTPResponse(status_code=value[0], text=value[1])
    if isinstance(value, Mapping):
        # A mapping with status/text is the convenient fake-transport shape;
        # any other mapping is treated as an already-decoded HTTP JSON body.
        if "status_code" in value or "status" in value or "text" in value:
            status = value.get("status_code", value.get("status", 200))
            text = value.get("text", "")
            if not isinstance(status, int) or not isinstance(text, str):
                raise TypeError("fake reward transport status_code/text have invalid types")
            return RewardHTTPResponse(status_code=status, text=text)
        return RewardHTTPResponse(status_code=200, text=json.dumps(value, ensure_ascii=False))
    status = getattr(value, "status_code", getattr(value, "status", None))
    text = getattr(value, "text", None)
    if isinstance(status, int) and isinstance(text, str):
        return RewardHTTPResponse(status_code=status, text=text)
    raise TypeError(f"unsupported reward transport result: {type(value).__name__}")


async def _default_reward_transport(
    *,
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout_seconds: float,
) -> RewardHTTPResponse:
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    async with (
        aiohttp.ClientSession(timeout=timeout) as session,
        session.post(url, headers=dict(headers), json=dict(payload)) as response,
    ):
        return RewardHTTPResponse(
            status_code=response.status,
            text=await response.text(),
            headers=dict(response.headers),
        )


async def _call_transport(
    transport: RewardTransportCallable | RewardTransport,
    *,
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout_seconds: float,
) -> RewardTransportResult:
    """Invoke the documented keyword transport and a few test-double shapes."""

    target: Any = transport
    if not callable(target):
        target = getattr(target, "request", None) or getattr(transport, "post", None)
    if target is None or not callable(target):
        raise TypeError("reward transport must be callable or expose request()")
    try:
        result = target(
            url=url,
            headers=headers,
            payload=payload,
            timeout_seconds=timeout_seconds,
        )
    except TypeError:
        try:
            parameters = inspect.signature(target).parameters
        except (TypeError, ValueError):
            raise
        kwargs: dict[str, Any] = {}
        if "url" in parameters:
            kwargs["url"] = url
        if "headers" in parameters:
            kwargs["headers"] = headers
        if "payload" in parameters:
            kwargs["payload"] = payload
        elif "json" in parameters:
            kwargs["json"] = payload
        if "timeout_seconds" in parameters:
            kwargs["timeout_seconds"] = timeout_seconds
        elif "timeout" in parameters:
            kwargs["timeout"] = timeout_seconds
        result = target(**kwargs) if kwargs else target(url, headers, payload, timeout_seconds)
    if inspect.isawaitable(result):
        return await cast(Awaitable[RewardTransportResult], result)
    return cast(RewardTransportResult, result)


class LangSmithTracer:
    """Small direct-LangSmith SDK adapter; no LangChain/LangGraph involved."""

    def __init__(self, *, project: str, api_key: str | None = None, client: Any = None) -> None:
        if client is None:
            try:
                from langsmith import Client
            except ImportError as exc:  # pragma: no cover - optional local install
                raise RuntimeError("LangSmith SDK is unavailable") from exc
            client = Client(api_key=api_key) if api_key else Client()
        self.client = client
        self.project = project

    def start(self, *, name: str, inputs: Mapping[str, Any]) -> str | None:
        try:
            run = self.client.create_run(
                name=name,
                run_type="llm",
                inputs=dict(inputs),
                project_name=self.project,
            )
            return str(getattr(run, "id", run)) if run is not None else None
        except Exception:
            return None

    def finish(
        self,
        run_id: str | None,
        *,
        outputs: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        if run_id is None:
            return
        try:
            kwargs: dict[str, Any] = {}
            if outputs is not None:
                kwargs["outputs"] = dict(outputs)
            if error is not None:
                kwargs["error"] = error
            self.client.update_run(run_id, **kwargs)
        except Exception:
            return
