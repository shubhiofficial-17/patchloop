"""A small client for any OpenAI-compatible chat completions API.

It only uses the standard library, so it works with OpenAI, OpenRouter, local
servers (vLLM, Ollama) or your own gateway, as long as the endpoint speaks the
`/chat/completions` format with tool calling.
"""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


class ModelError(RuntimeError):
    """The model API failed in a way that retrying did not fix."""


@dataclass
class Pricing:
    """Price in US dollars per million tokens."""

    input_per_million: float = 0.0
    output_per_million: float = 0.0

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_million + output_tokens * self.output_per_million
        ) / 1_000_000

    @property
    def is_set(self) -> bool:
        return self.input_per_million > 0 or self.output_per_million > 0


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str
    parse_error: str | None = None


@dataclass
class ModelResponse:
    message: dict[str, Any]  # assistant message to append to the history as-is
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    reported_cost: float | None = None  # some providers return the exact cost


class ChatModel(Protocol):
    name: str

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelResponse: ...


def _text_from_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # list of content parts
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)


def _parse_arguments(raw: Any) -> tuple[dict[str, Any], str, str | None]:
    if isinstance(raw, dict):
        return raw, json.dumps(raw), None
    raw_text = raw if isinstance(raw, str) and raw.strip() else "{}"
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return {}, raw_text, f"arguments are not valid JSON ({exc.msg})"
    if not isinstance(parsed, dict):
        return {}, raw_text, "arguments must be a JSON object"
    return parsed, raw_text, None


def parse_response(data: Mapping[str, Any]) -> ModelResponse:
    choices = data.get("choices") or []
    if not choices:
        raise ModelError(f"response has no choices: {json.dumps(data)[:500]}")
    msg = choices[0].get("message") or {}
    content = _text_from_content(msg.get("content"))

    tool_calls: list[ToolCall] = []
    for i, raw in enumerate(msg.get("tool_calls") or []):
        fn = raw.get("function") or {}
        args, raw_args, error = _parse_arguments(fn.get("arguments"))
        tool_calls.append(
            ToolCall(
                id=raw.get("id") or f"call_{i}",
                name=fn.get("name") or "",
                arguments=args,
                raw_arguments=raw_args,
                parse_error=error,
            )
        )

    # With tool calls the API accepts null content; without them it must be a string.
    message: dict[str, Any] = {"role": "assistant", "content": (content or None) if tool_calls else content}
    if tool_calls:
        message["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.raw_arguments}}
            for c in tool_calls
        ]

    usage = data.get("usage") or {}
    reported = usage.get("cost")
    return ModelResponse(
        message=message,
        content=content,
        tool_calls=tool_calls,
        input_tokens=int(usage.get("prompt_tokens") or 0),
        output_tokens=int(usage.get("completion_tokens") or 0),
        reported_cost=float(reported) if isinstance(reported, (int, float)) else None,
    )


class OpenAICompatibleModel:
    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        headers: Mapping[str, str] | None = None,
        temperature: float | None = None,
        extra_body: Mapping[str, Any] | None = None,
        timeout: float = 300.0,
        max_retries: int = 5,
        backoff_seconds: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.name = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.headers = dict(headers or {})
        self.temperature = temperature
        self.extra_body = dict(extra_body or {})
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self._sleep = sleep

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelResponse:
        payload: dict[str, Any] = {"model": self.name, "messages": messages}
        if tools:
            payload["tools"] = tools
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        payload.update(self.extra_body)
        return parse_response(self._post("/chat/completions", payload))

    def _delay(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(float(retry_after), 120.0)
            except ValueError:
                pass
        return min(self.backoff_seconds * 2**attempt, 60.0) + random.uniform(0, 1)

    def _post(self, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        url = self.base_url + path
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", **self.headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:2000]
                if exc.code in RETRYABLE_STATUS and attempt < self.max_retries:
                    self._sleep(self._delay(attempt, exc.headers.get("Retry-After")))
                    continue
                raise ModelError(f"HTTP {exc.code} from {url}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                if attempt < self.max_retries:
                    self._sleep(self._delay(attempt, None))
                    continue
                raise ModelError(f"could not reach {url}: {exc}") from exc
            except json.JSONDecodeError as exc:
                raise ModelError(f"{url} returned invalid JSON") from exc
        raise ModelError("unreachable")  # pragma: no cover
