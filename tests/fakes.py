"""Test doubles shared by the test modules."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from patchloop.model import ModelError, ModelResponse, ToolCall


def call(name, call_id=None, **arguments):
    return ToolCall(
        id=call_id or f"call_{name}",
        name=name,
        arguments=arguments,
        raw_arguments=json.dumps(arguments),
    )


def reply(*tool_calls, content="", input_tokens=100, output_tokens=20, cost=None):
    message = {"role": "assistant", "content": (content or None) if tool_calls else content}
    if tool_calls:
        message["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.raw_arguments}}
            for c in tool_calls
        ]
    return ModelResponse(
        message=message,
        content=content,
        tool_calls=list(tool_calls),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reported_cost=cost,
    )


class ScriptedModel:
    """Returns a fixed list of responses; repeats the last one if it runs out."""

    name = "scripted"

    def __init__(self, responses, repeat_last=False):
        self.responses = list(responses)
        self.repeat_last = repeat_last
        self.seen_messages = []

    def complete(self, messages, tools):
        self.seen_messages.append([dict(m) for m in messages])
        if not self.responses:
            raise AssertionError("ScriptedModel ran out of responses")
        item = self.responses[0] if (self.repeat_last and len(self.responses) == 1) else self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_git_repo(path: Path, files: dict[str, str]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    git = ["git", "-c", "user.email=dev@example.com", "-c", "user.name=dev"]
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run([*git, "add", "-A"], cwd=path, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "init"], cwd=path, check=True)
    return path


BUGGY_CALC = "def add(a, b):\n    return a - b\n"

__all__ = ["call", "reply", "ScriptedModel", "make_git_repo", "BUGGY_CALC", "ModelError"]
