"""The agent loop: ask the model, run its tool calls, repeat until it submits."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from . import prompts
from .model import ChatModel, ModelError, ModelResponse, Pricing, ToolCall
from .sandbox import Sandbox, SandboxError
from .tools import TOOLS, format_observation

log = logging.getLogger("patchloop")

# Stage everything (new files included) and print the diff. Stderr is dropped so
# warnings can never end up inside the patch.
PATCH_COMMAND = (
    "git -c core.fileMode=false add -A >/dev/null 2>&1 && "
    "git -c core.fileMode=false diff --cached --no-color --no-ext-diff 2>/dev/null"
)


@dataclass
class AgentConfig:
    step_limit: int = 100
    cost_limit: float = 3.0  # US dollars; 0 disables the limit
    command_timeout: float = 120.0
    max_output_chars: int = 10_000
    max_turns_without_tools: int = 3


@dataclass
class StepRecord:
    index: int
    tool_calls: list[dict[str, Any]]
    input_tokens: int
    output_tokens: int
    cost: float
    model_seconds: float
    tool_seconds: float


@dataclass
class AgentResult:
    exit_status: str  # submitted | step_limit | cost_limit | no_tool_calls | model_error | sandbox_error
    patch: str
    steps: int
    input_tokens: int
    output_tokens: int
    cost: float
    duration_seconds: float
    summary: str | None = None
    error: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    step_records: list[StepRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Agent:
    def __init__(
        self,
        model: ChatModel,
        sandbox: Sandbox,
        config: AgentConfig | None = None,
        pricing: Pricing | None = None,
        system_prompt: str = prompts.SYSTEM_PROMPT,
    ) -> None:
        self.model = model
        self.sandbox = sandbox
        self.config = config or AgentConfig()
        self.pricing = pricing or Pricing()
        self.system_prompt = system_prompt

    def run(self, task: str) -> AgentResult:
        cfg = self.config
        started = time.monotonic()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]
        records: list[StepRecord] = []
        totals = {"input": 0, "output": 0, "cost": 0.0}
        exit_status, summary, error = "step_limit", None, None
        turns_without_tools = 0

        while len(records) < cfg.step_limit:
            if cfg.cost_limit > 0 and totals["cost"] >= cfg.cost_limit:
                exit_status = "cost_limit"
                break

            t0 = time.monotonic()
            try:
                response = self.model.complete(messages, TOOLS)
            except ModelError as exc:
                exit_status, error = "model_error", str(exc)
                break
            model_seconds = time.monotonic() - t0

            cost = self._cost(response)
            totals["input"] += response.input_tokens
            totals["output"] += response.output_tokens
            totals["cost"] += cost
            messages.append(response.message)

            t1 = time.monotonic()
            submitted = False
            if not response.tool_calls:
                turns_without_tools += 1
                messages.append({"role": "user", "content": prompts.NO_TOOL_CALL_REMINDER})
            else:
                turns_without_tools = 0
                try:
                    submitted, summary = self._run_tool_calls(response.tool_calls, messages)
                except SandboxError as exc:
                    exit_status, error = "sandbox_error", str(exc)
                    records.append(self._record(len(records) + 1, response, cost, model_seconds, t1))
                    break

            records.append(self._record(len(records) + 1, response, cost, model_seconds, t1))
            log.info(
                "step %d: %s | tokens %d/%d | cost $%.4f",
                len(records),
                ", ".join(c.name for c in response.tool_calls) or "no tool call",
                response.input_tokens,
                response.output_tokens,
                totals["cost"],
            )

            if submitted:
                exit_status = "submitted"
                break
            if turns_without_tools >= cfg.max_turns_without_tools:
                exit_status = "no_tool_calls"
                break

        patch, patch_error = self._extract_patch()
        if patch_error and not error:
            error = patch_error

        return AgentResult(
            exit_status=exit_status,
            patch=patch,
            steps=len(records),
            input_tokens=totals["input"],
            output_tokens=totals["output"],
            cost=totals["cost"],
            duration_seconds=time.monotonic() - started,
            summary=summary,
            error=error,
            messages=messages,
            step_records=records,
        )

    def _cost(self, response: ModelResponse) -> float:
        if response.reported_cost is not None:
            return response.reported_cost
        return self.pricing.cost(response.input_tokens, response.output_tokens)

    def _run_tool_calls(
        self, calls: list[ToolCall], messages: list[dict[str, Any]]
    ) -> tuple[bool, str | None]:
        """Run calls in order. Every call gets a tool message back, as the API requires."""
        submitted, summary = False, None
        for call in calls:
            if submitted:
                content = "Skipped: the task was already submitted."
            elif call.parse_error:
                content = f"Error: {call.parse_error}. Please call the tool again with valid JSON arguments."
            elif call.name == "bash":
                command = call.arguments.get("command")
                if not isinstance(command, str) or not command.strip():
                    content = "Error: the bash tool needs a non-empty 'command' string."
                else:
                    result = self.sandbox.run(command, timeout=self.config.command_timeout)
                    content = format_observation(result, self.config.max_output_chars)
            elif call.name == "submit":
                submitted = True
                raw = call.arguments.get("summary")
                summary = raw if isinstance(raw, str) else None
                content = "Submitted."
            else:
                content = f"Error: unknown tool '{call.name}'. Available tools: bash, submit."
            messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
        return submitted, summary

    def _extract_patch(self) -> tuple[str, str | None]:
        try:
            result = self.sandbox.run(PATCH_COMMAND, timeout=120)
        except SandboxError as exc:
            return "", f"could not read the patch: {exc}"
        if result.timed_out or result.exit_code != 0:
            return "", f"could not read the patch (exit code {result.exit_code})"
        return result.output, None

    @staticmethod
    def _record(
        index: int, response: ModelResponse, cost: float, model_seconds: float, tools_started: float
    ) -> StepRecord:
        return StepRecord(
            index=index,
            tool_calls=[{"name": c.name, "arguments": c.arguments} for c in response.tool_calls],
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost=cost,
            model_seconds=round(model_seconds, 3),
            tool_seconds=round(time.monotonic() - tools_started, 3),
        )
