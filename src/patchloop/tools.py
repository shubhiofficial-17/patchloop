"""Tool schemas the model can call, and how their results are shown back to it."""

from __future__ import annotations

from .sandbox import CommandResult

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Run one shell command in the repository root and return its combined stdout and stderr. "
            "Each call starts a fresh shell, so `cd` and `export` do not persist between calls."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to run."},
            },
            "required": ["command"],
        },
    },
}

SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit",
        "description": (
            "Finish the task. The current state of the repository is taken as your final patch, "
            "so only call this once the fix is in place."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "One or two sentences on what you changed."},
            },
            "required": [],
        },
    },
}

TOOLS = [BASH_TOOL, SUBMIT_TOOL]


def truncate(text: str, limit: int) -> str:
    """Keep the start and the end of long output, where the useful lines usually are."""
    if limit <= 0 or len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    omitted = len(text) - limit
    return (
        f"{text[:head]}\n\n[... {omitted} characters omitted. "
        f"Use grep, head, tail or sed -n to look at a smaller part ...]\n\n{text[-tail:]}"
    )


def format_observation(result: CommandResult, limit: int) -> str:
    if result.timed_out:
        status = "The command timed out and was stopped."
    else:
        status = f"Exit code: {result.exit_code}"
    body = truncate(result.output, limit) if result.output else "(no output)"
    return f"{status}\n{body}"
