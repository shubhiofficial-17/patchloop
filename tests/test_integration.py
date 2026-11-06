"""The real HTTP model client driving the agent loop against a fake API."""

import json

from fakes import BUGGY_CALC, make_git_repo

from patchloop.loop import Agent
from patchloop.model import OpenAICompatibleModel, Pricing
from patchloop.sandbox import LocalSandbox


def tool_reply(call_id, name, **arguments):
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": call_id, "type": "function",
                         "function": {"name": name, "arguments": json.dumps(arguments)}}
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 50},
    }


def test_agent_over_http(server, tmp_path):
    repo = make_git_repo(tmp_path / "repo", {"calc.py": BUGGY_CALC})
    server["queue"] += [
        (200, tool_reply("c1", "bash", command="grep -n 'a - b' calc.py")),
        (429, {"error": "rate limited"}),
        (200, tool_reply("c2", "bash", command="sed -i 's/a - b/a + b/' calc.py")),
        (200, tool_reply("c3", "submit", summary="fixed add")),
    ]
    model = OpenAICompatibleModel("m", base_url=server["url"], api_key="k", sleep=lambda s: None)
    result = Agent(model, LocalSandbox(repo), pricing=Pricing(1.0, 4.0)).run("fix add")

    assert result.exit_status == "submitted"
    assert "+    return a + b" in result.patch
    assert result.cost == 3 * (1000 * 1.0 + 50 * 4.0) / 1_000_000

    # The history sent back to the API pairs every tool call with a tool message.
    last_request = server["requests"][-1]["body"]["messages"]
    assert [m["role"] for m in last_request] == ["system", "user", "assistant", "tool", "assistant", "tool"]
    assert last_request[3]["tool_call_id"] == "c1"
    assert "2:    return a - b" in last_request[3]["content"]
    assert last_request[2]["tool_calls"][0]["function"]["name"] == "bash"
