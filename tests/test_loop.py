import pytest
from fakes import BUGGY_CALC, ModelError, ScriptedModel, call, make_git_repo, reply

from patchloop.loop import Agent, AgentConfig
from patchloop.model import Pricing
from patchloop.sandbox import LocalSandbox, SandboxError
from patchloop.tools import TOOLS, truncate


@pytest.fixture
def repo(tmp_path):
    return make_git_repo(tmp_path / "repo", {"calc.py": BUGGY_CALC, "README.md": "calc\n"})


def agent_for(repo, responses, **config):
    model = ScriptedModel(responses)
    agent = Agent(
        model,
        LocalSandbox(repo),
        config=AgentConfig(**config),
        pricing=Pricing(input_per_million=1.0, output_per_million=5.0),
    )
    return agent, model


def test_fixes_bug_and_returns_patch(repo):
    agent, model = agent_for(
        repo,
        [
            reply(call("bash", command="cat calc.py")),
            reply(call("bash", command="sed -i 's/a - b/a + b/' calc.py")),
            reply(call("submit", summary="add now adds")),
        ],
    )
    result = agent.run("add() subtracts instead of adding")

    assert result.exit_status == "submitted"
    assert result.summary == "add now adds"
    assert "-    return a - b" in result.patch
    assert "+    return a + b" in result.patch
    assert result.patch.endswith("\n")
    assert result.steps == 3
    assert result.input_tokens == 300 and result.output_tokens == 60
    assert result.cost == pytest.approx(3 * (100 * 1.0 + 20 * 5.0) / 1_000_000)

    # The model saw the file contents from the first command.
    tool_messages = [m for m in result.messages if m["role"] == "tool"]
    assert "return a - b" in tool_messages[0]["content"]
    assert tool_messages[0]["content"].startswith("Exit code: 0")
    assert model.seen_messages[0][0]["role"] == "system"


def test_new_files_are_included_and_tmp_files_are_not(repo):
    agent, _ = agent_for(
        repo,
        [
            reply(call("bash", command="echo 'x = 1' > newmod.py && echo scratch > /tmp/patchloop_scratch.txt")),
            reply(call("submit")),
        ],
    )
    result = agent.run("task")
    assert "newmod.py" in result.patch
    assert "scratch" not in result.patch


def test_every_tool_call_gets_a_response(repo):
    agent, _ = agent_for(
        repo,
        [
            reply(
                call("bash", call_id="a", command="echo one"),
                call("nope", call_id="b"),
                call("submit", call_id="c"),
                call("bash", call_id="d", command="echo never"),
            )
        ],
    )
    result = agent.run("task")
    tool_messages = {m["tool_call_id"]: m["content"] for m in result.messages if m["role"] == "tool"}
    assert set(tool_messages) == {"a", "b", "c", "d"}
    assert "one" in tool_messages["a"]
    assert "unknown tool" in tool_messages["b"]
    assert tool_messages["c"] == "Submitted."
    assert tool_messages["d"].startswith("Skipped")
    assert result.exit_status == "submitted"


def test_step_limit(repo):
    agent, _ = agent_for(repo, [reply(call("bash", command="echo hi"))] * 5, step_limit=3)
    result = agent.run("task")
    assert result.exit_status == "step_limit"
    assert result.steps == 3


def test_cost_limit_uses_reported_cost(repo):
    responses = [reply(call("bash", command="echo hi"), cost=0.4) for _ in range(10)]
    agent, _ = agent_for(repo, responses, cost_limit=1.0)
    result = agent.run("task")
    assert result.exit_status == "cost_limit"
    assert result.steps == 3  # 0.4 + 0.4 + 0.4 >= 1.0
    assert result.cost == pytest.approx(1.2)


def test_stops_after_turns_without_tools(repo):
    agent, _ = agent_for(repo, [reply(content="thinking...")] * 5, max_turns_without_tools=2)
    result = agent.run("task")
    assert result.exit_status == "no_tool_calls"
    assert result.steps == 2
    reminders = [m for m in result.messages if m["role"] == "user"][1:]
    assert len(reminders) == 2


def test_model_error_still_returns_work_so_far(repo):
    agent, _ = agent_for(
        repo,
        [reply(call("bash", command="sed -i 's/a - b/a + b/' calc.py")), ModelError("HTTP 400: context too long")],
    )
    result = agent.run("task")
    assert result.exit_status == "model_error"
    assert "context too long" in result.error
    assert "+    return a + b" in result.patch


def test_bad_arguments_are_reported_to_the_model(repo):
    bad = call("bash")
    bad.parse_error = "arguments are not valid JSON"
    agent, _ = agent_for(repo, [reply(bad), reply(call("bash", command="")), reply(call("submit"))])
    result = agent.run("task")
    tool_messages = [m["content"] for m in result.messages if m["role"] == "tool"]
    assert "valid JSON" in tool_messages[0]
    assert "non-empty 'command'" in tool_messages[1]
    assert result.exit_status == "submitted"


def test_long_output_is_truncated(repo):
    agent, _ = agent_for(
        repo,
        [reply(call("bash", command="python3 -c \"print('x' * 50000)\"")), reply(call("submit"))],
        max_output_chars=1000,
    )
    result = agent.run("task")
    observation = [m["content"] for m in result.messages if m["role"] == "tool"][0]
    assert len(observation) < 1300
    assert "characters omitted" in observation


def test_command_timeout_is_reported(repo):
    agent, _ = agent_for(
        repo, [reply(call("bash", command="sleep 10")), reply(call("submit"))], command_timeout=1
    )
    result = agent.run("task")
    observation = [m["content"] for m in result.messages if m["role"] == "tool"][0]
    assert "timed out" in observation


def test_sandbox_error_stops_the_run(repo):
    class BrokenSandbox(LocalSandbox):
        def run(self, command, timeout=None):
            raise SandboxError("container is gone")

    model = ScriptedModel([reply(call("bash", command="ls"))])
    result = Agent(model, BrokenSandbox(repo)).run("task")
    assert result.exit_status == "sandbox_error"
    assert result.patch == ""
    assert "container is gone" in result.error


def test_result_is_serialisable(repo):
    import json

    agent, _ = agent_for(repo, [reply(call("submit"))])
    data = agent.run("task").to_dict()
    json.dumps(data)
    assert data["step_records"][0]["tool_calls"] == [{"name": "submit", "arguments": {}}]


def test_truncate_keeps_head_and_tail():
    text = "A" * 100 + "B" * 100
    out = truncate(text, 50)
    assert out.startswith("A" * 25) and out.endswith("B" * 25)
    assert truncate("short", 50) == "short"


def test_tool_schemas_are_well_formed():
    names = [t["function"]["name"] for t in TOOLS]
    assert names == ["bash", "submit"]
    assert TOOLS[0]["function"]["parameters"]["required"] == ["command"]
