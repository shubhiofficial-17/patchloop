import json

import pytest

from patchloop.model import ModelError, OpenAICompatibleModel, Pricing, parse_response


def completion(content=None, tool_calls=None, prompt_tokens=10, completion_tokens=5, **usage):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "choices": [{"message": message}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, **usage},
    }


def bash_call(command, call_id="call_1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "bash", "arguments": json.dumps({"command": command})},
    }


def make_model(server, **kwargs):
    return OpenAICompatibleModel(
        "test-model", base_url=server["url"], api_key="sk-test", sleep=lambda s: None, **kwargs
    )


def test_sends_request_and_parses_tool_calls(server):
    server["queue"].append((200, completion(tool_calls=[bash_call("ls")])))
    tools = [{"type": "function", "function": {"name": "bash"}}]
    response = make_model(server, temperature=0.0).complete([{"role": "user", "content": "hi"}], tools)

    request = server["requests"][0]
    assert request["path"] == "/v1/chat/completions"
    assert request["headers"]["Authorization"] == "Bearer sk-test"
    assert request["body"]["model"] == "test-model"
    assert request["body"]["tools"] == tools
    assert request["body"]["temperature"] == 0.0

    assert [c.name for c in response.tool_calls] == ["bash"]
    assert response.tool_calls[0].arguments == {"command": "ls"}
    assert response.input_tokens == 10 and response.output_tokens == 5
    assert response.message["tool_calls"][0]["id"] == "call_1"


def test_retries_rate_limits_then_succeeds(server):
    server["queue"] += [(429, {"error": "slow down"}), (503, {"error": "busy"}), (200, completion("done"))]
    response = make_model(server).complete([], [])
    assert response.content == "done"
    assert len(server["requests"]) == 3


def test_does_not_retry_client_errors(server):
    server["queue"].append((400, {"error": "context too long"}))
    with pytest.raises(ModelError, match="HTTP 400.*context too long"):
        make_model(server).complete([], [])
    assert len(server["requests"]) == 1


def test_gives_up_after_max_retries(server):
    server["queue"] += [(500, {})] * 3
    with pytest.raises(ModelError, match="HTTP 500"):
        make_model(server, max_retries=2).complete([], [])


def test_unreachable_server():
    model = OpenAICompatibleModel("m", base_url="http://127.0.0.1:9/v1", max_retries=1, sleep=lambda s: None, timeout=2)
    with pytest.raises(ModelError, match="could not reach"):
        model.complete([], [])


def test_extra_body_and_headers_are_sent(server):
    server["queue"].append((200, completion("ok")))
    make_model(server, headers={"X-Team": "infra"}, extra_body={"reasoning_effort": "low"}).complete([], [])
    request = server["requests"][0]
    assert request["headers"]["X-Team"] == "infra"
    assert request["body"]["reasoning_effort"] == "low"
    assert "temperature" not in request["body"]


def test_bad_tool_arguments_are_flagged():
    data = completion(tool_calls=[{"id": "x", "function": {"name": "bash", "arguments": "{not json"}}])
    call = parse_response(data).tool_calls[0]
    assert call.arguments == {}
    assert "not valid JSON" in call.parse_error


def test_content_parts_and_reported_cost():
    data = completion(content=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}], cost=0.0123)
    response = parse_response(data)
    assert response.content == "ab"
    assert response.reported_cost == pytest.approx(0.0123)
    assert response.message == {"role": "assistant", "content": "ab"}


def test_no_choices_raises():
    with pytest.raises(ModelError, match="no choices"):
        parse_response({"choices": []})


def test_pricing():
    pricing = Pricing(input_per_million=2.0, output_per_million=10.0)
    assert pricing.cost(1_000_000, 100_000) == pytest.approx(3.0)
    assert pricing.is_set
    assert not Pricing().is_set
