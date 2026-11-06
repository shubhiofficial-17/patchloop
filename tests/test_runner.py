import json

import pytest
from fakes import BUGGY_CALC, ScriptedModel, call, make_git_repo, reply

from patchloop.cli import build_parser, main
from patchloop.model import Pricing
from patchloop.runner import BatchRunner
from patchloop.sandbox import LocalSandbox, SandboxError
from patchloop.swebench import build_task, image_name, load_instances


def instance(iid, problem="add() subtracts"):
    return {"instance_id": iid, "problem_statement": problem, "repo": "acme/calc", "base_commit": "abc"}


def test_image_name_matches_swebench_hub_naming():
    assert (
        image_name({"instance_id": "django__django-11099"})
        == "docker.io/swebench/sweb.eval.x86_64.django_1776_django-11099:latest"
    )
    assert image_name({"instance_id": "Foo__Bar-1"}).endswith("foo_1776_bar-1:latest")
    assert image_name({"instance_id": "x", "image_name": "custom:1"}) == "custom:1"


def test_build_task_wraps_issue():
    task = build_task(instance("a", problem="  Something breaks  \n"))
    assert "<issue>\nSomething breaks\n</issue>" in task


def test_load_local_jsonl_with_filters(tmp_path):
    path = tmp_path / "tasks.jsonl"
    rows = [instance("django__django-1"), instance("django__django-2"), instance("sympy__sympy-3")]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    assert len(load_instances(str(path))) == 3
    assert [r["instance_id"] for r in load_instances(str(path), pattern="^django")] == [
        "django__django-1",
        "django__django-2",
    ]
    assert len(load_instances(str(path), limit=1)) == 1
    assert [r["instance_id"] for r in load_instances(str(path), instance_ids=["sympy__sympy-3"])] == [
        "sympy__sympy-3"
    ]
    with pytest.raises(SystemExit, match="not found"):
        load_instances(str(path), instance_ids=["nope"])


def test_load_local_json_dict(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({"a": instance("a"), "b": instance("b")}))
    assert {r["instance_id"] for r in load_instances(str(path))} == {"a", "b"}


@pytest.fixture
def setup(tmp_path):
    """One throwaway git repo per instance, and a scripted model that fixes the bug."""
    created = []

    def sandbox_factory(inst):
        iid = inst["instance_id"]
        if iid == "broken":
            raise SandboxError("image pull failed")
        created.append(iid)  # a fresh repo every time, like a fresh container
        repo = make_git_repo(tmp_path / "repos" / f"{iid}-{len(created)}", {"calc.py": BUGGY_CALC})
        return LocalSandbox(repo)

    def model_factory():
        return ScriptedModel(
            [
                reply(call("bash", command="sed -i 's/a - b/a + b/' calc.py"), cost=0.01),
                reply(call("submit", summary="fixed"), cost=0.01),
            ]
        )

    def make_runner(**kwargs):
        return BatchRunner(
            output_dir=tmp_path / "run",
            model_factory=model_factory,
            sandbox_factory=sandbox_factory,
            model_name="scripted",
            pricing=Pricing(),
            **kwargs,
        )

    return make_runner, tmp_path / "run"


def test_batch_writes_predictions_results_and_trajectories(setup):
    make_runner, out = setup
    summary = make_runner(workers=2).run([instance("a"), instance("b"), instance("broken")])

    preds = [json.loads(line) for line in (out / "preds.jsonl").read_text().splitlines()]
    assert {p["instance_id"] for p in preds} == {"a", "b"}
    assert all(p["model_name_or_path"] == "scripted" for p in preds)
    assert all("+    return a + b" in p["model_patch"] for p in preds)

    trajectory = json.loads((out / "trajectories" / "a.json").read_text())
    assert trajectory["exit_status"] == "submitted"
    assert trajectory["messages"][0]["role"] == "system"
    assert "image pull failed" in json.loads((out / "trajectories" / "broken.json").read_text())["error"]

    assert summary["instances"] == 3
    assert summary["exit_status"] == {"submitted": 2, "error": 1}
    assert summary["errors"] == ["broken"]
    assert summary["non_empty_patches"] == 2
    assert summary["total_cost"] == pytest.approx(0.04)
    assert json.loads((out / "summary.json").read_text()) == summary


def test_batch_resumes_and_redoes(setup):
    make_runner, out = setup
    make_runner().run([instance("a")])
    make_runner().run([instance("a"), instance("b")])  # "a" is skipped
    preds = (out / "preds.jsonl").read_text().splitlines()
    assert [json.loads(p)["instance_id"] for p in preds] == ["a", "b"]

    make_runner(redo=True).run([instance("a")])
    preds = [json.loads(p)["instance_id"] for p in (out / "preds.jsonl").read_text().splitlines()]
    assert sorted(preds) == ["a", "b"]  # no duplicate lines for "a"


def test_cli_parses_run_arguments(tmp_path):
    args = build_parser().parse_args(
        [
            "run", "--model", "gpt-x", "--output", str(tmp_path / "out"),
            "--instance", "a", "--instance", "b", "--header", "X-Team=infra",
            "--input-price", "0.25", "--workers", "4",
        ]
    )
    assert args.instances == ["a", "b"]
    assert args.header == [("X-Team", "infra")]
    assert args.input_price == 0.25 and args.workers == 4
    assert args.dataset == "verified-mini"


def test_cli_rejects_bad_header(tmp_path):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "--model", "m", "--output", "o", "--header", "nonsense"])


def test_cli_reports_no_matches(tmp_path, capsys):
    path = tmp_path / "tasks.jsonl"
    path.write_text(json.dumps(instance("a")) + "\n")
    code = main(["run", "--model", "m", "--output", str(tmp_path / "o"), "--dataset", str(path), "--filter", "zzz"])
    assert code == 1
    assert "No instances matched" in capsys.readouterr().err
