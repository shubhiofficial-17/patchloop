import subprocess
import time
from types import SimpleNamespace

import pytest

from patchloop import sandbox as sandbox_mod
from patchloop.sandbox import DockerSandbox, LocalSandbox, SandboxError


def test_local_runs_in_workdir(tmp_path):
    box = LocalSandbox(tmp_path)
    result = box.run("pwd")
    assert result.exit_code == 0
    assert result.output.strip() == str(tmp_path)


def test_local_merges_stderr_and_keeps_exit_code(tmp_path):
    result = LocalSandbox(tmp_path).run("echo out; echo err >&2; exit 3")
    assert "out" in result.output and "err" in result.output
    assert result.exit_code == 3
    assert not result.timed_out


def test_local_state_does_not_carry_over(tmp_path):
    (tmp_path / "sub").mkdir()
    box = LocalSandbox(tmp_path)
    box.run("cd sub && export FOO=1")
    assert box.run("pwd").output.strip() == str(tmp_path)
    assert box.run("echo ${FOO:-unset}").output.strip() == "unset"


def test_local_timeout_kills_child_processes(tmp_path):
    start = time.monotonic()
    result = LocalSandbox(tmp_path).run("echo started; sleep 30 & sleep 30", timeout=1)
    assert result.timed_out
    assert "started" in result.output
    assert time.monotonic() - start < 10


def test_local_stdin_is_closed(tmp_path):
    result = LocalSandbox(tmp_path).run("read line || echo no-input", timeout=5)
    assert "no-input" in result.output


def test_local_sets_default_env(tmp_path):
    assert LocalSandbox(tmp_path).run("echo $PAGER").output.strip() == "cat"


def test_local_init_command_runs_first(tmp_path):
    box = LocalSandbox(tmp_path, init_command="export GREETING=hi")
    assert box.run("echo $GREETING").output.strip() == "hi"


def test_local_missing_workdir(tmp_path):
    with pytest.raises(SandboxError):
        LocalSandbox(tmp_path / "missing")


class FakeDocker:
    """Records docker CLI calls instead of running them."""

    def __init__(self, run_returncode=0):
        self.calls = []
        self.run_returncode = run_returncode

    def __call__(self, argv, **kwargs):
        self.calls.append(argv)
        if argv[1] == "run":
            return SimpleNamespace(returncode=self.run_returncode, stdout="abc123\n", stderr="no such image")
        return SimpleNamespace(returncode=0, stdout="", stderr="")


def test_docker_start_exec_and_close(monkeypatch):
    fake = FakeDocker()
    monkeypatch.setattr(subprocess, "run", fake)
    box = DockerSandbox("swebench/some-image:latest", platform="linux/amd64")

    start = fake.calls[0]
    assert start[:3] == ["docker", "run", "-d"]
    assert "--platform" in start and "linux/amd64" in start
    assert start[-3:] == ["swebench/some-image:latest", "sleep", "infinity"]
    assert box.container_id == "abc123"

    argv = box.exec_argv("ls -la", timeout=30)
    assert argv[:4] == ["docker", "exec", "-w", "/testbed"]
    assert "PAGER=cat" in argv
    idx = argv.index("abc123")
    assert argv[idx + 1 : idx + 4] == ["timeout", "--kill-after=5", "30"]
    assert argv[-3:-1] == ["bash", "-c"]
    assert argv[-1].startswith("source /opt/miniconda3/bin/activate testbed")
    assert argv[-1].endswith("ls -la")

    box.close()
    assert fake.calls[-1] == ["docker", "rm", "-f", "abc123"]
    assert box.container_id is None
    with pytest.raises(SandboxError):
        box.exec_argv("ls", timeout=1)


def test_docker_start_failure(monkeypatch):
    monkeypatch.setattr(subprocess, "run", FakeDocker(run_returncode=1))
    with pytest.raises(SandboxError, match="no such image"):
        DockerSandbox("missing:latest")


def test_docker_marks_timeout_exit_code(monkeypatch):
    monkeypatch.setattr(subprocess, "run", FakeDocker())
    box = DockerSandbox("img")
    monkeypatch.setattr(
        sandbox_mod, "_run_process", lambda argv, timeout: sandbox_mod.CommandResult("partial", 124)
    )
    result = box.run("sleep 999", timeout=1)
    assert result.timed_out
