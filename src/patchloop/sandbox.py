"""Places where the agent's shell commands run.

Every command runs in a fresh bash process, so `cd` and exported variables do
not carry over between commands. That keeps each step independent and makes a
saved trajectory easy to replay.
"""

from __future__ import annotations

import os
import signal
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

# Keep tools from opening pagers or printing progress bars into the transcript.
DEFAULT_ENV: dict[str, str] = {
    "PAGER": "cat",
    "MANPAGER": "cat",
    "GIT_PAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
}

# SWE-bench images ship a conda env named "testbed" with the repo's dependencies.
SWEBENCH_INIT = "source /opt/miniconda3/bin/activate testbed >/dev/null 2>&1 || true"

# GNU `timeout` exits with 124 when it had to stop the command.
TIMEOUT_EXIT_CODE = 124


@dataclass
class CommandResult:
    output: str
    exit_code: int
    timed_out: bool = False


class SandboxError(RuntimeError):
    """The sandbox could not be started or reached."""


class Sandbox(Protocol):
    def run(self, command: str, timeout: float | None = None) -> CommandResult: ...

    def close(self) -> None: ...


def _with_init(init_command: str, command: str) -> str:
    return f"{init_command}\n{command}" if init_command else command


def _run_process(
    argv: Sequence[str],
    timeout: float,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> CommandResult:
    """Run argv with stdout and stderr merged; kill the whole process group on timeout."""
    try:
        proc = subprocess.Popen(
            list(argv),
            cwd=cwd,
            env=dict(env) if env is not None else None,
            stdin=subprocess.DEVNULL,  # interactive programs fail fast instead of hanging
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        raise SandboxError(f"executable not found: {argv[0]}") from exc

    try:
        output, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            output, _ = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            output = ""
        return CommandResult(output or "", -1, timed_out=True)
    return CommandResult(output or "", proc.returncode)


class LocalSandbox:
    """Runs commands directly on this machine inside `workdir`.

    There is no isolation at all, so use it for tests and trusted experiments only.
    """

    def __init__(
        self,
        workdir: str | Path,
        env: Mapping[str, str] | None = None,
        init_command: str = "",
        default_timeout: float = 60.0,
    ) -> None:
        self.workdir = Path(workdir)
        if not self.workdir.is_dir():
            raise SandboxError(f"workdir does not exist: {self.workdir}")
        self.env = {**os.environ, **DEFAULT_ENV, **(env or {})}
        self.init_command = init_command
        self.default_timeout = default_timeout

    def run(self, command: str, timeout: float | None = None) -> CommandResult:
        script = _with_init(self.init_command, command)
        return _run_process(
            ["bash", "-c", script],
            timeout or self.default_timeout,
            cwd=self.workdir,
            env=self.env,
        )

    def close(self) -> None:
        pass

    def __enter__(self) -> LocalSandbox:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class DockerSandbox:
    """Runs commands in a long-lived container started from `image`.

    The container just sleeps; each command is a separate `docker exec`. The
    command is wrapped in `timeout` inside the container, because killing the
    `docker exec` client alone would leave the process running in there.
    """

    def __init__(
        self,
        image: str,
        workdir: str = "/testbed",
        env: Mapping[str, str] | None = None,
        init_command: str = SWEBENCH_INIT,
        default_timeout: float = 120.0,
        platform: str | None = None,
        docker_bin: str = "docker",
        start_timeout: float = 900.0,
        name_prefix: str = "patchloop",
    ) -> None:
        self.image = image
        self.workdir = workdir
        self.env = {**DEFAULT_ENV, **(env or {})}
        self.init_command = init_command
        self.default_timeout = default_timeout
        self.platform = platform
        self.docker_bin = docker_bin
        self.start_timeout = start_timeout
        self.name = f"{name_prefix}-{uuid.uuid4().hex[:10]}"
        self.container_id: str | None = None
        self._start()

    def _start(self) -> None:
        argv = [self.docker_bin, "run", "-d", "--rm", "--name", self.name, "-w", self.workdir]
        if self.platform:
            argv += ["--platform", self.platform]
        argv += [self.image, "sleep", "infinity"]
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=self.start_timeout)
        except FileNotFoundError as exc:
            raise SandboxError(f"docker not found (looked for '{self.docker_bin}')") from exc
        except subprocess.TimeoutExpired as exc:
            raise SandboxError(f"timed out starting a container from {self.image}") from exc
        if result.returncode != 0:
            raise SandboxError(f"could not start {self.image}: {result.stderr.strip()}")
        self.container_id = result.stdout.strip()

    def exec_argv(self, command: str, timeout: float) -> list[str]:
        if not self.container_id:
            raise SandboxError("sandbox is closed")
        argv = [self.docker_bin, "exec", "-w", self.workdir]
        for key, value in self.env.items():
            argv += ["-e", f"{key}={value}"]
        argv += [
            self.container_id,
            "timeout",
            "--kill-after=5",
            f"{timeout:g}",
            "bash",
            "-c",
            _with_init(self.init_command, command),
        ]
        return argv

    def run(self, command: str, timeout: float | None = None) -> CommandResult:
        limit = timeout or self.default_timeout
        # The host-side limit is only a safety net for a stuck docker client.
        result = _run_process(self.exec_argv(command, limit), timeout=limit + 30)
        if result.exit_code == TIMEOUT_EXIT_CODE:
            result.timed_out = True
        return result

    def close(self) -> None:
        if not self.container_id:
            return
        try:
            subprocess.run(
                [self.docker_bin, "rm", "-f", self.container_id],
                capture_output=True,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
        self.container_id = None

    def __enter__(self) -> DockerSandbox:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
