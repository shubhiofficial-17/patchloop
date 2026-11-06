"""Running the agent over many SWE-bench instances in parallel.

Output layout (one folder per run):

    preds.jsonl              one line per instance, in the format the SWE-bench harness reads
    results.jsonl            per-instance stats: exit status, steps, tokens, cost, time
    summary.json             totals for the whole run
    trajectories/<id>.json   every message and tool call, for debugging and replay
"""

from __future__ import annotations

import json
import logging
import threading
import traceback
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .loop import Agent, AgentConfig, AgentResult
from .model import ChatModel, Pricing
from .sandbox import DockerSandbox, Sandbox
from .swebench import build_task, image_name

log = logging.getLogger("patchloop")

ModelFactory = Callable[[], ChatModel]
SandboxFactory = Callable[[dict[str, Any]], Sandbox]


def docker_sandbox_factory(command_timeout: float, platform: str | None = None) -> SandboxFactory:
    def factory(instance: dict[str, Any]) -> Sandbox:
        return DockerSandbox(image_name(instance), default_timeout=command_timeout, platform=platform)

    return factory


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


class BatchRunner:
    def __init__(
        self,
        output_dir: Path,
        model_factory: ModelFactory,
        sandbox_factory: SandboxFactory,
        model_name: str,
        config: AgentConfig | None = None,
        pricing: Pricing | None = None,
        workers: int = 1,
        redo: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.model_factory = model_factory
        self.sandbox_factory = sandbox_factory
        self.model_name = model_name
        self.config = config or AgentConfig()
        self.pricing = pricing or Pricing()
        self.workers = max(1, workers)
        self.redo = redo
        self._lock = threading.Lock()

        self.preds_path = self.output_dir / "preds.jsonl"
        self.results_path = self.output_dir / "results.jsonl"
        self.trajectory_dir = self.output_dir / "trajectories"

    def run(self, instances: list[dict[str, Any]]) -> dict[str, Any]:
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        ids = [i["instance_id"] for i in instances]
        done = {p["instance_id"] for p in _read_jsonl(self.preds_path)}

        if self.redo:
            redo_ids = set(ids)
            for path in (self.preds_path, self.results_path):
                _write_jsonl(path, [r for r in _read_jsonl(path) if r["instance_id"] not in redo_ids])
            done -= redo_ids

        todo = [i for i in instances if i["instance_id"] not in done]
        log.info("%d instances: %d already done, %d to run", len(instances), len(instances) - len(todo), len(todo))

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(self._run_one, inst): inst["instance_id"] for inst in todo}
            for future in as_completed(futures):
                future.result()  # _run_one never raises; this just surfaces bugs in the runner

        summary = self.summarize()
        (self.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def _run_one(self, instance: dict[str, Any]) -> None:
        iid = instance["instance_id"]
        log.info("[%s] starting", iid)
        try:
            sandbox = self.sandbox_factory(instance)
            try:
                agent = Agent(self.model_factory(), sandbox, config=self.config, pricing=self.pricing)
                result = agent.run(build_task(instance))
            finally:
                sandbox.close()
        except Exception as exc:  # infrastructure failure: record it, leave it out of preds so a rerun retries it
            log.error("[%s] failed: %s", iid, exc)
            self._save_trajectory(iid, {"instance_id": iid, "error": traceback.format_exc()})
            self._append(self.results_path, {"instance_id": iid, "exit_status": "error", "error": str(exc)})
            return

        self._save_trajectory(iid, {"instance_id": iid, "model": self.model_name, **result.to_dict()})
        self._append(
            self.preds_path,
            {"instance_id": iid, "model_name_or_path": self.model_name, "model_patch": result.patch},
        )
        self._append(self.results_path, self._result_row(iid, result))
        log.info("[%s] %s after %d steps, $%.4f", iid, result.exit_status, result.steps, result.cost)

    @staticmethod
    def _result_row(iid: str, result: AgentResult) -> dict[str, Any]:
        return {
            "instance_id": iid,
            "exit_status": result.exit_status,
            "steps": result.steps,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cost": round(result.cost, 6),
            "duration_seconds": round(result.duration_seconds, 1),
            "patch_lines": result.patch.count("\n"),
            "error": result.error,
        }

    def _append(self, path: Path, row: dict[str, Any]) -> None:
        with self._lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")

    def _save_trajectory(self, iid: str, data: dict[str, Any]) -> None:
        path = self.trajectory_dir / f"{iid}.json"
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def summarize(self) -> dict[str, Any]:
        latest: dict[str, dict[str, Any]] = {}
        for row in _read_jsonl(self.results_path):
            latest[row["instance_id"]] = row  # later rows win
        rows = list(latest.values())
        statuses = Counter(r["exit_status"] for r in rows)
        total_cost = sum(r.get("cost", 0.0) for r in rows)
        with_patch = sum(1 for r in rows if r.get("patch_lines", 0) > 0)
        return {
            "model": self.model_name,
            "instances": len(rows),
            "exit_status": dict(statuses),
            "non_empty_patches": with_patch,
            "total_cost": round(total_cost, 4),
            "cost_per_instance": round(total_cost / len(rows), 4) if rows else 0.0,
            "input_tokens": sum(r.get("input_tokens", 0) for r in rows),
            "output_tokens": sum(r.get("output_tokens", 0) for r in rows),
            "errors": sorted(r["instance_id"] for r in rows if r["exit_status"] == "error"),
        }
