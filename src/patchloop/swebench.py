"""Loading SWE-bench instances and mapping them to their Docker images."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .prompts import TASK_TEMPLATE

DATASETS = {
    "verified": "princeton-nlp/SWE-bench_Verified",
    "verified-mini": "MariusHobbhahn/swe-bench-verified-mini",
    "lite": "princeton-nlp/SWE-bench_Lite",
    "full": "princeton-nlp/SWE-bench",
}


def resolve_dataset(name: str) -> str:
    return DATASETS.get(name, name)


def _read_local(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    return list(data.values()) if isinstance(data, dict) else list(data)


def load_instances(
    dataset: str,
    split: str = "test",
    instance_ids: Iterable[str] | None = None,
    pattern: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load instances from a dataset alias, a HuggingFace id, or a local .json/.jsonl file."""
    path = Path(dataset)
    if path.suffix in {".json", ".jsonl"} and path.exists():
        rows = _read_local(path)
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise SystemExit(
                "Loading from HuggingFace needs the 'datasets' package: pip install -e '.[swebench]'"
            ) from exc
        rows = [dict(row) for row in load_dataset(resolve_dataset(dataset), split=split)]

    if instance_ids:
        wanted = set(instance_ids)
        rows = [r for r in rows if r["instance_id"] in wanted]
        missing = wanted - {r["instance_id"] for r in rows}
        if missing:
            raise SystemExit(f"instances not found in {dataset}: {', '.join(sorted(missing))}")
    if pattern:
        regex = re.compile(pattern)
        rows = [r for r in rows if regex.search(r["instance_id"])]
    if limit is not None:
        rows = rows[:limit]
    return rows


def image_name(instance: dict[str, Any]) -> str:
    """Name of the prebuilt SWE-bench evaluation image for an instance on Docker Hub."""
    if instance.get("image_name"):
        return instance["image_name"]
    docker_id = instance["instance_id"].replace("__", "_1776_")
    return f"docker.io/swebench/sweb.eval.x86_64.{docker_id}:latest".lower()


def build_task(instance: dict[str, Any]) -> str:
    return TASK_TEMPLATE.format(problem_statement=instance["problem_statement"].strip())
