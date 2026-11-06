"""Command line entry point: `patchloop run ...`."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from .loop import AgentConfig
from .model import OpenAICompatibleModel, Pricing
from .runner import BatchRunner, docker_sandbox_factory
from .swebench import load_instances, resolve_dataset


def _header(value: str) -> tuple[str, str]:
    key, sep, val = value.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError("headers look like Name=value")
    return key.strip(), val.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="patchloop", description="Run a coding agent on SWE-bench tasks.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the agent on SWE-bench instances, each in its own Docker container.")
    data = run.add_argument_group("tasks")
    data.add_argument("--dataset", default="verified-mini",
                      help="verified, verified-mini, lite, full, a HuggingFace id, or a local .json/.jsonl file")
    data.add_argument("--split", default="test")
    data.add_argument("--instance", action="append", dest="instances", metavar="ID",
                      help="run only this instance (repeatable)")
    data.add_argument("--filter", dest="pattern", metavar="REGEX", help="run instances whose id matches")
    data.add_argument("--limit", type=int, help="run at most this many instances")

    model = run.add_argument_group("model")
    model.add_argument("--model", required=True, help="model name as the API expects it")
    model.add_argument("--base-url", default=os.environ.get("PATCHLOOP_BASE_URL", "https://api.openai.com/v1"),
                       help="OpenAI-compatible endpoint (default: $PATCHLOOP_BASE_URL or OpenAI)")
    model.add_argument("--api-key-env", default="OPENAI_API_KEY",
                       help="environment variable that holds the API key (default: OPENAI_API_KEY)")
    model.add_argument("--header", action="append", type=_header, default=[], metavar="NAME=VALUE",
                       help="extra HTTP header for the model API (repeatable)")
    model.add_argument("--temperature", type=float)
    model.add_argument("--input-price", type=float, default=0.0, help="US$ per 1M input tokens")
    model.add_argument("--output-price", type=float, default=0.0, help="US$ per 1M output tokens")

    limits = run.add_argument_group("limits")
    limits.add_argument("--step-limit", type=int, default=100)
    limits.add_argument("--cost-limit", type=float, default=3.0, help="US$ per instance, 0 for no limit")
    limits.add_argument("--command-timeout", type=float, default=120.0, help="seconds per shell command")

    out = run.add_argument_group("output")
    out.add_argument("--output", type=Path, required=True, help="folder for predictions and trajectories")
    out.add_argument("--workers", type=int, default=1, help="instances to run in parallel")
    out.add_argument("--redo", action="store_true", help="rerun instances that already have a prediction")
    out.add_argument("--platform", help="docker platform, e.g. linux/amd64 on Apple Silicon")
    out.add_argument("-v", "--verbose", action="store_true")
    return parser


def run_command(args: argparse.Namespace) -> int:
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        logging.warning("$%s is not set; calling the model API without a key", args.api_key_env)

    pricing = Pricing(args.input_price, args.output_price)
    if args.cost_limit > 0 and not pricing.is_set:
        logging.warning("no prices given, so the cost limit only applies if the API reports cost")

    instances = load_instances(args.dataset, args.split, args.instances, args.pattern, args.limit)
    if not instances:
        print("No instances matched.", file=sys.stderr)
        return 1

    headers = dict(args.header)

    def model_factory() -> OpenAICompatibleModel:
        return OpenAICompatibleModel(
            args.model, base_url=args.base_url, api_key=api_key, headers=headers, temperature=args.temperature
        )

    runner = BatchRunner(
        output_dir=args.output,
        model_factory=model_factory,
        sandbox_factory=docker_sandbox_factory(args.command_timeout, args.platform),
        model_name=args.model,
        config=AgentConfig(
            step_limit=args.step_limit, cost_limit=args.cost_limit, command_timeout=args.command_timeout
        ),
        pricing=pricing,
        workers=args.workers,
        redo=args.redo,
    )
    summary = runner.run(instances)
    print(json.dumps(summary, indent=2))

    dataset = resolve_dataset(args.dataset)
    if dataset == "MariusHobbhahn/swe-bench-verified-mini":
        dataset = "princeton-nlp/SWE-bench_Verified"  # the mini set's instances all come from Verified
    print(
        "\nScore the patches with the official harness:\n"
        f"  python -m swebench.harness.run_evaluation --dataset_name {dataset} "
        f"--predictions_path {runner.preds_path} --max_workers 4 --run_id {args.output.name}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.command == "run":
        return run_command(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
