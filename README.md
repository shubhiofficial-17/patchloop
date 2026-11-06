# patchloop

A minimal coding agent that reads a GitHub issue, explores the repository, edits the code and
submits a patch. Every task runs in its own Docker container, and the patches are scored with the
official SWE-bench harness.

The point of the project is not only to build the agent but to **measure it honestly**.

## Why measure it this way

Public SWE-bench Verified scores no longer say much on their own. Frontier models now score in the
90s on it with very small agents, and the benchmark is known to be contaminated (models have seen
many of its fixes during training). So instead of chasing a headline number, patchloop reports:

1. **A same-model comparison** against [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent),
   a ~100-line bash-only agent used as the standard baseline.
2. **Results on fresh tasks** (issues created after a model's training data), not just Verified.
3. **Ablations**: each feature of the agent can be switched off to show what it is worth.
4. **Cost per resolved task**, not just the resolve rate.

## How it works

```
 issue text ──► agent loop ──────────────► model (any OpenAI-compatible API)
                  │    ▲
        tool call │    │ command output (exit code, truncated output)
                  ▼    │
          Docker sandbox: SWE-bench image, repository at /testbed
                  │
          git add -A && git diff ──► preds.jsonl ──► SWE-bench harness ──► resolved / not
```

The agent has two tools: `bash` and `submit`. Design choices worth knowing:

- **Stateless commands.** Every `bash` call is a fresh shell, so a step never depends on hidden
  state from an earlier one. This makes trajectories easy to read and replay.
- **Timeouts inside the container.** Commands are wrapped in `timeout` inside the container,
  because killing the `docker exec` client alone would leave the process running.
- **Output truncation.** Long output keeps its first and last parts, where errors and summaries
  usually are, and tells the model how to look at less.
- **Every tool call gets an answer**, including unknown tools and malformed arguments, so the
  conversation stays valid for the API and the model can correct itself.
- **Limits** on steps, cost and command time. When a run stops early, whatever the agent has
  changed so far is still saved as the patch.
- **No SDK dependencies.** The model client is ~200 lines of standard library code and works with
  any OpenAI-compatible endpoint (OpenAI, OpenRouter, vLLM, Ollama or your own gateway).

## Quick start

Needs Python 3.10+, Docker and an x86_64 Linux machine (WSL2 works). On Apple Silicon add
`--platform linux/amd64`.

```bash
git clone https://github.com/shubhiofficial-17/patchloop
cd patchloop
python -m venv .venv && source .venv/bin/activate
pip install -e ".[swebench,dev]"
pytest
```

Run one task from the 50-task SWE-bench Verified Mini subset:

```bash
export OPENAI_API_KEY=...
patchloop run --dataset verified-mini --limit 1 \
  --model <model-name> --input-price 0.25 --output-price 2.0 \
  --output runs/smoke -v
```

Then score the patches with the official harness:

```bash
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --predictions_path runs/smoke/preds.jsonl \
  --max_workers 4 --run_id smoke
```

Other providers: point `--base-url` at the endpoint, name the key's environment variable with
`--api-key-env`, and add headers with `--header Name=value`. Run `patchloop run --help` for all options.

## What a run produces

```
runs/<name>/
  preds.jsonl              predictions in the format the SWE-bench harness reads
  results.jsonl            per task: exit status, steps, tokens, cost, time
  summary.json             totals for the run
  trajectories/<id>.json   every message and tool call
```

Runs can be resumed: rerunning the same command skips tasks that already have a prediction.
Tasks that failed for infrastructure reasons (for example an image pull) are not written to
`preds.jsonl`, so they are retried on the next run. Use `--redo` to rerun finished tasks.

## Results

See [RESULTS.md](RESULTS.md). No numbers yet: they will only be added from real, reproducible runs.

## Roadmap

- [x] Agent loop, Docker sandbox, OpenAI-compatible client, parallel runner
- [ ] Reproduce the mini-SWE-agent baseline on Verified Mini with the same model
- [ ] Reproduction step: write a failing test before fixing
- [ ] Code search tool
- [ ] Self-review before submitting
- [ ] Fresh-task evaluation on SWE-rebench
- [ ] Ablation report and a trajectory viewer

## Project layout

```
src/patchloop/
  loop.py       the agent loop, limits and patch extraction
  tools.py      tool schemas and how command output is shown to the model
  prompts.py    system prompt and task template
  model.py      OpenAI-compatible chat client with retries and cost tracking
  sandbox.py    local and Docker sandboxes
  swebench.py   dataset loading and SWE-bench image names
  runner.py     parallel batch runs, resumable output
  cli.py        the `patchloop` command
tests/          unit tests plus an end-to-end test against a fake API
```

## License

MIT
