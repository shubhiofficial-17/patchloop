# patchloop

A minimal coding agent that reads a GitHub issue, explores the repository, edits the code and
submits a patch. Every task runs in its own Docker container. The goal of the project is not only
to build the agent but to **measure it honestly**: against a known baseline, on fresh tasks, with
ablations that show what each part of the agent is worth.

> Work in progress. The first milestone (agent loop, sandbox, SWE-bench runner) is being built now.
