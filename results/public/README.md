# Public result exports

`Project_planning_and_ReAct_answers.jsonl` and `Project_planning_and_ReAct_and_Reflection_answers.jsonl` contain one row per active task (403 rows each). Each row has the task ID, terminal status, answer status, answer value and unit. `recovery_round` is set only when Reflection supplied the final answer. These exports omit local filesystem paths, model transcripts, tool artifacts and gold answers.

`metrics.json` records the provisional offline counts and SHA-256 hashes of the full source answer files. The evaluator and its draft rules are in `evaluation/` and `data/preprocessed/evaluation_rules/`. The original full run directories are local replay data and are excluded from Git by `.gitignore`.
