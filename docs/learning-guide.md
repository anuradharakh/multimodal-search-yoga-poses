# Learning guide — building the pipeline step by step

Each section below matches **one commit** in the Git history, so you can check out a step, read
what was added and why, run it, and compare your output with the numbers recorded here:

```bash
git log --oneline --reverse          # the steps, oldest first
git show <commit> --stat             # files added in a step
git checkout <commit>                # the repo exactly as it was after that step
```

---

## Step 1 — Project scaffold

**Goal.** A skeleton every later step plugs into, mirroring a production ML repo:

| File | Role |
|---|---|
| `config.yaml` | every setting in one place; `pipeline:` switches stages on/off |
| `run_pipeline.py` | runs the enabled stages in order (`--only <stage>` to run one) |
| `src/<stage>/` | one sub-package per pipeline stage, small single-purpose modules |
| `src/utils/` | config loading, logging, JSON/CSV I/O, device selection |
| `data/` (git-ignored) | everything generated: images, features, the vector index |
| `outputs/` (committed) | metrics, figures and screenshots that document the results |

**Why config-driven?** Experiments change settings, not code. Keeping paths, model names and
hyper-parameters in `config.yaml` makes runs reproducible and diffable.

**Run it.**

```bash
python run_pipeline.py
# [WARNING] No stages selected — enable some under `pipeline:` in config.yaml or pass --only.
```
