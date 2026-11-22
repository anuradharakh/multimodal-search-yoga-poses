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

---

## Step 2 — Data preparation

**Goal.** Turn the raw Kaggle download into a clean, de-duplicated, leakage-free dataset.

| Module | What it does |
|---|---|
| `src/data/download.py` | fetches `arrowe/yoga-poses-dataset-107` with `kagglehub` (cached) |
| `src/data/cleaning.py` | MD5-hashes every file, drops ambiguous / duplicate / tiny / unreadable images, converts to RGB JPEG ≤ 512 px |
| `src/data/splitting.py` | stratified 70 / 15 / 15 split into **gallery / val / test** |
| `src/data/manifest.py` | reads `manifest.csv`, the single table every later stage uses |
| `src/data/pose_names.py` | Sanskrit ↔ English pose names |

**Key decisions.**
* **Ambiguous labels.** 177 images have *byte-identical* copies filed under two different poses
  (e.g. *Ardha Uttanasana* vs. *Uttanasana*). Their label is unknowable, so all copies are dropped.
* **Re-split after de-duplication.** The dataset's own train/test folders share duplicates; merging
  and re-splitting guarantees no identical image is both in the database and a query.
* **Three splits, not two.** Retrieval has no "training", but it has *tuning*. Choosing settings on
  `val` and reporting on untouched `test` keeps the final numbers honest.

**Run it.**

```bash
python run_pipeline.py --only data_prep
# Cleaning report: {'files_seen': 5994, 'unreadable': 0, 'too_small': 29,
#                   'duplicate_within_pose': 14, 'ambiguous_label': 177, 'kept': 5774}
# Kept 5,774 images of 107 poses — {'gallery': 4041, 'test': 867, 'val': 866}
```

Output: `data/processed/images/<pose>/*.jpg`, `data/processed/manifest.csv`,
`outputs/metrics/data_cleaning.json`.

---

## Step 3 — Image embeddings from pre-trained CNNs

**Goal.** Represent every image as a vector such that *similar poses → nearby vectors*.

| Module | What it does |
|---|---|
| `src/embedding/backbones.py` | ResNet-50, EfficientNetV2-S, ConvNeXt-Tiny (ImageNet), CLIP ResNet-50 and CLIP ViT-B/32 (reference) — frozen |
| `src/embedding/pooling.py` | feature map → vector: average (GAP) vs. generalised-mean (GeM) pooling |
| `src/embedding/preprocessing.py` | centre-crop vs. letterbox to 224 × 224 |
| `src/embedding/extractor.py` | batched extraction of original + mirrored views, cached as `.npy` |

**Key decisions.**
* **Extract once, tune many times.** Raw pooled features for every backbone, both preprocessings
  and both views are cached (40 matrices). All later tuning is NumPy on these caches — seconds
  instead of minutes per configuration.
* **One forward pass, several poolings.** Each backbone returns GAP and GeM (and CLIP's attention
  pool) from the same feature map.
* **Normalisation on the GPU.** The data loader ships uint8 tensors; each backbone applies its own
  mean/std (ImageNet vs. CLIP), so one loader feeds all five networks.

**Run it.**

```bash
python run_pipeline.py --only embedding
# ≈ 3.5 min per preprocessing on an Apple M-series GPU; later runs report "all cached"
# 40 feature matrices in data/embeddings/
```
