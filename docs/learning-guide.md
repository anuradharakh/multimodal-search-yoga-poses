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

---

## Step 4 — Evaluation and tuning on the validation split

**Goal.** Decide, with numbers, which embedding pipeline to put in the database.

| Module | What it does |
|---|---|
| `src/embedding/postprocess.py` | `SearchConfig` (one tunable configuration), flip-TTA, PCA-whitening / LDA `Projector`, 4-CNN ensemble |
| `src/evaluation/metrics.py` | exact cosine top-k; P@k, mAP@k, Hit@5 (relevant = same pose) |
| `src/evaluation/tuning.py` | scores all 176 configurations on **val**, ablation and per-backbone tables |

**What the grid shows** (`outputs/metrics/tuning_*.csv`, validation mAP@10):

| Change | mAP@10 |
|---|---|
| ResNet-50, GAP, centre-crop (ImageNet default) | 0.089 |
| + letterbox, + flip-TTA | 0.091 |
| + GeM pooling | 0.121 |
| + PCA-whitening | 0.120 |
| + **LDA projection** (supervised, CNN frozen) | **0.399** |
| backbone → CLIP-RN50 | 0.469 |
| backbone → ensemble of 4 CNNs → **selected** | **0.499** |

**Key decisions.**
* **Frozen CNN features find photos that *look* alike** (clothing, background), not poses that look
  alike — every backbone scores ≤ 0.14 without a projection.
* **LDA is the decisive step.** The gallery is labelled (like a product catalogue), so a ≤ 106-d
  linear map can be learned on it that emphasises between-pose differences: ×3–4.5 mAP for every
  backbone. Only gallery labels are used; val and test labels never touch the fit.
* **Select on val, report on test.** The selection is saved to `outputs/metrics/selected_config.json`.

**Run it.**

```bash
python run_pipeline.py --only tuning
# 176 configurations scored on val
# Selected ensemble|mixed|pad|tta|lda0.1 — val P@1 0.678 · mAP@10 0.499
```

---

## Step 5 — Indexing in a vector database (Qdrant)

**Goal.** Store every image's vector so queries are a database lookup, not a NumPy scan.

| Module | What it does |
|---|---|
| `src/indexing/vector_store.py` | thin Qdrant wrapper: named vectors, payload filters, single and batch search |
| `src/indexing/index_builder.py` | embeds all 5,774 images with the selected pipeline, saves the fitted projection, upserts |

**Design.** One point per image, two **named vectors** — `image` (106-d tuned embedding) and `clip`
(1024-d CLIP space for text queries) — plus a payload (`path`, `pose`, `pose_display`, `split`).
The `split` payload lets evaluation restrict results to the gallery with a filter, while the app
searches everything.

**Why Qdrant (local mode)?** Named vectors and payload filtering out of the box, no server to run
(data lives in `data/indexes/qdrant`), and the same client code works against a Qdrant server or
Qdrant Cloud by changing one line.

**Run it.**

```bash
python run_pipeline.py --only indexing
# Indexed 5,774 points · vectors {'image': 106, 'clip': 1024} · 3.5 s
```

---

## Step 6 — Retrieval and reranking

**Goal.** Turn the index into a two-stage search engine — *retrieve, then rerank* — exactly the
shape of a text RAG retriever, and prove each stage with numbers.

```
query image ─► QueryEncoder (same pipeline + saved projection) ─► Qdrant top-fetch_k (gallery filter)
            ─► NeighborhoodReranker ─► top-k results
```

| Module | What it does |
|---|---|
| `src/retrieval/query_encoder.py` | embeds an uploaded photo (or text, for CLIP) exactly like the database images |
| `src/retrieval/reranker.py` | gallery-neighbour table + **neighbourhood-overlap reranker**; α-query expansion for comparison |
| `src/retrieval/rrf.py` | Reciprocal Rank Fusion (for the hybrid-retrieval comparison) |
| `src/retrieval/search_service.py` | `PoseSearch`: encode → Qdrant → rerank, with per-stage timings |
| `src/evaluation/retrieval_eval.py` | reranker tuning on val, alternatives, final test evaluation through Qdrant |
| `src/evaluation/text_search.py` | text → image evaluation with CLIP prompts |

**How the reranker works.** Dense retrieval scores each gallery image *independently*. The reranker
adds context: a candidate that really shows the query's pose has nearest neighbours that overlap
with the query's own top results. Score = `(1 − λ)·cosine + λ·overlap`, using a neighbour table
precomputed at indexing time (cost ≈ 0.02 ms per query). There is no image equivalent of a text
cross-encoder; neighbourhood (k-reciprocal-style) reranking is the standard image-retrieval tool.

**Selecting it with a guardrail.** On val, pushing λ up keeps raising mAP@10 (to ~0.59) but drops
**Hit@5** from 0.85 to 0.77 — the reranker "commits" to the dominant cluster, and when that cluster
is the wrong pose the user sees *no* correct result. So the selection rule (config `reranker.selection`)
maximises mAP@10 **subject to Hit@5 falling ≤ 2 points** → `fetch_k=20, k=30, λ=0.3`.

**Alternatives considered (val, mAP@10):**

| Method | mAP@10 |
|---|---|
| dense retrieval only | 0.499 |
| hybrid: tuned + raw CLIP, fused with RRF | **0.254** ✗ |
| α-query expansion (k = 10) | 0.549 |
| **neighbourhood reranker (selected)** | **0.565** |

*Why hybrid RRF fails here but works in text RAG:* BM25 and dense retrieval are both decent and make
*different* mistakes. Raw CLIP is far weaker than the tuned vector (0.12 vs. 0.50 mAP) and RRF gives
both lists equal votes, so it mostly imports CLIP's errors.

**Final result — held-out test split, through Qdrant:**

| | P@1 | P@5 | mAP@10 | Hit@5 |
|---|---|---|---|---|
| dense only | 0.697 | 0.628 | 0.525 | 0.870 |
| **dense + reranker** | **0.706** | **0.656** | **0.584** | 0.849 |

**Run it.**

```bash
python run_pipeline.py --only indexing retrieval_eval
# Reranker selected on val (with Hit@5 guardrail): RerankerParams(fetch_k=20, k=30, weight=0.3)
# TEST dense         P@1 0.697 · P@5 0.628 · mAP@10 0.525 · Hit@5 0.870
# TEST dense+rerank  P@1 0.706 · P@5 0.656 · mAP@10 0.584 · Hit@5 0.849
```

---

## Step 7 — Web application

**Goal.** A UI that accepts an image upload and shows similar poses — and makes the pipeline visible.

| File | What it does |
|---|---|
| `app.py` | Streamlit UI: upload or random held-out test photo, text search tab, reranker toggle |
| `scripts/capture_screenshots.py` | headless-Chromium screenshots into `outputs/screenshots/` |

**Explainability, as in a RAG UI.** Each result shows its final score, its raw cosine similarity
and how the reranker moved it (`was #7 ↑`); the caption shows per-stage timings
(`embed → Qdrant → rerank`); the sidebar shows the tuned reranker settings and the test metrics.
For a dataset query, ✅/❌ marks each result against the true pose.

**Run it.**

```bash
streamlit run app.py                      # http://localhost:8501
python scripts/capture_screenshots.py     # with the app running
```

**Try this.** Toggle the reranker on a random test photo and watch results move. On the stylised
Triangle-pose upload (screenshot), the reranker reinforces the *Revolved* Triangle cluster — a
reminder of its failure mode: it strengthens whatever pose dominates the candidate list.
