"""Streamlit app: search yoga poses by photo (or by text) — `streamlit run app.py`.

Pipeline per query: encode → Qdrant top-fetch_k from the gallery → neighbourhood reranker → top-k.
Run `python run_pipeline.py` first so the index, projection and reranker selection exist.
"""

import json
import random

import pandas as pd
import streamlit as st
from PIL import Image

from src.data.manifest import image_dir, load_manifest
from src.data.pose_names import display_name
from src.retrieval.search_service import PoseSearch, SearchResult
from src.utils.config_loader import get_path, load_config

CONFIG = load_config()
UI = CONFIG["ui"]
st.set_page_config(page_title=UI["title"], page_icon=UI["page_icon"], layout="wide")

TEXT_EXAMPLES = [
    "a headstand",
    "Warrior II pose with arms stretched out",
    "a person balancing on one leg like a tree",
    "deep backbend on the knees",
    "seated forward fold",
]


@st.cache_resource(show_spinner="Loading models, vector index and reranker …")
def get_search() -> PoseSearch:
    return PoseSearch(CONFIG)


@st.cache_data
def get_manifest() -> pd.DataFrame:
    return load_manifest(CONFIG)


def pipeline_caption(res: SearchResult, reranked: bool) -> str:
    t = res.timings_ms
    steps = [f"embed {t.get('embed', 0):.0f} ms", f"Qdrant {t.get('vector_db', 0):.1f} ms"]
    if reranked:
        steps.append(f"rerank {t.get('rerank', 0):.2f} ms")
    return " → ".join(steps)


def results_grid(res: SearchResult, query_pose: str | None, reranked: bool, cols: int = 4) -> None:
    hits = res.hits
    caption = pipeline_caption(res, reranked)
    if query_pose:
        p = sum(h.pose == query_pose for h in hits) / max(1, len(hits))
        caption = f"Precision@{len(hits)} vs. the true pose: **{p:.0%}** · {caption}"
    st.caption(caption)
    root = image_dir(CONFIG)
    for start in range(0, len(hits), cols):
        for col, (rank, h) in zip(
            st.columns(cols), enumerate(hits[start : start + cols], start=start + 1), strict=False
        ):
            with col:
                st.image(str(root / h.path), use_container_width=True)
                badge = "" if query_pose is None else ("✅ " if h.pose == query_pose else "❌ ")
                moved = h.dense_rank - rank
                move = f" · was #{h.dense_rank} {'↑' if moved > 0 else '↓'}" if reranked and moved else ""
                score = (
                    f"score `{h.score:.3f}` · cosine `{h.dense_score:.3f}`"
                    if reranked
                    else f"cosine `{h.dense_score:.3f}`"
                )
                st.markdown(f"{badge}**{h.pose_display}**  \n{score}{move}")


def sidebar(search: PoseSearch) -> tuple[int, bool, str]:
    st.sidebar.header("Search settings")
    k = st.sidebar.slider("Results (top-k)", 4, 24, CONFIG["retrieval"]["top_k"], step=4)
    rerank = st.sidebar.toggle("Neighbourhood reranker", value=CONFIG["reranker"]["enabled"])
    vector = st.sidebar.radio(
        "Image embedding",
        ["image", "clip"],
        format_func=lambda v: {"image": "Tuned (4-CNN ensemble + LDA)", "clip": "CLIP RN50 (raw)"}[v],
    )
    p = search.params
    st.sidebar.caption(f"Reranker: fetch_k={p.fetch_k}, k={p.k}, λ={p.weight} (tuned on val)")
    final = get_path(CONFIG, "output_dir") / "metrics" / "retrieval_final.json"
    if final.exists():
        test = json.loads(final.read_text())["test"]
        st.sidebar.divider()
        st.sidebar.subheader("Held-out test metrics")
        table = pd.DataFrame({"dense": test["dense"], "+ reranker": test["dense+rerank"]}).loc[
            ["P@1", "P@5", "mAP@10", "Hit@5"]
        ]
        st.sidebar.dataframe(table.style.format("{:.3f}"), use_container_width=True)
    st.sidebar.caption("Database: gallery split · 107 poses · Qdrant (local mode)")
    return k, rerank and vector == "image", vector


def image_tab(search: PoseSearch, k: int, rerank: bool, vector: str) -> None:
    left, right = st.columns([1, 3])
    with left:
        upload = st.file_uploader("Upload a yoga photo", type=["jpg", "jpeg", "png", "webp"])
        # The most recent action wins: a newly uploaded file or a click on the random-photo button.
        if upload is not None and upload.file_id != st.session_state.get("last_upload"):
            st.session_state["last_upload"] = upload.file_id
            st.session_state["mode"] = "upload"
        if st.button("🎲 Random held-out test photo", use_container_width=True):
            test = get_manifest().query("split == 'test'")
            st.session_state["sample"] = test.iloc[random.randrange(len(test))].to_dict()
            st.session_state["mode"] = "sample"
        if upload is None and st.session_state.get("mode") == "upload":
            st.session_state.pop("mode")

        query_img, query_pose = None, None
        if st.session_state.get("mode") == "upload":
            query_img = Image.open(upload).convert("RGB")
            st.image(query_img, caption="Your query", use_container_width=True)
        elif st.session_state.get("mode") == "sample":
            s = st.session_state["sample"]
            query_img, query_pose = Image.open(s["abs_path"]).convert("RGB"), s["pose"]
            st.image(query_img, caption=f"Query (ground truth): {display_name(query_pose)}", use_container_width=True)
    with right:
        if query_img is None:
            st.info("Upload a photo of a yoga pose, or pick a random held-out test photo.")
            return
        results_grid(search.search_image(query_img, k=k, rerank=rerank, vector=vector), query_pose, rerank)


def text_tab(search: PoseSearch, k: int) -> None:
    st.write("Describe a pose in words — CLIP maps text and images into one embedding space.")
    example = st.selectbox("Examples", ["(type your own)", *TEXT_EXAMPLES])
    text = st.text_input("Describe the pose", value="" if example.startswith("(") else example)
    if text:
        results_grid(search.search_text(text, k=k), None, reranked=False)


def main() -> None:
    st.title(f"{UI['page_icon']} {UI['title']}")
    st.write(
        "Find similar yoga poses in a database of labelled photos. Images are embedded with pre-trained CNNs, "
        "retrieved from a **Qdrant** vector database and **reranked** by neighbourhood overlap."
    )
    try:
        search = get_search()
    except FileNotFoundError as e:
        st.error(f"{e}\n\nRun `python run_pipeline.py` first.")
        st.stop()
    k, rerank, vector = sidebar(search)
    tab_img, tab_txt = st.tabs(["🔍 Search by image", "💬 Search by text"])
    with tab_img:
        image_tab(search, k, rerank, vector)
    with tab_txt:
        text_tab(search, k)


main()
