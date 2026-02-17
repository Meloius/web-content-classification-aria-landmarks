from pathlib import Path
import sys
import streamlit as st
import pandas as pd

# Ensure repo root is on PYTHONPATH for Streamlit
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aria_bench.fetch import slugify, extract_elements_to_csv
from aria_bench.build_features import build_features_csv
from aria_bench.predict import predict_features_csv
from aria_bench.cluster import cluster_candidates
from aria_bench.run_summary import summarize_run
from aria_bench.overlay import make_overlay_from_rows

st.set_page_config(page_title="ARIA Landmark Identification Demo", layout="wide")

st.title("ARIA Landmark Identification (Benchmark Demo)")
st.write("Paste a URL, run extraction → features → prediction → clustering → overlay.")

url = st.text_input("Website URL", value="https://example.com")

colA, colB, colC = st.columns(3)
with colA:
    min_prob = st.slider("Min probability threshold", 0.0, 0.95, 0.50, 0.05)
with colB:
    iou_thr = st.slider("IoU clustering threshold", 0.0, 0.90, 0.20, 0.05)
with colC:
    max_per_class = st.slider("Max candidates per landmark", 20, 500, 300, 10)

model_dir = st.text_input("Model directory", value="models")

model_ok = Path(model_dir).exists() and any(Path(model_dir).glob("*.sav"))
if not model_ok:
    st.warning(
        "Model files are not included in this repo. "
        "To enable prediction, place trained artifacts in ./models "
        "(extractor-*.sav and pipeline-*.sav)."
    )

run = st.button("Run")

if run:
    if not url.startswith("http"):
        st.error("Please enter a valid URL starting with http/https.")
        st.stop()

    out_dir = Path("outputs") / slugify(url)
    out_dir.mkdir(parents=True, exist_ok=True)

    elements_csv = out_dir / "elements.csv"
    screenshot_png = out_dir / "screenshot.png"
    features_csv = out_dir / "features.csv"
    predictions_csv = out_dir / "predictions.csv"

    clusters_csv = out_dir / "clusters.csv"
    top_csv = out_dir / "top_per_cluster.csv"
    summary_csv = out_dir / "run_summary.csv"
    overlay_png = out_dir / "overlay.png"

    with st.status("Running pipeline...", expanded=True) as status:
        st.write("1) Fetch + screenshot + elements.csv")
        extract_elements_to_csv(url, elements_csv, screenshot_path=screenshot_png)

        st.write("2) Build features.csv (training schema)")
        build_features_csv(str(elements_csv), str(features_csv), screenshot_name="screenshot.png")

        if not model_ok:
            status.update(label="Stopped (model missing). Extraction + features completed.", state="complete")
            st.info(f"Saved to `{out_dir}`. Add model files to `{model_dir}/` to enable prediction + overlay.")
        else:
            st.write("3) Predict landmark probabilities")
            predict_features_csv(str(features_csv), model_dir, str(predictions_csv))

            st.write("4) Cluster candidates (paper-style) + select top per cluster")
            cluster_candidates(
                str(predictions_csv),
                str(clusters_csv),
                str(top_csv),
                min_prob=float(min_prob),
                iou_threshold=float(iou_thr),
                max_per_class=int(max_per_class),
            )

            st.write("5) Save run summary (table-ready)")
            summarize_run(
                str(predictions_csv),
                str(clusters_csv),
                str(top_csv),
                str(summary_csv),
                url=url
            )

            st.write("6) Create overlay.png (using top-per-cluster)")
            make_overlay_from_rows(str(top_csv), str(screenshot_png), str(overlay_png))

            status.update(label="Done!", state="complete")

    # Display outputs
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Screenshot / Overlay")
        if overlay_png.exists():
            st.image(str(overlay_png), use_container_width=True)
        elif screenshot_png.exists():
            st.image(str(screenshot_png), use_container_width=True)

    with col2:
        st.subheader("Saved outputs")
        st.write(f"Folder: `{out_dir}`")

        if elements_csv.exists():
            st.download_button("Download elements.csv", data=elements_csv.read_bytes(), file_name="elements.csv")
        if features_csv.exists():
            st.download_button("Download features.csv", data=features_csv.read_bytes(), file_name="features.csv")

        if predictions_csv.exists():
            st.download_button("Download predictions.csv", data=predictions_csv.read_bytes(), file_name="predictions.csv")
        if clusters_csv.exists():
            st.download_button("Download clusters.csv", data=clusters_csv.read_bytes(), file_name="clusters.csv")
        if top_csv.exists():
            st.download_button("Download top_per_cluster.csv", data=top_csv.read_bytes(), file_name="top_per_cluster.csv")
        if summary_csv.exists():
            st.download_button("Download run_summary.csv", data=summary_csv.read_bytes(), file_name="run_summary.csv")
            st.subheader("Run summary")
            st.dataframe(pd.read_csv(summary_csv))

        if overlay_png.exists():
            st.download_button("Download overlay.png", data=overlay_png.read_bytes(), file_name="overlay.png")
