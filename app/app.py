from pathlib import Path
import sys

# Ensure repo root is on PYTHONPATH for Streamlit
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import os
from pathlib import Path
import streamlit as st

from aria_bench.fetch import slugify, extract_elements_to_csv
from aria_bench.build_features import build_features_csv
from aria_bench.predict import predict_features_csv
from aria_bench.overlay import make_overlay

st.set_page_config(page_title="ARIA Landmark Identification Demo", layout="wide")

st.title("ARIA Landmark Identification (Benchmark Demo)")
st.write("Paste a URL, run extraction → features → prediction → overlay.")

url = st.text_input("Website URL", value="https://example.com")
top_k = st.slider("Top-K boxes per landmark", 1, 5, 1)

model_dir = st.text_input("Model directory", value="models")

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
    overlay_png = out_dir / "overlay.png"

    with st.status("Running pipeline...", expanded=True) as status:
        st.write("1) Fetch + screenshot + elements.csv")
        extract_elements_to_csv(url, elements_csv, screenshot_path=screenshot_png)

        st.write("2) Build features.csv (training schema)")
        build_features_csv(str(elements_csv), str(features_csv), screenshot_name="screenshot.png")

        st.write("3) Predict landmark probabilities")
        predict_features_csv(str(features_csv), model_dir, str(predictions_csv))

        st.write("4) Create overlay.png")
        make_overlay(str(predictions_csv), str(screenshot_png), str(overlay_png), top_k=top_k)

        status.update(label="Done!", state="complete")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Overlay")
        st.image(str(overlay_png), use_column_width=True)

    with col2:
        st.subheader("Outputs")
        st.write(f"Saved to: `{out_dir}`")
        st.download_button("Download predictions.csv", data=predictions_csv.read_bytes(), file_name="predictions.csv")
        st.download_button("Download overlay.png", data=overlay_png.read_bytes(), file_name="overlay.png")
