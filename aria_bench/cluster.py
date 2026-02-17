from __future__ import annotations
import pandas as pd
import numpy as np

LANDMARKS = ["banner", "navigation", "main", "complementary", "contentinfo", "form", "search", "region"]

def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return (inter / union) if union > 0 else 0.0

def _to_boxes(df: pd.DataFrame):
    x1 = df["left"].to_numpy(float)
    y1 = df["top"].to_numpy(float)
    x2 = (df["left"] + df["width"]).to_numpy(float)
    y2 = (df["top"] + df["height"]).to_numpy(float)
    return np.stack([x1, y1, x2, y2], axis=1)

def cluster_candidates(
    predictions_csv: str,
    out_clusters_csv: str,
    out_top_csv: str,
    min_prob: float = 0.50,
    iou_threshold: float = 0.20,
    max_per_class: int = 300,
):
    df = pd.read_csv(predictions_csv)

    for c in ["left", "top", "width", "height"]:
        df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0.0)

    df = df[(df["width"] > 1) & (df["height"] > 1)].copy()

    all_clusters = []
    all_top = []

    for lm in LANDMARKS:
        pcol = f"proba_{lm}"
        if pcol not in df.columns:
            continue

        cand = df[df[pcol] >= float(min_prob)].copy()
        if cand.empty:
            continue

        cand = cand.sort_values(pcol, ascending=False).head(int(max_per_class)).copy()
        cand.reset_index(drop=True, inplace=True)

        boxes = _to_boxes(cand)
        n = len(cand)
        cluster_id = [-1] * n
        cur = 0

        for i in range(n):
            if cluster_id[i] != -1:
                continue
            cluster_id[i] = cur
            rep_box = boxes[i]
            for j in range(i + 1, n):
                if cluster_id[j] != -1:
                    continue
                if iou(rep_box, boxes[j]) >= float(iou_threshold):
                    cluster_id[j] = cur
            cur += 1

        cand["landmark"] = lm
        cand["cluster_id"] = cluster_id
        all_clusters.append(cand)

        top = cand.sort_values(pcol, ascending=False).drop_duplicates(subset=["cluster_id"]).copy()
        top["rep_proba"] = top[pcol]
        all_top.append(top)

    if all_clusters:
        clusters_out = pd.concat(all_clusters, ignore_index=True)
    else:
        clusters_out = pd.DataFrame(columns=["landmark", "cluster_id", "left", "top", "width", "height"])

    if all_top:
        top_out = pd.concat(all_top, ignore_index=True)
    else:
        top_out = pd.DataFrame(columns=["landmark", "cluster_id", "left", "top", "width", "height", "rep_proba"])

    clusters_out.to_csv(out_clusters_csv, index=False)
    top_out.to_csv(out_top_csv, index=False)
