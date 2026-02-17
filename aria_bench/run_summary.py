from pathlib import Path
import pandas as pd
from pandas.errors import EmptyDataError

LANDMARKS = ["banner", "navigation", "main", "complementary", "contentinfo", "form", "search", "region"]

def _safe_read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except EmptyDataError:
        return pd.DataFrame()

def summarize_run(predictions_csv: str, clusters_csv: str, top_csv: str, out_csv: str, url: str = ""):
    pred = _safe_read_csv(predictions_csv)
    clusters = _safe_read_csv(clusters_csv)
    top = _safe_read_csv(top_csv)

    row = {"url": url, "total_elements": int(len(pred)) if not pred.empty else 0}

    for lm in LANDMARKS:
        pcol = f"proba_{lm}"
        row[f"max_proba_{lm}"] = float(pred[pcol].max()) if (not pred.empty and pcol in pred.columns) else None

        row[f"candidates_{lm}"] = int((clusters.get("landmark", pd.Series(dtype=str)) == lm).sum()) if not clusters.empty else 0
        row[f"clusters_{lm}"] = int((top.get("landmark", pd.Series(dtype=str)) == lm).sum()) if not top.empty else 0

        if not top.empty and pcol in top.columns and "landmark" in top.columns:
            vals = top.loc[top["landmark"] == lm, pcol]
            row[f"rep_mean_proba_{lm}"] = float(vals.mean()) if len(vals) else None
        else:
            row[f"rep_mean_proba_{lm}"] = None

        cand = row[f"candidates_{lm}"]
        clu = row[f"clusters_{lm}"]
        row[f"reduction_{lm}"] = (1 - (clu / cand)) if cand else None

    pd.DataFrame([row]).to_csv(out_csv, index=False)
