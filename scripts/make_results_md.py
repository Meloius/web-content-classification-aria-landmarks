import json
import pandas as pd
from pathlib import Path

def md(df): 
    return df.to_markdown(index=False)

def _as_float(x):
    """Convert value that may be float/int or list of floats into a single float (mean)."""
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, list) and len(x) > 0:
        # mean of list
        return sum(float(v) for v in x) / len(x)
    return None

def main():
    artifacts = Path("docs/artifacts")
    report_path = artifacts / "report-rf-True-False.json"
    cv_path = artifacts / "cv-all.csv"
    out_path = Path("docs/results.md")

    if not report_path.exists() or not cv_path.exists():
        out_path.write_text(
            "# Results\n\nMissing artifacts.\n\n"
            "Add:\n- docs/artifacts/report-rf-True-False.json\n- docs/artifacts/cv-all.csv\n"
            "Then run: python scripts/make_results_md.py\n"
        )
        print("Artifacts missing; wrote placeholder docs/results.md")
        return

    report = json.loads(report_path.read_text())

    rows = []
    for k, v in report.items():
        if not isinstance(v, dict):
            continue

        prec = _as_float(v.get("precision"))
        rec = _as_float(v.get("recall"))
        f1 = _as_float(v.get("f1-score"))
        sup = v.get("support", 0)

        # Keep rows that look like class/avg metric dicts
        if prec is None or rec is None or f1 is None:
            continue

        # support may be list too; take sum or mean; we'll sum if list
        if isinstance(sup, list) and len(sup) > 0:
            try:
                sup_val = int(sum(int(s) for s in sup))
            except Exception:
                sup_val = int(len(sup))
        else:
            try:
                sup_val = int(sup)
            except Exception:
                sup_val = 0

        rows.append({
            "class": k,
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1": round(f1, 3),
            "support": sup_val,
        })

    per_class = pd.DataFrame(rows)

    order = ["banner","complementary","contentinfo","form","main","navigation","region","search","macro avg","weighted avg"]
    if not per_class.empty:
        per_class["__ord"] = per_class["class"].apply(lambda x: order.index(x) if x in order else 999)
        per_class = per_class.sort_values("__ord").drop(columns="__ord")

    cv = pd.read_csv(cv_path)
    if "classifier" in cv.columns:
        cv_rf = cv[cv["classifier"].astype(str).str.contains("rf", case=False, na=False)].copy()
    else:
        cv_rf = cv.copy()

    keep = [c for c in ["classifier","accuracy","macro_f1","weighted_f1","params"] if c in cv_rf.columns]
    if keep:
        cv_rf = cv_rf[keep]

    out = []
    out.append("# Results\n")
    out.append("## Cross-validation summary (RF)\n")
    out.append(md(cv_rf.head(15)))
    out.append("\n\n## Per-class metrics (RF)\n")
    out.append(md(per_class))
    out.append("\n\n### Notes\n")
    out.append("- If metrics are stored as lists (e.g., per-fold), this report shows the mean.\n")

    out_path.write_text("\n".join(out))
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
