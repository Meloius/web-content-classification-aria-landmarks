import pandas as pd
import joblib

META_COLS = {
    "class", "url", "xpath", "screenshot", "tagName", "role", "className",
    "parent_landmark", "label"
}

def load_model(model_dir: str):
    extractor = joblib.load(f"{model_dir}/extractor-rf-True-False.sav")
    pipeline = joblib.load(f"{model_dir}/pipeline-rf-True-False.sav")
    return extractor, pipeline

def predict_features_csv(features_csv: str, model_dir: str, out_csv: str):
    df = pd.read_csv(features_csv)

    feature_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feature_cols].copy()

    extractor, pipeline = load_model(model_dir)

    X_t = extractor.transform(X)
    probs = pipeline.predict_proba(X_t)
    preds = pipeline.predict(X_t)

    out = df.copy()
    out["pred"] = preds

    classes = list(getattr(pipeline, "classes_", range(probs.shape[1])))
    for i, cls in enumerate(classes):
        out[f"proba_{cls}"] = probs[:, i]

    out.to_csv(out_csv, index=False)
