import argparse
from pathlib import Path
from aria_bench.fetch import extract_elements_to_csv, slugify
from aria_bench.build_features import build_features_csv
from aria_bench.predict import predict_features_csv
from aria_bench.overlay import make_overlay

def main():
    p = argparse.ArgumentParser(prog="aria_bench")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run-url", help="Fetch a URL and export per-element table + screenshot")
    run.add_argument("url")
    run.add_argument("--out", default="outputs", help="Output root folder")

    feat = sub.add_parser("build-features", help="Convert elements.csv into training-schema features.csv")
    feat.add_argument("elements_csv", help="Path to elements.csv")
    feat.add_argument("--out", required=True, help="Output features.csv path")
    feat.add_argument("--screenshot", default="", help="Screenshot filename to store in 'screenshot' column")

    pred = sub.add_parser("predict", help="Predict landmark class probabilities from features.csv")
    pred.add_argument("features_csv")
    pred.add_argument("--model-dir", default="models", help="Folder containing extractor/pipeline .sav files")
    pred.add_argument("--out", required=True, help="Output predictions CSV")

    ov = sub.add_parser("overlay", help="Draw top predicted landmarks on screenshot")
    ov.add_argument("predictions_csv")
    ov.add_argument("--screenshot", required=True, help="Path to screenshot.png")
    ov.add_argument("--out", required=True, help="Output overlay image path")
    ov.add_argument("--top-k", type=int, default=1, help="Top K elements per landmark to draw")

    args = p.parse_args()

    if args.cmd == "run-url":
        out_root = Path(args.out)
        out_dir = out_root / slugify(args.url)
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_path = out_dir / "elements.csv"
        screenshot_path = out_dir / "screenshot.png"

        extract_elements_to_csv(args.url, csv_path, screenshot_path=screenshot_path)
        print(f"Wrote: {csv_path}")
        print(f"Wrote: {screenshot_path}")

    elif args.cmd == "build-features":
        build_features_csv(args.elements_csv, args.out, screenshot_name=args.screenshot)
        print(f"Wrote: {args.out}")

    elif args.cmd == "predict":
        predict_features_csv(args.features_csv, args.model_dir, args.out)
        print(f"Wrote: {args.out}")

    elif args.cmd == "overlay":
        make_overlay(args.predictions_csv, args.screenshot, args.out, top_k=args.top_k)
        print(f"Wrote: {args.out}")

if __name__ == "__main__":
    main()
