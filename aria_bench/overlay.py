from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

LANDMARKS = ["banner", "navigation", "main", "complementary", "contentinfo", "form", "search", "region"]

def _load_font(size: int = 18):
    try:
        return ImageFont.truetype("Arial.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()

def _text_wh(draw: ImageDraw.ImageDraw, text: str, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return (bbox[2] - bbox[0], bbox[3] - bbox[1])

def make_overlay(predictions_csv: str, screenshot_path: str, out_path: str, top_k: int = 1):
    df = pd.read_csv(predictions_csv)

    for c in ["left", "top", "width", "height"]:
        df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0.0)

    img = Image.open(screenshot_path).convert("RGBA")
    draw = ImageDraw.Draw(img)
    font = _load_font(16)

    img_w, img_h = img.size

    for lm in LANDMARKS:
        proba_col = f"proba_{lm}"
        if proba_col not in df.columns:
            continue

        top = df.sort_values(proba_col, ascending=False).head(top_k)

        for _, row in top.iterrows():
            x = float(row["left"]); y = float(row["top"])
            w = float(row["width"]); h = float(row["height"])
            if w <= 1 or h <= 1:
                continue

            # Clamp bbox to image bounds
            x0 = max(0.0, min(x, img_w - 1))
            y0_box = max(0.0, min(y, img_h - 1))
            x1 = max(0.0, min(x + w, img_w - 1))
            y1 = max(0.0, min(y + h, img_h - 1))

            if x1 <= x0 or y1 <= y0_box:
                continue

            draw.rectangle([x0, y0_box, x1, y1], outline="red", width=3)

            label = f"{lm} ({float(row[proba_col]):.2f})"
            tw, th = _text_wh(draw, label, font=font)
            pad = 4

            # Prefer label above the box; if not enough room, put it below
            above_top = y0_box - (th + 2 * pad)
            if above_top >= 0:
                ly0 = above_top
                ly1 = y0_box
            else:
                ly0 = min(img_h - 1, y1)
                ly1 = min(img_h - 1, y1 + th + 2 * pad)

            # Clamp label rect and ensure valid ordering
            lx0 = x0
            lx1 = min(img_w - 1, x0 + tw + 2 * pad)
            ly0 = max(0.0, min(ly0, img_h - 1))
            ly1 = max(0.0, min(ly1, img_h - 1))
            if ly1 < ly0:
                ly0, ly1 = ly1, ly0
            if lx1 < lx0:
                lx0, lx1 = lx1, lx0

            draw.rectangle([lx0, ly0, lx1, ly1], fill="red")
            draw.text((lx0 + pad, ly0 + pad), label, fill="white", font=font)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
