import pandas as pd
import numpy as np

# Paste your header-derived ordered columns here (exactly as training expects).
FEATURE_COLUMNS = [
"a_count","abbr_count","acronym_count","address_count","applet_count","area_count","article_count","aside_count","audio_count",
"avg_height","avg_left","avg_top","avg_width",
"b_count","base_count","basefont_count","bdi_count","bdo_count","big_count","blockquote_count","body_count","br_count",
"button_count","canvas_count","caption_count","center_count","childs_count","cite_count","className","code_count","col_count",
"colgroup_count","data_count","datalist_count","dd_count","del_count","details_count","dfn_count","dialog_count","dir_count",
"div_count","dl_count","dt_count","em_count","embed_count","fieldset_count","figcaption_count","figure_count","font_count",
"footer_count","form_count","frame_count","frameset_count","h1_count","h2_count","h3_count","h4_count","h5_count","h6_count",
"head_count","header_count","height","hr_count","html_count","i_count","iframe_count","img_count","input_count","ins_count",
"kbd_count","label","label_count","left","legend_count","li_count","link_count","main_count","map_count","mark_count",
"meta_count","meter_count","nav_count","noframes_count","noscript_count","object_count","ol_count","optgroup_count","option_count",
"output_count","p_count","param_count","parent_landmark","picture_count","pre_count","progress_count","q_count","role","rp_count",
"rt_count","ruby_count","s_count","samp_count","script_count",
"sd_height","sd_left","sd_top","sd_width",
"section_count","select_count","small_count","source_count","span_count","strike_count","strong_count","style_count","sub_count",
"summary_count","sup_count","svg_count","table_count","tagName","tbody_count","td_count","template_count","textarea_count",
"tfoot_count","th_count","thead_count","time_count","title_count","top","tr_count","track_count","tt_count","u_count","ul_count",
"url","var_count","video_count","wbr_count",
"weighted_height","weighted_left","weighted_top","weighted_width",
"width","window_elements_count","window_height","word_count","xpath","screenshot","website_total_elements","class"
]

# Tags we count into *_count. Derived from columns that end with _count.
TAG_COUNT_COLS = [c for c in FEATURE_COLUMNS if c.endswith("_count")]

def _safe_std(x: pd.Series) -> float:
    # Match training style: standard deviation, but avoid NaN when 0/1 elements
    if len(x) < 2:
        return 0.0
    v = float(x.std(ddof=0))
    return 0.0 if np.isnan(v) else v

def build_features_from_elements(elements_df: pd.DataFrame, screenshot_name: str = "") -> pd.DataFrame:
    """
    Convert per-element extraction (elements.csv) to model feature schema.
    - Counts tags across the whole page (same count copied onto each row).
    - Computes avg/sd of bbox stats across page (copied onto each row).
    - Computes weighted_* as normalized bbox values (0..1) using viewport and page max.
    """

    df = elements_df.copy()

    # Required base columns from extractor
    # Map what we already have
    df["className"] = df.get("class", "")
    df["tagName"] = df.get("tagName", "").fillna("").astype(str).str.lower()
    df["role"] = df.get("role", "").fillna("").astype(str)
    df["url"] = df.get("url", "")
    df["xpath"] = df.get("xpath", "")
    df["left"] = pd.to_numeric(df.get("left", 0), errors="coerce").fillna(0)
    df["top"] = pd.to_numeric(df.get("top", 0), errors="coerce").fillna(0)
    df["width"] = pd.to_numeric(df.get("width", 0), errors="coerce").fillna(0)
    df["height"] = pd.to_numeric(df.get("height", 0), errors="coerce").fillna(0)

    # word_count: approximate from text_len if present, else 0
    text_len = pd.to_numeric(df.get("text_len", 0), errors="coerce").fillna(0)
    # crude conversion: ~5 chars per word average
    df["word_count"] = (text_len / 5.0).round().astype(int)

    # label + label_count: use aria_label if present (you can refine later)
    aria_label = df.get("aria_label", "")
    if isinstance(aria_label, pd.Series):
        df["label"] = aria_label.fillna("").astype(str)
        df["label_count"] = (df["label"].str.len() > 0).astype(int)
    else:
        df["label"] = ""
        df["label_count"] = 0

    # parent_landmark and class (target label) are unknown for inference
    df["parent_landmark"] = ""
    df["class"] = ""  # set later by prediction; training used this as label

    # Screenshot column: store filename (not full path) for portability
    df["screenshot"] = screenshot_name

    # Page-level totals
    website_total = int(len(df))
    df["website_total_elements"] = website_total
    df["window_elements_count"] = website_total

    # window_height: we don’t have it from extractor yet; approximate using max(bottom)
    page_bottom = float((df["top"] + df["height"]).max()) if website_total else 0.0
    df["window_height"] = page_bottom

    # Tag counts across the page
    tag_counts = df["tagName"].value_counts().to_dict()
    for col in TAG_COUNT_COLS:
        tag = col.replace("_count", "")
        df[col] = int(tag_counts.get(tag, 0))

    # childs_count: we don't have child relationships yet; default 0 for now
    df["childs_count"] = 0

    # avg_* and sd_* across the page
    df["avg_left"] = float(df["left"].mean()) if website_total else 0.0
    df["avg_top"] = float(df["top"].mean()) if website_total else 0.0
    df["avg_width"] = float(df["width"].mean()) if website_total else 0.0
    df["avg_height"] = float(df["height"].mean()) if website_total else 0.0

    df["sd_left"] = _safe_std(df["left"])
    df["sd_top"] = _safe_std(df["top"])
    df["sd_width"] = _safe_std(df["width"])
    df["sd_height"] = _safe_std(df["height"])

    # Weighted features: normalize to [0,1] using page extents (safer than fixed viewport)
    page_right = float((df["left"] + df["width"]).max()) if website_total else 1.0
    page_bottom = float((df["top"] + df["height"]).max()) if website_total else 1.0
    page_right = page_right if page_right > 0 else 1.0
    page_bottom = page_bottom if page_bottom > 0 else 1.0

    df["weighted_left"] = (df["left"] / page_right).clip(0, 1)
    df["weighted_top"] = (df["top"] / page_bottom).clip(0, 1)
    df["weighted_width"] = (df["width"] / page_right).clip(0, 1)
    df["weighted_height"] = (df["height"] / page_bottom).clip(0, 1)

    # Ensure every required column exists
    for c in FEATURE_COLUMNS:
        if c not in df.columns:
            df[c] = 0

    # Order columns exactly as training expects
    out = df[FEATURE_COLUMNS].copy()
    return out

def build_features_csv(elements_csv_path: str, out_csv_path: str, screenshot_name: str = ""):
    elements_df = pd.read_csv(elements_csv_path)
    features_df = build_features_from_elements(elements_df, screenshot_name=screenshot_name)
    features_df.to_csv(out_csv_path, index=False)
