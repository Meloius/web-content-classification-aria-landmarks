import re
import pandas as pd
from playwright.sync_api import sync_playwright

def slugify(url: str) -> str:
    s = re.sub(r"^https?://", "", url.strip())
    s = re.sub(r"[^\w\-\.]+", "-", s)
    s = s.strip("-").lower()
    return s[:80] if len(s) > 80 else s

# Small helper to compute an XPath for an element inside the page context
_XPATH_JS = r"""
(el) => {
  function getXPath(node) {
    if (node === document.body) return '/html/body';
    if (!node || node.nodeType !== Node.ELEMENT_NODE) return '';
    let ix = 0;
    const siblings = node.parentNode ? node.parentNode.childNodes : [];
    for (let i = 0; i < siblings.length; i++) {
      const sib = siblings[i];
      if (sib === node) {
        const tag = node.tagName.toLowerCase();
        return getXPath(node.parentNode) + '/' + tag + '[' + (ix + 1) + ']';
      }
      if (sib.nodeType === Node.ELEMENT_NODE && sib.tagName === node.tagName) ix++;
    }
    return '';
  }
  return getXPath(el);
}
"""

def extract_elements(url: str, screenshot_path=None):
    rows = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.goto(url, wait_until="networkidle", timeout=60000)

        # Screenshot (full page) if requested
        if screenshot_path is not None:
            page.screenshot(path=str(screenshot_path), full_page=True)

        handles = page.query_selector_all("*")

        for i, el in enumerate(handles):
            try:
                bbox = el.bounding_box()
                if not bbox:
                    continue

                tag = (el.evaluate("e => e.tagName") or "").lower()
                el_id = el.get_attribute("id") or ""
                el_class = el.get_attribute("class") or ""
                role = el.get_attribute("role") or ""
                aria_label = el.get_attribute("aria-label") or ""
                aria_labelledby = el.get_attribute("aria-labelledby") or ""
                aria_describedby = el.get_attribute("aria-describedby") or ""
                href = el.get_attribute("href") or ""

                # Keep text extraction lightweight
                text_len = 0
                if tag not in ["script", "style"]:
                    try:
                        text = el.inner_text(timeout=500)
                        text_len = len(text.strip())
                    except Exception:
                        text_len = 0

                xpath = ""
                try:
                    xpath = el.evaluate(_XPATH_JS) or ""
                except Exception:
                    xpath = ""

                rows.append({
                    "idx": i,
                    "url": url,
                    "xpath": xpath,
                    "tagName": tag,
                    "id": el_id,
                    "class": el_class,
                    "role": role,
                    "aria_label": aria_label,
                    "aria_labelledby": aria_labelledby,
                    "aria_describedby": aria_describedby,
                    "href_present": 1 if href else 0,
                    "text_len": text_len,
                    "left": bbox["x"],
                    "top": bbox["y"],
                    "width": bbox["width"],
                    "height": bbox["height"],
                })
            except Exception:
                continue

        browser.close()

    return pd.DataFrame(rows)

def extract_elements_to_csv(url: str, csv_path, screenshot_path=None):
    df = extract_elements(url, screenshot_path=screenshot_path)
    df.to_csv(csv_path, index=False)
