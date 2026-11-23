"""Capture screenshots of the running app with headless Chromium (Playwright).

    streamlit run app.py                      # terminal 1
    python scripts/capture_screenshots.py     # terminal 2  → outputs/screenshots/*.png

One-time setup: ``pip install playwright && playwright install chromium``.
"""

import argparse
import sys
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.manifest import load_manifest  # noqa: E402
from src.utils.config_loader import get_path, load_config  # noqa: E402

RESULT_MARKER = "text=cosine >> visible=true"  # results in the *active* tab only


def wait_for_results(page: Page) -> None:
    page.locator(RESULT_MARKER).first.wait_for(timeout=120_000)
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(2_500)  # let thumbnails finish rendering


def upload_query_image(config: dict) -> Path:
    """A photo that is not in the database: a stylised image from the style-transfer study if present."""
    stylised = get_path(config, "style_dir") / "grid" / "utthita_trikonasana__great_wave__default.jpg"
    if stylised.exists():
        return stylised
    df = load_manifest(config)
    return Path(df[(df.split == "test") & (df.pose == "utthita trikonasana")].iloc[0].abs_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8501")
    args = parser.parse_args()
    config = load_config()
    out = get_path(config, "output_dir") / "screenshots"
    out.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.goto(args.url)
        page.wait_for_selector("text=Held-out test metrics", timeout=180_000)
        page.wait_for_timeout(1_500)
        page.screenshot(path=out / "home.png")

        page.set_input_files("input[type=file]", str(upload_query_image(config)))
        wait_for_results(page)
        page.screenshot(path=out / "image_upload_results.png", full_page=True)

        # Draw random held-out test photos until one shows a typical (not a worst-case) result.
        for _ in range(8):
            page.get_by_role("button", name="Random held-out test photo").click()
            page.wait_for_selector("text=Query (ground truth)", timeout=60_000)
            wait_for_results(page)
            caption = page.locator("text=Precision@").first.inner_text()
            if int(caption.split(":")[1].split("%")[0]) >= 75:
                break
        page.screenshot(path=out / "random_test_query.png", full_page=True)

        page.get_by_role("tab", name="Search by text").click()
        page.get_by_label("Describe the pose").fill("crow pose")
        page.keyboard.press("Enter")
        wait_for_results(page)
        page.screenshot(path=out / "text_search.png", full_page=True)
        browser.close()
    print(sorted(f.name for f in out.glob("*.png")))


if __name__ == "__main__":
    main()
