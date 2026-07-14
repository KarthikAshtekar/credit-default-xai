"""Capture Streamlit dashboard screenshots for local UI review."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from playwright.sync_api import Page, TimeoutError, sync_playwright

DEFAULT_URL = "http://localhost:8510"
DEFAULT_OUTPUT_DIR = Path("reports/ui_review")


def _parse_viewport(value: str) -> tuple[int, int]:
    try:
        width, height = value.lower().split("x", maxsplit=1)
        return int(width), int(height)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Viewport must be formatted as WIDTHxHEIGHT.") from exc


def _wait_for_streamlit(page: Page) -> None:
    page.wait_for_selector("[data-testid='stApp']", timeout=60_000)
    page.wait_for_timeout(1_500)


def _click_tab(page: Page, name: str) -> None:
    page.get_by_role("tab", name=name).click()
    page.wait_for_timeout(1_000)


def _capture(page: Page, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(path), full_page=False)


def _scroll_to_text(page: Page, text: str) -> None:
    try:
        locator = page.get_by_text(text, exact=True).first
        locator.wait_for(timeout=10_000)
        locator.evaluate("element => element.scrollIntoView({block: 'start', inline: 'nearest'})")
        page.wait_for_timeout(750)
    except TimeoutError:
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(750)


def _select_demo_profile(page: Page, profile_name: str) -> None:
    profile_selectors = [
        page.get_by_label("Demo applicant profile"),
        page.locator("[data-testid='stSelectbox']")
        .filter(has_text="Demo applicant profile")
        .get_by_role("combobox"),
    ]
    for selector in profile_selectors:
        try:
            selector.click(timeout=5_000)
            page.get_by_text(profile_name, exact=True).click(timeout=5_000)
            page.wait_for_timeout(750)
            return
        except TimeoutError:
            continue


def _generate_applicant_result(page: Page) -> None:
    _click_tab(page, "Applicant Report")
    _select_demo_profile(page, "High-delay profile")

    action = page.get_by_role("button", name=re.compile("Predict|Generate", re.IGNORECASE))
    action.first.click()
    page.wait_for_timeout(4_000)
    _scroll_to_text(page, "Predicted default risk")


def capture_dashboard(
    url: str,
    output_dir: Path,
    suffix: str,
    viewport: tuple[int, int],
) -> list[Path]:
    width, height = viewport
    captured: list[Path] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": width, "height": height})
        page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        _wait_for_streamlit(page)

        _click_tab(page, "Applicant Report")
        applicant_path = output_dir / f"01_applicant_report_{suffix}.png"
        _capture(page, applicant_path)
        captured.append(applicant_path)

        _generate_applicant_result(page)

        _click_tab(page, "Improvement Guidance")
        guidance_path = output_dir / f"02_improvement_guidance_{suffix}.png"
        _capture(page, guidance_path)
        captured.append(guidance_path)

        _scroll_to_text(page, "Scenario simulation")
        scenario_path = output_dir / f"05_improvement_scenario_{suffix}.png"
        _capture(page, scenario_path)
        captured.append(scenario_path)

        _click_tab(page, "Model Governance")
        governance_path = output_dir / f"03_model_governance_{suffix}.png"
        _capture(page, governance_path)
        captured.append(governance_path)

        _click_tab(page, "Applicant Report")
        _scroll_to_text(page, "Predicted default risk")
        filled_path = output_dir / f"04_applicant_report_filled_{suffix}.png"
        _capture(page, filled_path)
        captured.append(filled_path)

        browser.close()
    return captured


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture dashboard screenshots for UI review.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--suffix", default="initial")
    parser.add_argument("--viewport", type=_parse_viewport, default=(1366, 768))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    captured = capture_dashboard(args.url, args.output_dir, args.suffix, args.viewport)
    for path in captured:
        print(path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
