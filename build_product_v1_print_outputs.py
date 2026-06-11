from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from strawberry_vigour_polygon_report.src.growth_stage import growth_stage_for_date


ROOT = Path(__file__).resolve().parent
PRODUCT_DIR = ROOT / "human_anchored_vigour_zone_preview_product"
OUTPUT_DIR = ROOT / "daily_outputs"
FIELD_NAME = "Strawberry 1"
DATES = ["2026-05-08", "2026-05-12", "2026-05-18", "2026-05-22", "2026-05-27", "2026-05-29"]
STATUS_COLORS = {
    "Stable": (70, 160, 85),
    "Monitor": (0, 170, 255),
    "Expanding": (40, 115, 230),
    "Review": (35, 35, 210),
}


def read_summary() -> dict[str, dict[str, str]]:
    path = PRODUCT_DIR / "product_audit_summary.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["date"]: row for row in csv.DictReader(handle)}


def read_temporal() -> dict[str, dict[str, str]]:
    path = ROOT / "temporal_analytics.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["date"]: row for row in csv.DictReader(handle)}


def load(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def fit(image: np.ndarray, size: tuple[int, int], background: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    target_w, target_h = size
    canvas = np.full((target_h, target_w, 3), background, dtype=np.uint8)
    h, w = image.shape[:2]
    scale = min(target_w / max(1, w), target_h / max(1, h))
    resized = cv2.resize(image, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    y = (target_h - resized.shape[0]) // 2
    x = (target_w - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def put_text(image: np.ndarray, text: str, org: tuple[int, int], scale: float, color=(35, 39, 46), thickness=2) -> None:
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def concern_density(seed_polygons: np.ndarray, raw_shape: tuple[int, int, int]) -> np.ndarray:
    mask = np.any(seed_polygons > 0, axis=2).astype(np.float32)
    if not np.any(mask):
        return np.zeros(raw_shape, dtype=np.uint8)
    blurred = cv2.GaussianBlur(mask, (0, 0), sigmaX=35, sigmaY=35)
    blurred = blurred / max(float(blurred.max()), 1e-6)
    heat = np.zeros(raw_shape, dtype=np.uint8)
    heat[:, :, 1] = np.clip(180 * blurred, 0, 180).astype(np.uint8)
    heat[:, :, 2] = np.clip(255 * blurred, 0, 255).astype(np.uint8)
    return heat


def overlay_density(base: np.ndarray, density: np.ndarray, alpha: float = 0.28) -> np.ndarray:
    active = np.any(density > 0, axis=2)
    mixed = cv2.addWeighted(base, 1.0 - alpha, density, alpha, 0)
    out = base.copy()
    out[active] = mixed[active]
    return out


def thumbnail(image: np.ndarray, title: str) -> np.ndarray:
    width, height = 360, 250
    title_h = 34
    panel = np.full((height + title_h, width, 3), 255, dtype=np.uint8)
    panel[title_h:] = fit(image, (width, height), (245, 247, 249))
    put_text(panel, title, (12, 24), 0.55, (45, 51, 60), 1)
    cv2.rectangle(panel, (0, title_h), (width - 1, height + title_h - 1), (210, 216, 224), 1)
    return panel


def draw_legend(canvas: np.ndarray, x: int, y: int) -> None:
    items = [
        ("Low vigour", (35, 35, 235)),
        ("Medium vigour", (215, 90, 190)),
        ("High vigour", (80, 235, 120)),
        ("Concern density", (0, 180, 255)),
    ]
    put_text(canvas, "Legend", (x, y), 0.58, (35, 39, 46), 2)
    yy = y + 30
    for label, color in items:
        cv2.rectangle(canvas, (x, yy - 16), (x + 26, yy + 5), color, cv2.FILLED)
        cv2.rectangle(canvas, (x, yy - 16), (x + 26, yy + 5), (40, 40, 40), 1)
        put_text(canvas, label, (x + 38, yy), 0.48, (56, 64, 74), 1)
        yy += 28


def draw_status_badge(canvas: np.ndarray, status: str, x: int, y: int) -> None:
    color = STATUS_COLORS.get(status, (120, 120, 120))
    cv2.rectangle(canvas, (x, y), (x + 220, y + 48), color, cv2.FILLED)
    cv2.rectangle(canvas, (x, y), (x + 220, y + 48), (35, 39, 46), 1)
    put_text(canvas, status.upper(), (x + 18, y + 32), 0.72, (255, 255, 255), 2)


def build_daily_print(date: str, summary: dict[str, str], temporal: dict[str, str]) -> Path:
    date_dir = PRODUCT_DIR / date
    raw = load(date_dir / "raw_ndvi.png")
    human = load(date_dir / "human_annotation_outlines.png")
    seed = load(date_dir / "seed_polygons.png")
    preview = load(date_dir / "human_anchored_vigour_preview.png")
    uncertainty = load(date_dir / "concern_uncertainty_panel.png")
    temporal_overlay = load(date_dir / "temporal_change_overlay.png")
    if temporal_overlay.shape[2] == 4:
        temporal_bgr = temporal_overlay[:, :, :3]
    else:
        temporal_bgr = temporal_overlay
    density = concern_density(seed, raw.shape)

    main_map = overlay_density(human, density)

    page_w, page_h = 1600, 2200
    page = np.full((page_h, page_w, 3), 255, dtype=np.uint8)

    put_text(page, f"{FIELD_NAME} - Human-Anchored Vigour Review", (70, 70), 0.95, (25, 31, 38), 2)
    put_text(page, date, (70, 118), 0.72, (58, 66, 76), 2)
    put_text(page, growth_stage_for_date(date), (290, 118), 0.72, (58, 66, 76), 2)
    source_name = Path(summary["raw_path"]).name
    put_text(page, f"Source: {source_name}", (70, 160), 0.52, (91, 101, 113), 1)
    status = temporal.get("status", "Monitor")
    draw_status_badge(page, status, 1280, 58)

    map_x, map_y, map_w, map_h = 70, 205, 1460, 1370
    page[map_y : map_y + map_h, map_x : map_x + map_w] = fit(main_map, (map_w, map_h), (245, 247, 249))
    cv2.rectangle(page, (map_x, map_y), (map_x + map_w, map_y + map_h), (185, 192, 201), 2)
    draw_legend(page, 1115, 245)

    strip_y = 1618
    thumbs = [
        thumbnail(raw, "Raw NDVI"),
        thumbnail(human, "Human annotations"),
        thumbnail(overlay_density(raw, density, 0.45), "Concern density"),
        thumbnail(preview, "Vigour zone preview"),
    ]
    x = 70
    for thumb in thumbs:
        page[strip_y : strip_y + thumb.shape[0], x : x + thumb.shape[1]] = thumb
        x += thumb.shape[1] + 20

    note_y = 1950
    cv2.rectangle(page, (70, note_y - 34), (1530, note_y + 42), (244, 247, 250), cv2.FILLED)
    cv2.rectangle(page, (70, note_y - 34), (1530, note_y + 42), (212, 218, 226), 1)
    put_text(page, "Use marked concern areas as scouting priorities. Ground inspection required.", (96, note_y + 10), 0.66, (35, 39, 46), 2)

    timeline = load(OUTPUT_DIR / "temporal_timeline_chart.png") if (OUTPUT_DIR / "temporal_timeline_chart.png").exists() else temporal_bgr
    timeline_thumb = fit(timeline, (560, 150), (255, 255, 255))
    page[2018 : 2018 + timeline_thumb.shape[0], 970 : 970 + timeline_thumb.shape[1]] = timeline_thumb
    put_text(page, f"Status: {status}", (70, 2028), 0.58, STATUS_COLORS.get(status, (60, 60, 60)), 2)
    put_text(page, f"New {temporal.get('new_concern_percent_field', '0')}% | Persistent {temporal.get('persistent_concern_percent_field', '0')}% | Recovered {temporal.get('recovered_percent_field', '0')}%", (70, 2060), 0.47, (67, 76, 88), 1)
    status_note = temporal.get("status_note", "")
    if date == "2026-05-29":
        status_note = "Concern expanded sharply since previous flight."
    put_text(page, status_note[:72], (70, 2090), 0.47, (67, 76, 88), 1)

    put_text(page, "For scouting prioritization only. Not a disease diagnosis.", (70, 2168), 0.5, (91, 101, 113), 1)
    put_text(page, "Appendix caveat: temporal change is image-space review of human concern polygons, not surveyed area.", (70, 2196), 0.42, (124, 132, 143), 1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"{date}_print.png"
    cv2.imwrite(str(out), page)
    return out


def build_contact_sheet(paths: list[Path]) -> Path:
    cards = []
    for path in paths:
        image = load(path)
        card = fit(image, (500, 690), (255, 255, 255))
        cards.append(card)
    per_row = 3
    rows = []
    for i in range(0, len(cards), per_row):
        chunk = cards[i : i + per_row]
        while len(chunk) < per_row:
            chunk.append(np.full_like(cards[0], 255))
        rows.append(np.hstack(chunk))
    sheet = np.vstack(rows)
    out = OUTPUT_DIR / "contact_sheet_all_days.png"
    cv2.imwrite(str(out), sheet)
    return out


def main() -> None:
    summary = read_summary()
    temporal = read_temporal()
    outputs = []
    audit_rows = []
    for date in DATES:
        if date not in summary:
            audit_rows.append({"date": date, "output": "", "status": "SKIPPED", "reason": "No matched product summary row"})
            continue
        out = build_daily_print(date, summary[date], temporal.get(date, {}))
        outputs.append(out)
        audit_rows.append({"date": date, "output": str(out), "status": "PASS", "reason": ""})
        print(f"[WRITE] {out}")
    contact = build_contact_sheet(outputs)
    print(f"[WRITE] {contact}")

    audit_path = OUTPUT_DIR / "product_audit_summary.csv"
    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "output", "status", "reason"])
        writer.writeheader()
        writer.writerows(audit_rows)
    print(f"[WRITE] {audit_path}")


if __name__ == "__main__":
    main()
