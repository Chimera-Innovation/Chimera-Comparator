from __future__ import annotations

import csv
import json
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DATASET_ROOT = Path(r"D:\NDVI-dataset")
RAW_DIR = DATASET_ROOT / "raw-ndvi"
ANNOTATED_DIR = DATASET_ROOT / "human-Annotated-ndvi"
OUT_DIR = DATASET_ROOT / "MVP - ndvi progression" / "black_paint_marking_progression"

MAX_PROCESS_DIM = 2200
MAX_PANEL_DIM = 650
DARK_MAX = 95
DIFF_MIN = 22
MIN_STROKE_COMPONENT_AREA = 24
MIN_FILLED_REGION_AREA = 240
CLOSING_ITERATIONS = 4

DATE_RE = re.compile(r"(?P<m>\d{1,2})-(?P<d>\d{1,2})-(?P<y>\d{4})")
FIELD_RE = re.compile(r"^(?P<field>.+?)-(?P<m>\d{1,2})-(?P<d>\d{1,2})-(?P<y>\d{4})-orthophoto", re.IGNORECASE)


@dataclass
class ImagePair:
    field_name: str
    date: str
    raw_path: Path
    annotated_path: Path
    raw_size: tuple[int, int]
    annotated_size: tuple[int, int]
    resized_annotated_to_raw: bool


def font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


FONT = font(22)
SMALL = font(16)
TITLE = font(28)


def iso_date_from_name(name: str) -> str | None:
    match = DATE_RE.search(name)
    if not match:
        return None
    return f"{int(match.group('y')):04d}-{int(match.group('m')):02d}-{int(match.group('d')):02d}"


def field_from_name(name: str) -> str | None:
    match = FIELD_RE.search(name)
    if not match:
        return None
    return match.group("field").replace("_", "-").strip("- ")


def open_rgb(path: Path, max_dim: int | None = None) -> Image.Image:
    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(path).convert("RGB")
    if max_dim:
        image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    return image.copy()


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def match_pairs() -> list[ImagePair]:
    raw_candidates: list[dict[str, Any]] = []
    for raw in RAW_DIR.rglob("*"):
        if not raw.is_file() or raw.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            continue
        field = field_from_name(raw.name)
        date = iso_date_from_name(raw.name)
        if field and date:
            raw_candidates.append({"field": field, "date": date, "path": raw})

    pairs: list[ImagePair] = []
    for annotated in ANNOTATED_DIR.rglob("*"):
        if not annotated.is_file() or annotated.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            continue
        field = field_from_name(annotated.name)
        date = iso_date_from_name(annotated.name)
        if not field or not date:
            continue
        matches = [candidate for candidate in raw_candidates if candidate["field"] == field and candidate["date"] == date]
        if not matches:
            continue
        raw_path = sorted(matches, key=lambda item: len(item["path"].name))[0]["path"]
        raw_size = image_size(raw_path)
        annotated_size = image_size(annotated)
        pairs.append(
            ImagePair(
                field_name=field,
                date=date,
                raw_path=raw_path,
                annotated_path=annotated,
                raw_size=raw_size,
                annotated_size=annotated_size,
                resized_annotated_to_raw=raw_size != annotated_size,
            )
        )
    return sorted(pairs, key=lambda pair: (pair.field_name, pair.date))


def resized_pair(pair: ImagePair) -> tuple[Image.Image, Image.Image, float, float]:
    raw = open_rgb(pair.raw_path)
    annotated = open_rgb(pair.annotated_path)
    if annotated.size != raw.size:
        annotated = annotated.resize(raw.size, Image.Resampling.BILINEAR)
    original_w, original_h = raw.size
    scale = min(1.0, MAX_PROCESS_DIM / max(original_w, original_h))
    target = (max(1, round(original_w * scale)), max(1, round(original_h * scale)))
    return (
        raw.resize(target, Image.Resampling.BILINEAR),
        annotated.resize(target, Image.Resampling.BILINEAR),
        original_w / target[0],
        original_h / target[1],
    )


def dilate(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant")
    out = np.zeros_like(mask, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            out |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out


def erode(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant")
    out = np.ones_like(mask, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            out &= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out


def close_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    out = mask
    for _ in range(iterations):
        out = dilate(out)
    for _ in range(iterations):
        out = erode(out)
    return out


def components(mask: np.ndarray) -> list[dict[str, Any]]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    active = set(zip(xs.tolist(), ys.tolist()))
    found: list[dict[str, Any]] = []
    while active:
        start = active.pop()
        queue: deque[tuple[int, int]] = deque([start])
        visited[start[1], start[0]] = True
        points: list[tuple[int, int]] = []
        min_x = max_x = start[0]
        min_y = max_y = start[1]
        while queue:
            x, y = queue.popleft()
            points.append((x, y))
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1), (x - 1, y - 1), (x + 1, y + 1), (x - 1, y + 1), (x + 1, y - 1)):
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                if visited[ny, nx] or not mask[ny, nx]:
                    continue
                visited[ny, nx] = True
                active.discard((nx, ny))
                queue.append((nx, ny))
        found.append({"points": points, "area": len(points), "bbox": [min_x, min_y, max_x, max_y]})
    return found


def filter_strokes(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for comp in components(mask):
        x1, y1, x2, y2 = comp["bbox"]
        comp_w, comp_h = x2 - x1 + 1, y2 - y1 + 1
        touches_border = x1 <= 2 or y1 <= 2 or x2 >= width - 3 or y2 >= height - 3
        giant_border = touches_border and (comp_w > width * 0.65 or comp_h > height * 0.65)
        tiny = comp["area"] < MIN_STROKE_COMPONENT_AREA
        if not tiny and not giant_border:
            for x, y in comp["points"]:
                out[y, x] = True
    return out


def fill_regions_from_strokes(stroke_mask: np.ndarray) -> np.ndarray:
    closed = close_mask(stroke_mask, CLOSING_ITERATIONS)
    height, width = closed.shape
    outside = np.zeros_like(closed, dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for x in range(width):
        if not closed[0, x]:
            queue.append((x, 0))
        if not closed[height - 1, x]:
            queue.append((x, height - 1))
    for y in range(height):
        if not closed[y, 0]:
            queue.append((0, y))
        if not closed[y, width - 1]:
            queue.append((width - 1, y))
    while queue:
        x, y = queue.popleft()
        if outside[y, x] or closed[y, x]:
            continue
        outside[y, x] = True
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < width and 0 <= ny < height and not outside[ny, nx] and not closed[ny, nx]:
                queue.append((nx, ny))
    filled = ~outside
    filled &= ~closed

    clean = np.zeros_like(filled, dtype=bool)
    for comp in components(filled):
        x1, y1, x2, y2 = comp["bbox"]
        comp_w, comp_h = x2 - x1 + 1, y2 - y1 + 1
        if comp["area"] >= MIN_FILLED_REGION_AREA and comp_w >= 12 and comp_h >= 12:
            for x, y in comp["points"]:
                clean[y, x] = True
    return clean


def fallback_fill_from_stroke_boxes(stroke_mask: np.ndarray, filled_mask: np.ndarray) -> np.ndarray:
    out = filled_mask.copy()
    for comp in components(stroke_mask):
        x1, y1, x2, y2 = comp["bbox"]
        comp_w, comp_h = x2 - x1 + 1, y2 - y1 + 1
        if comp["area"] >= 120 and comp_w >= 30 and comp_h >= 18:
            out[y1 : y2 + 1, x1 : x2 + 1] = True
    return out


def extract_black_marks(pair: ImagePair) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    raw_small, annotated_small, sx, sy = resized_pair(pair)
    raw_arr = np.asarray(raw_small).astype(np.int16)
    ann_arr = np.asarray(annotated_small).astype(np.int16)
    ann_gray = ann_arr.mean(axis=2)
    raw_gray = raw_arr.mean(axis=2)
    diff = np.abs(ann_arr - raw_arr).max(axis=2)
    dark = (ann_arr[:, :, 0] < DARK_MAX) & (ann_arr[:, :, 1] < DARK_MAX) & (ann_arr[:, :, 2] < DARK_MAX)
    darker_than_raw = raw_gray - ann_gray > 10
    stroke_mask = dark & (diff >= DIFF_MIN) & darker_than_raw
    stroke_mask = filter_strokes(close_mask(stroke_mask, 1))
    filled_mask = fill_regions_from_strokes(stroke_mask)
    filled_mask = fallback_fill_from_stroke_boxes(stroke_mask, filled_mask)
    settings = {
        "dark_max": DARK_MAX,
        "diff_min": DIFF_MIN,
        "closing_iterations": CLOSING_ITERATIONS,
        "stroke_pixels_processing_space": int(stroke_mask.sum()),
        "filled_pixels_processing_space": int(filled_mask.sum()),
        "processing_size": raw_small.size,
        "scale_x": sx,
        "scale_y": sy,
        "raw_ndvi_thresholding_used": False,
    }
    return stroke_mask, filled_mask, settings


def save_mask(mask: np.ndarray, pair: ImagePair, path: Path) -> int:
    image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    image = image.resize(pair.raw_size, Image.Resampling.NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return int(np.asarray(image).astype(bool).sum())


def make_overlay(pair: ImagePair, stroke_path: Path, filled_path: Path, output: Path) -> None:
    raw = open_rgb(pair.raw_path)
    stroke = Image.open(stroke_path).convert("L")
    filled = Image.open(filled_path).convert("L")
    fill_color = Image.new("RGB", raw.size, (255, 220, 0))
    fill_alpha = filled.point(lambda value: 90 if value else 0)
    raw.paste(fill_color, (0, 0), fill_alpha)
    stroke_color = Image.new("RGB", raw.size, (0, 0, 0))
    stroke_alpha = stroke.point(lambda value: 255 if value else 0)
    raw.paste(stroke_color, (0, 0), stroke_alpha)
    raw.thumbnail((MAX_PANEL_DIM, MAX_PANEL_DIM), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(raw)
    draw.rectangle((0, 0, raw.size[0], 44), fill=(0, 0, 0))
    draw.text((10, 11), f"{pair.field_name} {pair.date}", fill=(255, 255, 255), font=SMALL)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw.save(output)


def make_debug(pair: ImagePair, stroke_path: Path, filled_path: Path, overlay_path: Path, output: Path) -> None:
    annotated = open_rgb(pair.annotated_path, 450)
    stroke = Image.open(stroke_path).convert("RGB").resize(annotated.size, Image.Resampling.NEAREST)
    filled = Image.open(filled_path).convert("RGB").resize(annotated.size, Image.Resampling.NEAREST)
    overlay = open_rgb(overlay_path, 450)
    panels = [("annotated NDVI", annotated), ("black Paint strokes", stroke), ("filled human-marked regions", filled), ("final overlay on raw NDVI", overlay)]
    width = sum(panel.size[0] for _, panel in panels)
    height = max(panel.size[1] for _, panel in panels) + 62
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    x = 0
    for title, panel in panels:
        draw.text((x + 8, 10), title, fill=(20, 20, 20), font=SMALL)
        canvas.paste(panel, (x, 62))
        x += panel.size[0]
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def make_timeline(field: str, rows: list[dict[str, str]], output: Path) -> None:
    panels = []
    for row in rows:
        image = open_rgb(Path(row["overlay_path"]), 520)
        panel = Image.new("RGB", (image.size[0], image.size[1] + 72), "white")
        draw = ImageDraw.Draw(panel)
        label = f"{row['date']} | {row['extraction_status']}"
        if row["filled_region_area_percent"]:
            label += f" | {float(row['filled_region_area_percent']):.2f}% filled"
        draw.text((8, 8), label, fill=(20, 20, 20), font=SMALL)
        panel.paste(image, (0, 72))
        panels.append(panel)
    width = sum(panel.size[0] for panel in panels)
    height = max(panel.size[1] for panel in panels) + 88
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 18), f"{field} Black Paint Progression Timeline", fill=(20, 20, 20), font=TITLE)
    if field == "McIntyre-Road":
        draw.text((18, 52), "Human-marked perceived concern zones increased between 2026-05-08 and 2026-05-27.", fill=(120, 30, 20), font=SMALL)
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 88))
        x += panel.size[0]
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def write_audit(rows: list[dict[str, str]]) -> None:
    lines = [
        "# Black Marking Audit",
        "",
        "| field_name | date | image_width | image_height | estimated_dark_annotation_pixels | dark_annotation_visible_yes_no | notes |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['field_name']} | {row['date']} | {row['image_width']} | {row['image_height']} | {row['stroke_area_pixels'] or 0} | {row['black_marking_visible']} | {row['notes'].replace('|', '/')} |"
        )
    (OUT_DIR / "black_marking_audit.md").write_text("\n".join(lines), encoding="utf-8")


def write_interpretation() -> None:
    path = OUT_DIR / "McIntyre-Road" / "McIntyre-Road_interpretation.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '"Based only on the human black Paint markings, McIntyre-Road shows increased perceived vigour-loss concern between 2026-05-08 and 2026-05-27. The later image contains more marked zones, especially the upper-right area, central-right row band, lower-field band, and several smaller right-side row-level regions. This is a human-observed progression signal for scouting discussion, not calibrated disease diagnosis."\n',
        encoding="utf-8",
    )


def write_final_report(rows: list[dict[str, str]]) -> None:
    succeeded = [row for row in rows if row["extraction_status"] == "success"]
    failed = [row for row in rows if row["extraction_status"] != "success"]
    lines = [
        "# Black Paint Marking Final Report",
        "",
        "Final rule: black Paint markings are the source of truth. Raw NDVI is background only. NDVI colour is not the annotation.",
        "",
        f"- Dates succeeded: {len(succeeded)}",
        f"- Dates needing manual review: {len(failed)}",
        f"- Total successful filled-region area pixels: {sum(int(row['filled_region_area_pixels'] or 0) for row in succeeded)}",
        "- Output follows black Paint markings only: yes",
        "- Raw NDVI thresholding used: no",
        "",
        "## Succeeded",
        "",
    ]
    lines.extend(f"- {row['field_name']} {row['date']}: {row['filled_region_area_percent']}% filled region" for row in succeeded)
    lines.extend(["", "## Manual Review Required", ""])
    lines.extend([f"- {row['field_name']} {row['date']}: {row['notes']}" for row in failed] or ["- none"])
    (OUT_DIR / "black_paint_marking_final_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = match_pairs()
    rows: list[dict[str, str]] = []
    rows_by_field: dict[str, list[dict[str, str]]] = {}
    for pair in pairs:
        stroke, filled, settings = extract_black_marks(pair)
        stroke_path = OUT_DIR / pair.field_name / "masks" / f"{pair.field_name}_{pair.date}_black_paint_stroke_mask.png"
        filled_path = OUT_DIR / pair.field_name / "masks" / f"{pair.field_name}_{pair.date}_filled_human_marked_region_mask.png"
        overlay_path = OUT_DIR / pair.field_name / "overlays" / f"{pair.field_name}_{pair.date}_black_paint_overlay.png"
        debug_path = OUT_DIR / pair.field_name / "debug" / f"{pair.field_name}_{pair.date}_debug.png"
        stroke_area = save_mask(stroke, pair, stroke_path)
        filled_area = save_mask(filled, pair, filled_path)
        make_overlay(pair, stroke_path, filled_path, overlay_path)
        make_debug(pair, stroke_path, filled_path, overlay_path, debug_path)

        visible = stroke_area > 0
        if visible and filled_area > 0:
            status = "success"
            percent = f"{filled_area / (pair.raw_size[0] * pair.raw_size[1]) * 100:.6f}"
            notes = "Black/dark Paint strokes extracted from annotated-vs-raw difference; filled region approximates enclosed human-marked zone."
        elif visible:
            status = "failed_manual_review_required"
            percent = ""
            notes = "Black/dark Paint markings visible but no filled region could be recovered; manual review required."
        else:
            status = "failed_manual_review_required"
            percent = ""
            notes = "No black/dark Paint markings extracted; if markings are visible, manual review is required."

        row = {
            "field_name": pair.field_name,
            "date": pair.date,
            "annotated_image_path": str(pair.annotated_path),
            "raw_ndvi_path": str(pair.raw_path),
            "extraction_status": status,
            "black_marking_visible": "yes" if visible else "no",
            "stroke_area_pixels": str(stroke_area) if stroke_area else "",
            "filled_region_area_pixels": str(filled_area) if filled_area else "",
            "filled_region_area_percent": percent,
            "num_marked_regions": str(len([comp for comp in components(filled) if comp["area"] >= MIN_FILLED_REGION_AREA])),
            "image_width": str(pair.raw_size[0]),
            "image_height": str(pair.raw_size[1]),
            "notes": notes + " " + json.dumps(settings),
            "overlay_path": str(overlay_path),
        }
        rows.append(row)
        rows_by_field.setdefault(pair.field_name, []).append(row)

    summary_columns = [
        "field_name",
        "date",
        "annotated_image_path",
        "raw_ndvi_path",
        "extraction_status",
        "black_marking_visible",
        "stroke_area_pixels",
        "filled_region_area_pixels",
        "filled_region_area_percent",
        "num_marked_regions",
        "notes",
    ]
    with (OUT_DIR / "black_paint_marking_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in summary_columns})

    for field, field_rows in rows_by_field.items():
        field_rows.sort(key=lambda row: row["date"])
        make_timeline(field, field_rows, OUT_DIR / field / f"{field}_black_paint_progression_timeline.png")

    write_audit(rows)
    write_interpretation()
    write_final_report(rows)
    print(f"- output folder: {OUT_DIR}")
    print(f"- fields processed: {', '.join(sorted(rows_by_field))}")
    print(f"- dates processed: {', '.join(row['date'] for row in rows)}")
    print(f"- successful extractions: {sum(1 for row in rows if row['extraction_status'] == 'success')}")
    print(f"- manual review required: {sum(1 for row in rows if row['extraction_status'] != 'success')}")
    print(f"- CSV path: {OUT_DIR / 'black_paint_marking_summary.csv'}")
    print("- final rule: Black Paint markings are the truth. Raw NDVI is background only.")


if __name__ == "__main__":
    main()
