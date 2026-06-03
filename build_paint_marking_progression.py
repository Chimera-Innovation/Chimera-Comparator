from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DATASET_ROOT = Path(r"D:\NDVI-dataset")
RAW_DIR = DATASET_ROOT / "raw-ndvi"
ANNOTATED_DIR = DATASET_ROOT / "human-Annotated-ndvi"
OUT_DIR = DATASET_ROOT / "MVP - ndvi progression" / "paint_marking_progression"

MAX_PROCESS_DIM = 2200
MAX_PANEL_DIM = 650
RED_MIN = 145
RED_DELTA = 42
SATURATION_MIN = 55
DIFF_MIN = 35
MIN_COMPONENT_AREA = 28

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


def red_paint_candidates(annotated_arr: np.ndarray) -> np.ndarray:
    arr = annotated_arr.astype(np.int16)
    red = arr[:, :, 0]
    green = arr[:, :, 1]
    blue = arr[:, :, 2]
    max_ch = arr.max(axis=2)
    min_ch = arr.min(axis=2)
    saturation = max_ch - min_ch
    return (
        (red >= RED_MIN)
        & (red >= green + RED_DELTA)
        & (red >= blue + RED_DELTA)
        & (saturation >= SATURATION_MIN)
    )


def clean_mask(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    mask = remove_isolated(mask)
    mask = dilate(mask)
    mask = erode(mask)
    return mask


def remove_isolated(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
    neighbors = (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    )
    return mask & (neighbors >= 2)


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


def component_filter(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    output = np.zeros_like(mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    active = set(zip(xs.tolist(), ys.tolist()))
    while active:
        start = active.pop()
        stack = [start]
        visited[start[1], start[0]] = True
        pts: list[tuple[int, int]] = []
        min_x = max_x = start[0]
        min_y = max_y = start[1]
        while stack:
            x, y = stack.pop()
            pts.append((x, y))
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1), (x - 1, y - 1), (x + 1, y + 1), (x - 1, y + 1), (x + 1, y - 1)):
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                if visited[ny, nx] or not mask[ny, nx]:
                    continue
                visited[ny, nx] = True
                active.discard((nx, ny))
                stack.append((nx, ny))
        comp_w = max_x - min_x + 1
        comp_h = max_y - min_y + 1
        touches_header = min_y < 42 and comp_w > width * 0.4
        text_like = len(pts) < MIN_COMPONENT_AREA and comp_w < 120 and comp_h < 45
        if len(pts) >= MIN_COMPONENT_AREA and not touches_header and not text_like:
            for x, y in pts:
                output[y, x] = True
    return output


def extract_mask(pair: ImagePair) -> tuple[np.ndarray, dict[str, Any]]:
    raw_small, annotated_small, sx, sy = resized_pair(pair)
    raw_arr = np.asarray(raw_small).astype(np.int16)
    ann_arr = np.asarray(annotated_small).astype(np.int16)
    diff = np.abs(ann_arr - raw_arr).max(axis=2)
    red_candidates = red_paint_candidates(ann_arr)
    changed_red = red_candidates & (diff >= DIFF_MIN)
    cleaned = component_filter(clean_mask(changed_red))
    visible_candidate_pixels = int(red_candidates.sum())
    extracted_pixels = int(cleaned.sum())
    settings = {
        "red_min": RED_MIN,
        "red_delta": RED_DELTA,
        "saturation_min": SATURATION_MIN,
        "diff_min": DIFF_MIN,
        "visible_red_candidate_pixels_processing_space": visible_candidate_pixels,
        "extracted_red_pixels_processing_space": extracted_pixels,
        "processing_size": raw_small.size,
        "scale_x": sx,
        "scale_y": sy,
        "raw_ndvi_thresholding_used": False,
    }
    return cleaned, settings


def save_mask(mask: np.ndarray, pair: ImagePair, path: Path) -> int:
    mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    mask_image = mask_image.resize(pair.raw_size, Image.Resampling.NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    mask_image.save(path)
    return int(np.asarray(mask_image).astype(bool).sum())


def make_overlay(pair: ImagePair, mask_path: Path, output: Path) -> None:
    raw = open_rgb(pair.raw_path)
    mask = Image.open(mask_path).convert("L")
    overlay = Image.new("RGB", raw.size, (255, 0, 0))
    alpha = mask.point(lambda value: 230 if value else 0)
    raw.paste(overlay, (0, 0), alpha)
    raw.thumbnail((MAX_PANEL_DIM, MAX_PANEL_DIM), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(raw)
    draw.rectangle((0, 0, raw.size[0], 40), fill=(0, 0, 0))
    draw.text((10, 10), f"{pair.field_name} {pair.date}", fill=(255, 255, 255), font=SMALL)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw.save(output)


def create_debug(pair: ImagePair, mask_path: Path, overlay_path: Path, output: Path) -> None:
    panels = [
        ("raw NDVI", open_rgb(pair.raw_path, 460)),
        ("Paint-marked annotated", open_rgb(pair.annotated_path, 460)),
        ("extracted Paint mask", Image.open(mask_path).convert("RGB")),
        ("final overlay", open_rgb(overlay_path, 460)),
    ]
    panels[2] = (panels[2][0], panels[2][1].resize(panels[0][1].size, Image.Resampling.NEAREST))
    panel_w = max(panel.size[0] for _, panel in panels)
    panel_h = max(panel.size[1] for _, panel in panels)
    canvas = Image.new("RGB", (panel_w * 4, panel_h + 70), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (title, panel) in enumerate(panels):
        x = idx * panel_w
        draw.text((x + 10, 12), title, fill=(20, 20, 20), font=SMALL)
        canvas.paste(panel, (x, 58))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def make_timeline(field: str, rows: list[dict[str, str]], output: Path) -> None:
    panels = []
    for row in rows:
        img = open_rgb(Path(row["overlay_path"]), 520)
        panel = Image.new("RGB", (img.size[0], img.size[1] + 62), "white")
        draw = ImageDraw.Draw(panel)
        status = row["extraction_status"]
        label = f"{row['date']} | {status}"
        if row["paint_marking_area_percent"]:
            label += f" | {float(row['paint_marking_area_percent']):.3f}%"
        draw.text((8, 8), label, fill=(120, 20, 20), font=SMALL)
        panel.paste(img, (0, 62))
        panels.append(panel)
    width = sum(panel.size[0] for panel in panels)
    height = max(panel.size[1] for panel in panels) + 70
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 18), f"{field} Paint Marking Timeline", fill=(20, 20, 20), font=TITLE)
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 70))
        x += panel.size[0]
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def write_report(rows: list[dict[str, str]]) -> None:
    succeeded = [row for row in rows if row["extraction_status"] == "success"]
    failed = [row for row in rows if row["extraction_status"] == "failed_manual_review_required"]
    no_visible = [row for row in rows if row["extraction_status"] == "no_red_paint_marking_detected"]
    lines = [
        "# Paint Marking Final Report",
        "",
        "The Paint markings are treated as the truth. Raw NDVI is only the background.",
        "",
        f"- Dates succeeded: {len(succeeded)}",
        f"- Dates failed: {len(failed)}",
        f"- Dates with no extracted red Paint marking: {len(no_visible)}",
        f"- Total markings extracted: {sum(int(row['paint_marking_area_pixels'] or 0) for row in succeeded)} pixels",
        "- Output follows Paint markings only: yes",
        "",
        "## Succeeded",
        "",
    ]
    lines.extend(f"- {row['field_name']} {row['date']}" for row in succeeded)
    lines.extend(["", "## Manual Review Required", ""])
    lines.extend([f"- {row['field_name']} {row['date']}" for row in failed] or ["- none"])
    lines.extend(["", "## No Red Paint Marking Detected", ""])
    lines.extend([f"- {row['field_name']} {row['date']}" for row in no_visible] or ["- none"])
    lines.extend(
        [
            "",
            "Final decision: Corrected MVP now follows red Paint markings only. Raw NDVI is background context, not the source of detected concern polygons.",
        ]
    )
    (OUT_DIR / "paint_marking_final_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = match_pairs()
    rows: list[dict[str, str]] = []
    rows_by_field: dict[str, list[dict[str, str]]] = {}
    for pair in pairs:
        mask, settings = extract_mask(pair)
        mask_path = OUT_DIR / pair.field_name / "masks" / f"{pair.field_name}_{pair.date}_paint_marking_mask.png"
        overlay_path = OUT_DIR / pair.field_name / "overlays" / f"{pair.field_name}_{pair.date}_paint_marking_overlay.png"
        debug_path = OUT_DIR / pair.field_name / "debug" / f"{pair.field_name}_{pair.date}_debug.png"
        area_pixels = save_mask(mask, pair, mask_path)
        make_overlay(pair, mask_path, overlay_path)
        create_debug(pair, mask_path, overlay_path, debug_path)
        visible = settings["visible_red_candidate_pixels_processing_space"] > 200
        extracted = area_pixels > 0
        if extracted:
            status = "success"
            area_percent = f"{area_pixels / (pair.raw_size[0] * pair.raw_size[1]) * 100:.6f}"
            notes = "Extracted red Paint pixels changed from raw image. Raw NDVI thresholding not used."
        elif visible:
            status = "failed_manual_review_required"
            area_percent = ""
            notes = "Red Paint-like markings were visible but extraction produced no mask; manual review required."
        else:
            status = "no_red_paint_marking_detected"
            area_percent = ""
            notes = "No changed red Paint markings detected. Raw NDVI was not used as a detection source."
        row = {
            "field_name": pair.field_name,
            "date": pair.date,
            "raw_ndvi_path": str(pair.raw_path),
            "annotated_image_path": str(pair.annotated_path),
            "extraction_status": status,
            "paint_marking_visible": "yes" if visible else "no",
            "paint_marking_area_pixels": str(area_pixels) if extracted else "",
            "paint_marking_area_percent": area_percent,
            "notes": notes + " " + json.dumps(settings),
            "overlay_path": str(overlay_path),
        }
        rows.append(row)
        rows_by_field.setdefault(pair.field_name, []).append(row)

    summary_path = OUT_DIR / "paint_marking_summary.csv"
    public_columns = [
        "field_name",
        "date",
        "raw_ndvi_path",
        "annotated_image_path",
        "extraction_status",
        "paint_marking_visible",
        "paint_marking_area_pixels",
        "paint_marking_area_percent",
        "notes",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=public_columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in public_columns})

    for field, field_rows in rows_by_field.items():
        field_rows.sort(key=lambda row: row["date"])
        make_timeline(field, field_rows, OUT_DIR / field / f"{field}_paint_marking_timeline.png")

    write_report(rows)
    print(f"- output folder: {OUT_DIR}")
    print(f"- fields processed: {', '.join(sorted(rows_by_field))}")
    print(f"- dates processed: {', '.join(row['date'] for row in rows)}")
    print(f"- successful extractions: {sum(1 for row in rows if row['extraction_status'] == 'success')}")
    print(f"- manual review required: {sum(1 for row in rows if row['extraction_status'] == 'failed_manual_review_required')}")
    print(f"- summary CSV: {summary_path}")
    print(f"- final report: {OUT_DIR / 'paint_marking_final_report.md'}")
    print("- final decision: The Paint markings are the truth. Raw NDVI is only the background.")


if __name__ == "__main__":
    main()
