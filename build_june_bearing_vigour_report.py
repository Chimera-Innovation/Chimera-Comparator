from __future__ import annotations

import csv
import math
import re
import textwrap
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


Image.MAX_IMAGE_PIXELS = None

ROOT = Path(r"D:\NDVI-dataset\MVP - ndvi progression")
RAW_DIR = Path(r"D:\NDVI-dataset\raw-ndvi")
ANNOTATION_DIR = Path(r"C:\Users\Chimera\Downloads\Strawberry 1 - annotated")
OUT_DIR = ROOT / "june_bearing_same_field_report"
ANALYSIS_MAX_DIM = 1800
REPORT_FIELD_NAME = "Strawberry-1"

SELECTED_ANNOTATION_FILES = [
    "Mallard-Avenue-5-12-2026-orthophoto-NDVI-annotated.png",
    "Mallard-Avenue-5-18-2026-orthophoto-NDVI-annotated.png",
    "Mallard-Avenue-5-22-2026-orthophoto-NDVI-annotated.png",
    "Mallard-Avenue-5-29-2026-orthophoto-NDVI---annotated-output.png",
    "McIntyre-Road-5-27-2026-orthophoto-NDVI-annotated.png",
]

CLASS_INFO = {
    "low": {
        "label": "Low vigour",
        "target": np.array([230, 20, 34], dtype=np.int16),
        "display": (226, 36, 42),
        "priority": 3,
    },
    "medium": {
        "label": "Medium vigour",
        "target": np.array([196, 180, 226], dtype=np.int16),
        "display": (154, 91, 212),
        "priority": 2,
    },
    "high": {
        "label": "High vigour",
        "target": np.array([178, 224, 20], dtype=np.int16),
        "display": (135, 214, 47),
        "priority": 1,
    },
}

STAGE_BY_DATE = {
    "2026-05-08": ("Bloom / early fruit set", "Early season: vigour marks can highlight establishment, stand gaps, or early resource-limitation areas before the canopy fully expresses."),
    "2026-05-12": ("Bloom to fruit set", "June-bearing strawberries are moving through flowering and early fruit set; early low-vigour marks are scouting flags, not diagnoses."),
    "2026-05-18": ("Green fruit development", "Fruit are sizing and canopy demand is rising; persistent low or medium vigour areas become more operationally important."),
    "2026-05-22": ("Fruit sizing / early ripening", "The crop is approaching ripening; expansion or persistence can indicate areas to inspect before harvest pressure peaks."),
    "2026-05-27": ("Ripening / early harvest window", "Harvest-time scouting should focus on repeated low-vigour and medium-vigour zones while separating field-edge effects from production rows."),
    "2026-05-29": ("Harvest-time ripening", "Late observation: mapped classes support harvest review and next-season scouting priorities."),
}


@dataclass
class Observation:
    field_name: str
    source_field_name: str
    date: str
    raw_path: Path
    annotation_path: Path
    analysis_size: tuple[int, int]
    stage: str
    stage_note: str
    stroke_masks: dict[str, np.ndarray]
    filled_masks: dict[str, np.ndarray]
    exclusive_masks: dict[str, np.ndarray]
    field_mask: np.ndarray
    percents: dict[str, float | None]
    stroke_pixel_counts: dict[str, int]
    filled_mask_paths: dict[str, Path]
    overlay_path: Path
    intelligent_overlay_path: Path
    debug_path: Path
    index_metrics: dict[str, float | None]
    status: str
    notes: str


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ("arial.ttf", "DejaVuSans.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


FONT_SMALL = font(16)
FONT_BODY = font(22)
FONT_BODY_BOLD = font(22, True)
FONT_H2 = font(30, True)
FONT_H1 = font(44, True)


def parse_filename(path: Path) -> tuple[str, str]:
    match = re.match(r"(?P<field>.+)-(?P<month>\d{1,2})-(?P<day>\d{1,2})-(?P<year>\d{4})-orthophoto-NDVI", path.name)
    if not match:
        raise ValueError(f"Cannot parse field/date from {path}")
    date = f"{int(match.group('year')):04d}-{int(match.group('month')):02d}-{int(match.group('day')):02d}"
    return match.group("field"), date


def date_label(date: str) -> str:
    parsed = datetime.strptime(date, "%Y-%m-%d")
    return f"May {parsed.day}" if parsed.month == 5 else parsed.strftime("%b %d")


def match_raw(field: str, date: str) -> Path:
    parsed = datetime.strptime(date, "%Y-%m-%d")
    needle = f"{field}-{parsed.month}-{parsed.day}-{parsed.year}-orthophoto-NDVI"
    matches = [path for path in RAW_DIR.rglob("*") if path.is_file() and needle in path.name]
    if not matches:
        raise FileNotFoundError(f"No raw NDVI match for {field} {date}")
    return matches[0]


def analysis_size(size: tuple[int, int]) -> tuple[int, int]:
    scale = min(1.0, ANALYSIS_MAX_DIM / max(size))
    return (max(1, round(size[0] * scale)), max(1, round(size[1] * scale)))


def open_rgb(path: Path, size: tuple[int, int] | None = None) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if size and image.size != size:
        image = image.resize(size, Image.Resampling.BILINEAR)
    return image.copy()


def dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = mask.astype(bool)
    for _ in range(iterations):
        padded = np.pad(out, 1, mode="constant")
        next_mask = np.zeros_like(out, dtype=bool)
        for dy in range(3):
            for dx in range(3):
                next_mask |= padded[dy : dy + out.shape[0], dx : dx + out.shape[1]]
        out = next_mask
    return out


def erode(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = mask.astype(bool)
    for _ in range(iterations):
        padded = np.pad(out, 1, mode="constant")
        next_mask = np.ones_like(out, dtype=bool)
        for dy in range(3):
            for dx in range(3):
                next_mask &= padded[dy : dy + out.shape[0], dx : dx + out.shape[1]]
        out = next_mask
    return out


def close_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    return erode(dilate(mask, iterations), iterations)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    background = ~mask
    visited = np.zeros_like(mask, dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for x in range(width):
        if background[0, x]:
            queue.append((x, 0))
        if background[height - 1, x]:
            queue.append((x, height - 1))
    for y in range(height):
        if background[y, 0]:
            queue.append((0, y))
        if background[y, width - 1]:
            queue.append((width - 1, y))
    while queue:
        x, y = queue.popleft()
        if x < 0 or y < 0 or x >= width or y >= height:
            continue
        if visited[y, x] or not background[y, x]:
            continue
        visited[y, x] = True
        queue.extend(((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)))
    return mask | (background & ~visited)


def components(mask: np.ndarray, min_area: int = 48) -> list[dict]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    found: list[dict] = []
    ys, xs = np.nonzero(mask)
    for start_x, start_y in zip(xs.tolist(), ys.tolist()):
        if visited[start_y, start_x]:
            continue
        queue: deque[tuple[int, int]] = deque([(start_x, start_y)])
        visited[start_y, start_x] = True
        points: list[tuple[int, int]] = []
        min_x = max_x = start_x
        min_y = max_y = start_y
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
                queue.append((nx, ny))
        if len(points) >= min_area:
            found.append({"points": points, "area": len(points), "bbox": (min_x, min_y, max_x, max_y)})
    found.sort(key=lambda item: item["area"], reverse=True)
    return found


def keep_meaningful_components(mask: np.ndarray) -> np.ndarray:
    min_area = max(32, int(mask.size * 0.00001))
    out = np.zeros_like(mask, dtype=bool)
    for comp in components(mask, min_area=min_area):
        x1, y1, x2, y2 = comp["bbox"]
        width = x2 - x1 + 1
        height = y2 - y1 + 1
        if width <= 2 and height <= 2:
            continue
        for x, y in comp["points"]:
            out[y, x] = True
    return out


def recover_filled_region(stroke_mask: np.ndarray) -> np.ndarray:
    if not stroke_mask.any():
        return stroke_mask
    iterations = max(3, min(9, round(max(stroke_mask.shape) / 260)))
    cleaned = keep_meaningful_components(stroke_mask)
    closed = close_mask(cleaned, iterations)
    filled = fill_holes(closed)
    filled = dilate(filled, max(1, iterations // 3))
    anchored = np.zeros_like(filled, dtype=bool)
    for comp in components(filled, min_area=max(80, int(filled.size * 0.000025))):
        comp_mask = np.zeros_like(filled, dtype=bool)
        for x, y in comp["points"]:
            comp_mask[y, x] = True
        if (comp_mask & dilate(cleaned, iterations)).any():
            for x, y in comp["points"]:
                anchored[y, x] = True
    return anchored


def extract_class_strokes(raw_path: Path, annotation_path: Path, size: tuple[int, int]) -> dict[str, np.ndarray]:
    raw = np.asarray(open_rgb(raw_path, size)).astype(np.int32)
    annotated = np.asarray(open_rgb(annotation_path, size)).astype(np.int32)
    diff = np.abs(annotated - raw).max(axis=2)
    changed = diff > 35
    masks: dict[str, np.ndarray] = {}
    for class_name, info in CLASS_INFO.items():
        target = info["target"].astype(np.int32)
        distance = np.sqrt(((annotated - target) ** 2).sum(axis=2))
        masks[class_name] = (distance < 78) & changed
    return {name: keep_meaningful_components(mask) for name, mask in masks.items()}


def rendered_indices(raw_path: Path, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(open_rgb(raw_path, size)).astype(np.float32) / 255.0
    red = raw[:, :, 0]
    green = raw[:, :, 1]
    blue = raw[:, :, 2]
    green_index = np.clip((2.0 * green - red - blue + 2.0) / 4.0, 0.0, 1.0)
    visual_ndvi_proxy = np.clip((green - red) / (green + red + 1e-6), -1.0, 1.0)
    return green_index, visual_ndvi_proxy


def index_metrics_for_masks(raw_path: Path, size: tuple[int, int], masks: dict[str, np.ndarray]) -> dict[str, float | None]:
    green_index, visual_ndvi_proxy = rendered_indices(raw_path, size)
    metrics: dict[str, float | None] = {}
    for class_name in ("low", "medium", "high"):
        mask = masks[class_name]
        if not mask.any():
            metrics[f"{class_name}_green_index"] = None
            metrics[f"{class_name}_visual_ndvi_proxy"] = None
            continue
        metrics[f"{class_name}_green_index"] = float(green_index[mask].mean())
        metrics[f"{class_name}_visual_ndvi_proxy"] = float(visual_ndvi_proxy[mask].mean())
    focus_mask = masks["low"] | masks["medium"]
    if focus_mask.any():
        metrics["low_medium_green_index"] = float(green_index[focus_mask].mean())
        metrics["low_medium_visual_ndvi_proxy"] = float(visual_ndvi_proxy[focus_mask].mean())
    else:
        metrics["low_medium_green_index"] = None
        metrics["low_medium_visual_ndvi_proxy"] = None
    return metrics


def make_exclusive_masks(filled: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    low = filled["low"]
    medium = filled["medium"] & ~low
    high = filled["high"] & ~low & ~medium
    exclusive = {"low": low, "medium": medium, "high": high}
    field = low | medium | high
    return exclusive, field


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def fit_image(image: Image.Image, box: tuple[int, int], fill: str = "white") -> Image.Image:
    out = Image.new("RGB", box, fill)
    work = image.convert("RGB").copy()
    work.thumbnail(box, Image.Resampling.LANCZOS)
    out.paste(work, ((box[0] - work.size[0]) // 2, (box[1] - work.size[1]) // 2))
    return out


def make_class_overlay(raw_path: Path, masks: dict[str, np.ndarray], date: str, field: str, output: Path) -> None:
    raw = open_rgb(raw_path)
    raw.thumbnail((1500, 1500), Image.Resampling.LANCZOS)
    overlay = np.zeros((raw.size[1], raw.size[0], 4), dtype=np.uint8)
    for class_name in ("high", "medium", "low"):
        mask_img = Image.fromarray((masks[class_name].astype(np.uint8) * 255), mode="L").resize(raw.size, Image.Resampling.NEAREST)
        mask = np.asarray(mask_img) > 0
        color = CLASS_INFO[class_name]["display"]
        overlay[mask] = (*color, 138 if class_name != "high" else 110)
    out = raw.convert("RGBA")
    out.alpha_composite(Image.fromarray(overlay, mode="RGBA"))
    out = out.convert("RGB")
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, out.size[0], 112), fill=(0, 0, 0))
    draw.text((18, 12), f"{field} {date_label(date)} Vigour Classes", fill="white", font=FONT_H2)
    draw.text((18, 54), STAGE_BY_DATE.get(date, ("Growth stage review", ""))[0], fill=(250, 230, 160), font=FONT_BODY)
    legend_x = 18
    for class_name in ("low", "medium", "high"):
        color = CLASS_INFO[class_name]["display"]
        draw.rectangle((legend_x, 86, legend_x + 22, 108), fill=color)
        draw.text((legend_x + 30, 84), CLASS_INFO[class_name]["label"], fill="white", font=FONT_SMALL)
        legend_x += 190
    output.parent.mkdir(parents=True, exist_ok=True)
    out.save(output)


def make_intelligent_overlay(raw_path: Path, masks: dict[str, np.ndarray], date: str, field: str, source_label: str, output: Path) -> None:
    raw = open_rgb(raw_path)
    raw.thumbnail((1500, 1500), Image.Resampling.LANCZOS)
    green_index, visual_ndvi_proxy = rendered_indices(raw_path, raw.size)
    ndvi_norm = (visual_ndvi_proxy + 1.0) / 2.0
    overlay = np.zeros((raw.size[1], raw.size[0], 4), dtype=np.uint8)
    for class_name in ("high", "medium", "low"):
        mask_img = Image.fromarray((masks[class_name].astype(np.uint8) * 255), mode="L").resize(raw.size, Image.Resampling.NEAREST)
        mask = np.asarray(mask_img) > 0
        color = CLASS_INFO[class_name]["display"]
        if class_name == "high":
            alpha = 60 + 100 * green_index
        elif class_name == "medium":
            alpha = 75 + 95 * (1.0 - ndvi_norm)
        else:
            alpha = 95 + 125 * (1.0 - ndvi_norm)
        overlay[mask, 0] = color[0]
        overlay[mask, 1] = color[1]
        overlay[mask, 2] = color[2]
        overlay[mask, 3] = np.clip(alpha[mask], 0, 225).astype(np.uint8)
    out = raw.convert("RGBA")
    out.alpha_composite(Image.fromarray(overlay, mode="RGBA"))
    out = out.convert("RGB")
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, out.size[0], 140), fill=(0, 0, 0))
    draw.text((18, 12), f"{field} {date_label(date)} Intelligent Vigour Overlay", fill="white", font=FONT_H2)
    draw.text((18, 52), f"Source label: {source_label}. Human zones weighted by rendered green/visual NDVI context.", fill=(250, 230, 160), font=FONT_SMALL)
    draw.text((18, 78), "Zones come from human annotation only; green/NDVI index changes overlay emphasis, not boundaries.", fill=(250, 230, 160), font=FONT_SMALL)
    legend_x = 18
    for class_name in ("low", "medium", "high"):
        color = CLASS_INFO[class_name]["display"]
        draw.rectangle((legend_x, 110, legend_x + 22, 132), fill=color)
        draw.text((legend_x + 30, 108), CLASS_INFO[class_name]["label"], fill="white", font=FONT_SMALL)
        legend_x += 190
    output.parent.mkdir(parents=True, exist_ok=True)
    out.save(output)


def make_debug(raw_path: Path, annotation_path: Path, stroke_masks: dict[str, np.ndarray], exclusive_masks: dict[str, np.ndarray], overlay_path: Path, output: Path) -> None:
    def mask_panel(masks: dict[str, np.ndarray], title: str) -> Image.Image:
        h, w = next(iter(masks.values())).shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for class_name in ("high", "medium", "low"):
            rgb[masks[class_name]] = CLASS_INFO[class_name]["display"]
        image = Image.fromarray(rgb, mode="RGB")
        return titled_panel(image, title)

    panels = [
        titled_panel(open_rgb(raw_path), "Raw NDVI background"),
        titled_panel(open_rgb(annotation_path), "Human vigour annotation"),
        mask_panel(stroke_masks, "Extracted annotation strokes"),
        mask_panel(exclusive_masks, "Recovered filled class regions"),
        titled_panel(Image.open(overlay_path), "Raw NDVI with class regions"),
    ]
    width = sum(panel.size[0] for panel in panels)
    height = max(panel.size[1] for panel in panels)
    canvas = Image.new("RGB", (width, height), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.size[0]
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def titled_panel(image: Image.Image, title: str) -> Image.Image:
    img = image.convert("RGB")
    img.thumbnail((420, 420), Image.Resampling.LANCZOS)
    out = Image.new("RGB", (img.size[0], img.size[1] + 48), "white")
    draw = ImageDraw.Draw(out)
    draw.text((8, 10), title, fill=(20, 20, 20), font=FONT_SMALL)
    out.paste(img, (0, 48))
    return out


def draw_wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, width: int, font_obj: ImageFont.ImageFont, fill: str | tuple[int, int, int] = "black", line_gap: int = 7) -> int:
    x, y = xy
    chars = max(12, width // 12)
    for paragraph in text.split("\n"):
        for line in textwrap.wrap(paragraph, chars) or [""]:
            draw.text((x, y), line, fill=fill, font=font_obj)
            y += getattr(font_obj, "size", 22) + line_gap
    return y


def location_from_mask(mask: np.ndarray | None) -> str:
    if mask is None or not mask.any():
        return "not marked"
    comps = components(mask, min_area=max(64, int(mask.size * 0.00002)))
    if not comps:
        return "not marked"
    x1, y1, x2, y2 = comps[0]["bbox"]
    cx = (x1 + x2) / 2 / mask.shape[1]
    cy = (y1 + y2) / 2 / mask.shape[0]
    vertical = "upper" if cy < 0.33 else "lower" if cy > 0.66 else "central"
    horizontal = "left" if cx < 0.33 else "right" if cx > 0.66 else "center"
    return f"{vertical}-field area" if horizontal == "center" else f"{vertical}-{horizontal} field area"


def process_observation(annotation_path: Path) -> Observation:
    source_field, date = parse_filename(annotation_path)
    field = REPORT_FIELD_NAME
    raw_path = match_raw(source_field, date)
    size = analysis_size(Image.open(annotation_path).size)
    stage, stage_note = STAGE_BY_DATE.get(date, ("Growth stage review", "Confirm growth stage from field scouting notes."))
    stroke_masks = extract_class_strokes(raw_path, annotation_path, size)
    filled = {name: recover_filled_region(mask) for name, mask in stroke_masks.items()}
    exclusive, field_mask = make_exclusive_masks(filled)
    field_dir = OUT_DIR / field
    mask_paths: dict[str, Path] = {}
    for class_name, mask in exclusive.items():
        mask_path = field_dir / "class_masks" / f"{source_field}_{date}_{class_name}_vigour_mask.png"
        save_mask(mask, mask_path)
        mask_paths[class_name] = mask_path
    overlay_path = field_dir / "overlays" / f"{source_field}_{date}_human_vigour_class_overlay.png"
    intelligent_overlay_path = field_dir / "overlays" / f"{source_field}_{date}_intelligent_vigour_overlay.png"
    debug_path = field_dir / "debug" / f"{source_field}_{date}_human_vigour_class_debug.png"
    metrics = index_metrics_for_masks(raw_path, size, exclusive)
    make_class_overlay(raw_path, exclusive, date, field, overlay_path)
    make_intelligent_overlay(raw_path, exclusive, date, field, source_field, intelligent_overlay_path)
    make_debug(raw_path, annotation_path, stroke_masks, exclusive, overlay_path, debug_path)
    denominator = int(field_mask.sum())
    percents: dict[str, float | None] = {}
    if denominator == 0:
        percents = {"low": None, "medium": None, "high": None, "low_plus_medium": None}
        status = "manual_review_required"
        notes = "No human vigour class regions were recovered; no 0% values reported."
    else:
        for class_name in ("low", "medium", "high"):
            percents[class_name] = int(exclusive[class_name].sum()) / denominator * 100
        percents["low_plus_medium"] = (int(exclusive["low"].sum()) + int(exclusive["medium"].sum())) / denominator * 100
        status = "success"
        notes = "Classified from human annotation colors after raw-vs-annotated comparison. Green/visual NDVI indices provide overlay context only."
    return Observation(
        field_name=field,
        source_field_name=source_field,
        date=date,
        raw_path=raw_path,
        annotation_path=annotation_path,
        analysis_size=size,
        stage=stage,
        stage_note=stage_note,
        stroke_masks=stroke_masks,
        filled_masks=filled,
        exclusive_masks=exclusive,
        field_mask=field_mask,
        percents=percents,
        stroke_pixel_counts={name: int(mask.sum()) for name, mask in stroke_masks.items()},
        filled_mask_paths=mask_paths,
        overlay_path=overlay_path,
        intelligent_overlay_path=intelligent_overlay_path,
        debug_path=debug_path,
        index_metrics=metrics,
        status=status,
        notes=notes,
    )


def write_summary_csv(observations: Iterable[Observation]) -> None:
    columns = [
        "field_name",
        "source_image_label",
        "date",
        "growth_stage",
        "low_vigour_percent",
        "medium_vigour_percent",
        "high_vigour_percent",
        "low_plus_medium_vigour_percent",
        "low_green_index",
        "low_visual_ndvi_proxy",
        "medium_green_index",
        "medium_visual_ndvi_proxy",
        "high_green_index",
        "high_visual_ndvi_proxy",
        "low_medium_green_index",
        "low_medium_visual_ndvi_proxy",
        "trend_from_previous",
        "highest_priority_area",
        "raw_ndvi_path",
        "annotated_image_path",
        "extraction_status",
        "notes",
    ]
    previous_by_field: dict[str, Observation] = {}
    with (OUT_DIR / "vigour_class_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for obs in observations:
            prev = previous_by_field.get(obs.field_name)
            current = obs.percents["low_plus_medium"]
            prior = prev.percents["low_plus_medium"] if prev else None
            if current is None:
                trend = "manual_review_required"
            elif prior is None:
                trend = "first_observation"
            elif abs(current - prior) < 0.5:
                trend = "stable"
            elif current > prior:
                trend = "increased"
            else:
                trend = "decreased"
            writer.writerow(
                {
                    "field_name": obs.field_name,
                    "source_image_label": obs.source_field_name,
                    "date": obs.date,
                    "growth_stage": obs.stage,
                    "low_vigour_percent": "" if obs.percents["low"] is None else f"{obs.percents['low']:.2f}",
                    "medium_vigour_percent": "" if obs.percents["medium"] is None else f"{obs.percents['medium']:.2f}",
                    "high_vigour_percent": "" if obs.percents["high"] is None else f"{obs.percents['high']:.2f}",
                    "low_plus_medium_vigour_percent": "" if current is None else f"{current:.2f}",
                    "low_green_index": "" if obs.index_metrics["low_green_index"] is None else f"{obs.index_metrics['low_green_index']:.3f}",
                    "low_visual_ndvi_proxy": "" if obs.index_metrics["low_visual_ndvi_proxy"] is None else f"{obs.index_metrics['low_visual_ndvi_proxy']:.3f}",
                    "medium_green_index": "" if obs.index_metrics["medium_green_index"] is None else f"{obs.index_metrics['medium_green_index']:.3f}",
                    "medium_visual_ndvi_proxy": "" if obs.index_metrics["medium_visual_ndvi_proxy"] is None else f"{obs.index_metrics['medium_visual_ndvi_proxy']:.3f}",
                    "high_green_index": "" if obs.index_metrics["high_green_index"] is None else f"{obs.index_metrics['high_green_index']:.3f}",
                    "high_visual_ndvi_proxy": "" if obs.index_metrics["high_visual_ndvi_proxy"] is None else f"{obs.index_metrics['high_visual_ndvi_proxy']:.3f}",
                    "low_medium_green_index": "" if obs.index_metrics["low_medium_green_index"] is None else f"{obs.index_metrics['low_medium_green_index']:.3f}",
                    "low_medium_visual_ndvi_proxy": "" if obs.index_metrics["low_medium_visual_ndvi_proxy"] is None else f"{obs.index_metrics['low_medium_visual_ndvi_proxy']:.3f}",
                    "trend_from_previous": trend,
                    "highest_priority_area": location_from_mask(obs.exclusive_masks["low"]),
                    "raw_ndvi_path": str(obs.raw_path),
                    "annotated_image_path": str(obs.annotation_path),
                    "extraction_status": obs.status,
                    "notes": obs.notes,
                }
            )
            previous_by_field[obs.field_name] = obs


def make_timeline(field: str, observations: list[Observation]) -> Path:
    output = OUT_DIR / field / "june_bearing_vigour_timeline.png"
    tile_w, tile_h = 450, 555
    cols = min(4, max(2, len(observations)))
    rows = math.ceil(len(observations) / cols)
    header_h = 136
    canvas = Image.new("RGB", (cols * tile_w, header_h + rows * tile_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.size[0], header_h), fill=(22, 32, 42))
    title_font = FONT_H1 if canvas.size[0] >= 1500 else FONT_H2
    draw.text((28, 18), f"{field} June-Bearing Vigour Timeline", fill="white", font=title_font)
    draw.text((30, 82), "Red = low vigour, purple = medium vigour, light green = high vigour.", fill=(240, 226, 165), font=FONT_BODY)
    for index, obs in enumerate(observations):
        x = (index % cols) * tile_w
        y = header_h + (index // cols) * tile_h
        draw.rectangle((x, y, x + tile_w - 1, y + tile_h - 1), outline=(216, 216, 216), width=2)
        image = fit_image(Image.open(obs.overlay_path), (tile_w - 24, 320))
        canvas.paste(image, (x + 12, y + 16))
        low = obs.percents["low"]
        med = obs.percents["medium"]
        high = obs.percents["high"]
        low_med = obs.percents["low_plus_medium"]
        draw.text((x + 22, y + 356), f"{date_label(obs.date)} - {obs.stage}", fill=(30, 30, 30), font=FONT_BODY_BOLD)
        draw.text((x + 22, y + 394), f"Low: {'review' if low is None else f'{low:.1f}%'}", fill=CLASS_INFO["low"]["display"], font=FONT_BODY)
        draw.text((x + 150, y + 394), f"Medium: {'review' if med is None else f'{med:.1f}%'}", fill=CLASS_INFO["medium"]["display"], font=FONT_BODY)
        draw.text((x + 22, y + 430), f"High: {'review' if high is None else f'{high:.1f}%'}", fill=(70, 145, 30), font=FONT_BODY)
        draw.text((x + 22, y + 466), f"Scout focus: {'review' if low_med is None else f'{low_med:.1f}%'} low+medium", fill=(65, 65, 65), font=FONT_BODY)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def make_trend_chart(field: str, observations: list[Observation]) -> Path:
    output = OUT_DIR / field / "vigour_class_percent_over_time.png"
    width, height = 1300, 760
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((48, 30), f"{field} Human-Marked Vigour Classes Over Time", fill=(20, 20, 20), font=FONT_H2)
    draw.text((48, 72), "Percentages are based on recovered human-marked class regions, not raw NDVI thresholds.", fill=(70, 70, 70), font=FONT_SMALL)
    chart = (110, 150, 1220, 615)
    draw.rectangle(chart, outline=(40, 40, 40), width=2)
    for step in range(6):
        y = chart[3] - (chart[3] - chart[1]) * step / 5
        draw.line((chart[0], y, chart[2], y), fill=(225, 225, 225), width=1)
        draw.text((50, y - 12), f"{step * 20}%", fill=(80, 80, 80), font=FONT_SMALL)
    bar_gap = 34
    bar_w = max(60, int((chart[2] - chart[0] - bar_gap * (len(observations) + 1)) / max(1, len(observations))))
    for index, obs in enumerate(observations):
        x0 = chart[0] + bar_gap + index * (bar_w + bar_gap)
        y_base = chart[3]
        for class_name in ("low", "medium", "high"):
            value = obs.percents[class_name]
            if value is None:
                continue
            seg_h = (chart[3] - chart[1]) * value / 100
            y1 = y_base - seg_h
            draw.rectangle((x0, y1, x0 + bar_w, y_base), fill=CLASS_INFO[class_name]["display"])
            y_base = y1
        draw.text((x0, chart[3] + 20), date_label(obs.date), fill=(30, 30, 30), font=FONT_SMALL)
        focus = obs.percents["low_plus_medium"]
        if focus is not None:
            draw.text((x0 - 6, y_base - 28), f"L+M {focus:.1f}%", fill=(30, 30, 30), font=FONT_SMALL)
    legend_x = 770
    for class_name in ("low", "medium", "high"):
        draw.rectangle((legend_x, 665, legend_x + 24, 689), fill=CLASS_INFO[class_name]["display"])
        draw.text((legend_x + 32, 662), CLASS_INFO[class_name]["label"], fill=(40, 40, 40), font=FONT_SMALL)
        legend_x += 160
    draw.text((570, 706), "Date", fill=(45, 45, 45), font=FONT_BODY_BOLD)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def make_dashboard(field: str, observations: list[Observation], timeline: Path, chart: Path) -> Path:
    output = OUT_DIR / field / "june_bearing_farmer_dashboard.png"
    width, height = 1800, 2120
    canvas = Image.new("RGB", (width, height), (248, 249, 247))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, 170), fill=(22, 32, 42))
    draw.text((54, 34), f"{field} June-Bearing Vigour Class Progression", fill="white", font=FONT_H1)
    draw.text((56, 96), "Human annotation classes by growth stage. For scouting prioritization only.", fill=(235, 225, 170), font=FONT_BODY)
    draw.text((54, 214), "1. Vigour Class Timeline", fill=(25, 25, 25), font=FONT_H2)
    canvas.paste(fit_image(Image.open(timeline), (1688, 585)), (56, 260))
    draw.rectangle((56, 260, 1744, 845), outline=(210, 210, 210), width=2)
    draw.text((54, 925), "2. Class Coverage Trend", fill=(25, 25, 25), font=FONT_H2)
    canvas.paste(fit_image(Image.open(chart), (824, 480)), (56, 970))
    draw.rectangle((56, 970, 880, 1450), outline=(210, 210, 210), width=2)
    latest = observations[-1]
    draw.text((936, 925), "3. Latest Class Overlay", fill=(25, 25, 25), font=FONT_H2)
    canvas.paste(fit_image(Image.open(latest.overlay_path), (808, 480)), (936, 970))
    draw.rectangle((936, 970, 1744, 1450), outline=(210, 210, 210), width=2)

    focus_obs = max(observations, key=lambda obs: obs.percents["low_plus_medium"] or -1)
    blocks = [
        ("4. Highest Low-Vigour Priority", f"{location_from_mask(focus_obs.exclusive_masks['low'])} on {date_label(focus_obs.date)}."),
        ("5. Growth Stage Context", latest.stage_note),
        ("6. Next Scouting Decision", "Prioritize red low-vigour regions first, then purple medium-vigour regions that persist or expand near harvest. Use light-green areas as high-vigour reference zones."),
    ]
    x_positions = [56, 636, 1216]
    for x, (title, body) in zip(x_positions, blocks):
        draw.rectangle((x, 1545, x + 500, 1940), fill="white", outline=(215, 215, 215), width=2)
        yy = draw_wrapped(draw, (x + 24, 1574), title, 440, FONT_BODY_BOLD, fill=(25, 25, 25))
        yy += 22
        draw_wrapped(draw, (x + 28, yy), body, 440, FONT_BODY, fill=(55, 55, 55))
    draw.rectangle((0, height - 108, width, height), fill=(22, 32, 42))
    draw.text((56, height - 72), "For scouting prioritization only. Not a disease diagnosis.", fill="white", font=FONT_BODY_BOLD)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def write_field_report(field: str, observations: list[Observation], timeline: Path, chart: Path, dashboard: Path) -> Path:
    output = OUT_DIR / field / "june_bearing_field_report.md"
    lines = [
        f"# {field} June-Bearing Strawberry Vigour Class Report",
        "",
        "Human annotation color key:",
        "",
        "- Red = low vigour",
        "- Purple = medium vigour",
        "- Light green = high vigour",
        "",
        "Raw NDVI was used as background/context only. The class report uses the added human annotation colors after raw-vs-annotated comparison.",
        "",
        f"- Timeline: [{timeline.name}]({timeline.name})",
        f"- Trend chart: [{chart.name}]({chart.name})",
        f"- Dashboard: [{dashboard.name}]({dashboard.name})",
        "",
        "## Date-Level Observations",
        "",
    ]
    for obs in observations:
        low = obs.percents["low"]
        med = obs.percents["medium"]
        high = obs.percents["high"]
        focus = obs.percents["low_plus_medium"]
        lines.extend(
            [
                f"### {date_label(obs.date)} - {obs.stage}",
                "",
                f"- Low vigour: {'manual review required' if low is None else f'{low:.2f}%'}",
                f"- Medium vigour: {'manual review required' if med is None else f'{med:.2f}%'}",
                f"- High vigour: {'manual review required' if high is None else f'{high:.2f}%'}",
                f"- Low + medium scouting focus: {'manual review required' if focus is None else f'{focus:.2f}%'}",
                f"- Highest priority area: {location_from_mask(obs.exclusive_masks['low'])}",
                f"- Growth-stage note: {obs.stage_note}",
                "",
            ]
        )
    first = observations[0]
    latest = observations[-1]
    first_focus = first.percents["low_plus_medium"]
    latest_focus = latest.percents["low_plus_medium"]
    trend = "manual review required"
    if len(observations) == 1:
        trend = "single-date snapshot"
    elif first_focus is not None and latest_focus is not None:
        if latest_focus > first_focus:
            trend = "increasing"
        elif latest_focus < first_focus:
            trend = "decreasing"
        else:
            trend = "stable"
    lines.extend(
        [
            "## Operational Interpretation",
            "",
            (
                "Only one selected date is available for this field, so increase/decrease is not reported."
                if len(observations) == 1
                else f"Low + medium vigour scouting focus is {trend} across the available observation window for this field."
            ),
            "",
            "Red regions should be treated as the highest scouting priority. Purple regions should be monitored as medium-vigour areas, especially where they persist near ripening or harvest-time observations. Light-green regions provide high-vigour reference areas for comparison.",
            "",
            "This is not a disease diagnosis. Use it to decide where to scout, compare management history, and investigate irrigation, drainage, fertility, crop history, field operations, and disease pressure only as possible factors to review.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def write_final_report(by_field: dict[str, list[Observation]], artifacts: dict[str, dict[str, Path]]) -> Path:
    output = OUT_DIR / "june_bearing_farmer_ready_report.md"
    lines = [
        "# June-Bearing Strawberry Human Vigour Class Report",
        "",
        "This report uses the updated human annotations as ground truth.",
        "",
        "- Red = low vigour",
        "- Purple = medium vigour",
        "- Light green = high vigour",
        "",
        "Raw NDVI is background/context only. No NDVI thresholding, disease diagnosis, SAM/DINO, or invented polygons were used.",
        "",
        "## Growth Stage Context",
        "",
        "The dates in the filenames are interpreted as June-bearing strawberry observations moving from bloom/fruit set into green fruit development and harvest-time ripening. Field notes should override this stage estimate if local phenology differs.",
        "",
    ]
    for field, observations in by_field.items():
        first = observations[0]
        latest = observations[-1]
        first_focus = first.percents["low_plus_medium"]
        latest_focus = latest.percents["low_plus_medium"]
        if len(observations) == 1:
            trend_word = "single-date snapshot"
            trend_sentence = f"Only {date_label(latest.date)} is available for this field, so increase/decrease is not reported."
        elif first_focus is None or latest_focus is None:
            trend_sentence = "Manual review is required before reporting a trend."
            trend_word = "manual_review_required"
        elif latest_focus > first_focus:
            trend_word = "increasing"
            trend_sentence = f"Low + medium vigour scouting focus increased from {first_focus:.2f}% to {latest_focus:.2f}%."
        elif latest_focus < first_focus:
            trend_word = "decreasing"
            trend_sentence = f"Low + medium vigour scouting focus decreased from {first_focus:.2f}% to {latest_focus:.2f}%."
        else:
            trend_word = "stable"
            trend_sentence = f"Low + medium vigour scouting focus remained stable at {latest_focus:.2f}%."
        coverage = "; ".join(
            f"{date_label(obs.date)} ({obs.stage}): low {'review' if obs.percents['low'] is None else f'{obs.percents['low']:.2f}%'}, medium {'review' if obs.percents['medium'] is None else f'{obs.percents['medium']:.2f}%'}, high {'review' if obs.percents['high'] is None else f'{obs.percents['high']:.2f}%'}"
            for obs in observations
        )
        focus_obs = max(observations, key=lambda obs: obs.percents["low_plus_medium"] or -1)
        lines.extend(
            [
                f"## {field}",
                "",
                f"- Field report: [{artifacts[field]['report'].name}]({field}/{artifacts[field]['report'].name})",
                f"- Dashboard: [{artifacts[field]['dashboard'].name}]({field}/{artifacts[field]['dashboard'].name})",
                f"- Timeline: [{artifacts[field]['timeline'].name}]({field}/{artifacts[field]['timeline'].name})",
                f"- Trend chart: [{artifacts[field]['chart'].name}]({field}/{artifacts[field]['chart'].name})",
                "",
                f"1. Where is the highest-priority low-vigour area? {location_from_mask(focus_obs.exclusive_masks['low'])}.",
                "",
                f"2. Which date carries the highest scouting focus? {date_label(focus_obs.date)} during {focus_obs.stage}.",
                "",
                f"3. What is the vigour class progression? {coverage}.",
                "",
                f"4. Is low + medium vigour focus increasing, stable, or decreasing? {trend_word}.",
                "",
                f"5. What changed operationally? {trend_sentence}",
                "",
                "6. What should be scouted next? Red low-vigour zones first, then purple medium-vigour zones that remain visible during fruit sizing, ripening, or harvest-time observations.",
                "",
                "The highlighted human-marked classes represent scouting priorities and vigour reference areas. They are not disease percentages.",
                "",
            ]
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    annotation_paths = [ANNOTATION_DIR / name for name in SELECTED_ANNOTATION_FILES]
    observations = [process_observation(path) for path in annotation_paths]
    observations.sort(key=lambda obs: (obs.field_name, obs.date))
    by_field: dict[str, list[Observation]] = {}
    for obs in observations:
        by_field.setdefault(obs.field_name, []).append(obs)
    write_summary_csv(observations)
    artifacts: dict[str, dict[str, Path]] = {}
    for field, field_observations in by_field.items():
        timeline = make_timeline(field, field_observations)
        chart = make_trend_chart(field, field_observations)
        dashboard = make_dashboard(field, field_observations, timeline, chart)
        report = write_field_report(field, field_observations, timeline, chart, dashboard)
        artifacts[field] = {"timeline": timeline, "chart": chart, "dashboard": dashboard, "report": report}
    final_report = write_final_report(by_field, artifacts)
    print(f"June-bearing vigour report output: {OUT_DIR}")
    print(f"Summary CSV: {OUT_DIR / 'vigour_class_summary.csv'}")
    print(f"Final report: {final_report}")
    print("Rule honored: classified only human-added annotation colors after raw-vs-annotated comparison.")


if __name__ == "__main__":
    main()
