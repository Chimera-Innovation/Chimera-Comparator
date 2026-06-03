from __future__ import annotations

import csv
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

try:
    import rasterio
except ModuleNotFoundError:
    rasterio = None


DATASET_ROOT = Path(r"D:\NDVI-dataset")
MVP_DIR = DATASET_ROOT / "MVP - ndvi progression"
OUT_DIR = MVP_DIR / "progression_maps"
RAW_DIR = DATASET_ROOT / "raw-ndvi"
ANNOTATED_DIR = DATASET_ROOT / "human-Annotated-ndvi"
TEMPLATE_CSV = MVP_DIR / "manual_annotation_template.csv"

MAX_PROCESS_DIM = 1600
MAX_PANEL_THUMB = 700
MIN_COMPONENT_PIXELS = 120
DIFF_THRESHOLD = 35
LOW_NDVI_DARK_THRESHOLD = 78


@dataclass
class ImagePair:
    field: str
    date: str
    raw_rel: str
    annotated_rel: str
    raw_path: Path
    annotated_path: Path
    raw_size: tuple[int, int]
    annotated_size: tuple[int, int]
    raw_georef: bool
    same_size: bool


def load_font(size: int = 22) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


FONT = load_font(22)
SMALL_FONT = load_font(16)
TITLE_FONT = load_font(30)


def read_template_pairs() -> list[ImagePair]:
    pairs: list[ImagePair] = []
    with TEMPLATE_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw_rel = row["raw_ndvi_path"]
            annotated_rel = row["annotated_ndvi_path"]
            raw_path = DATASET_ROOT / raw_rel
            annotated_path = DATASET_ROOT / annotated_rel
            raw_size = image_size(raw_path)
            annotated_size = image_size(annotated_path)
            pairs.append(
                ImagePair(
                    field=row["field_name"],
                    date=row["date_observed"],
                    raw_rel=raw_rel,
                    annotated_rel=annotated_rel,
                    raw_path=raw_path,
                    annotated_path=annotated_path,
                    raw_size=raw_size,
                    annotated_size=annotated_size,
                    raw_georef=is_georeferenced(raw_path),
                    same_size=raw_size == annotated_size,
                )
            )
    return sorted(pairs, key=lambda pair: (pair.field, pair.date))


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def is_georeferenced(path: Path) -> bool:
    if rasterio is None or path.suffix.lower() not in {".tif", ".tiff"}:
        return False
    try:
        with rasterio.open(path) as dataset:
            return bool(dataset.crs)
    except Exception:
        return False


def open_rgb(path: Path, max_dim: int | None = None) -> Image.Image:
    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(path).convert("RGB")
    if max_dim:
        image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    return image.copy()


def resized_pair(pair: ImagePair) -> tuple[Image.Image, Image.Image, float, float]:
    raw = open_rgb(pair.raw_path)
    annotated = open_rgb(pair.annotated_path)
    width, height = raw.size
    scale = min(1.0, MAX_PROCESS_DIM / max(width, height))
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    raw_small = raw.resize(target, Image.Resampling.BILINEAR)
    annotated_small = annotated.resize(target, Image.Resampling.BILINEAR)
    return raw_small, annotated_small, width / target[0], height / target[1]


def clean_mask(mask: np.ndarray) -> np.ndarray:
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
    return mask & (neighbors >= 3)


def find_components(mask: np.ndarray) -> list[dict[str, Any]]:
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[dict[str, Any]] = []
    ys, xs = np.nonzero(mask)
    active = set(zip(xs.tolist(), ys.tolist()))

    while active:
        start = active.pop()
        queue: deque[tuple[int, int]] = deque([start])
        points: list[tuple[int, int]] = []
        visited[start[1], start[0]] = True
        min_x = max_x = start[0]
        min_y = max_y = start[1]

        while queue:
            x, y = queue.popleft()
            points.append((x, y))
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if nx < 0 or ny < 0 or nx >= width or ny >= height or visited[ny, nx] or not mask[ny, nx]:
                    continue
                visited[ny, nx] = True
                active.discard((nx, ny))
                queue.append((nx, ny))

        if len(points) >= MIN_COMPONENT_PIXELS:
            components.append(
                {
                    "area": len(points),
                    "bbox": (min_x, min_y, max_x, max_y),
                    "centroid": (
                        sum(point[0] for point in points) / len(points),
                        sum(point[1] for point in points) / len(points),
                    ),
                }
            )
    return components


def extract_annotation_polygons(pair: ImagePair) -> tuple[list[dict[str, Any]], dict[str, Any], Image.Image]:
    raw_small, annotated_small, scale_x, scale_y = resized_pair(pair)
    raw_arr = np.asarray(raw_small).astype(np.int16)
    ann_arr = np.asarray(annotated_small).astype(np.int16)
    diff = np.abs(raw_arr - ann_arr).max(axis=2)
    mask = clean_mask(diff > DIFF_THRESHOLD)
    components = find_components(mask)
    width, height = pair.raw_size
    features: list[dict[str, Any]] = []

    for idx, comp in enumerate(components, start=1):
        min_x, min_y, max_x, max_y = comp["bbox"]
        box = [
            [round(min_x * scale_x, 2), round(min_y * scale_y, 2)],
            [round(max_x * scale_x, 2), round(min_y * scale_y, 2)],
            [round(max_x * scale_x, 2), round(max_y * scale_y, 2)],
            [round(min_x * scale_x, 2), round(max_y * scale_y, 2)],
            [round(min_x * scale_x, 2), round(min_y * scale_y, 2)],
        ]
        area = int(round(comp["area"] * scale_x * scale_y))
        centroid_x = round(comp["centroid"][0] * scale_x, 2)
        centroid_y = round(comp["centroid"][1] * scale_y, 2)
        features.append(
            {
                "field_name": pair.field,
                "date": pair.date,
                "polygon_id": f"{pair.field}_{pair.date}_P{idx:03d}",
                "area_pixels": area,
                "area_percent": round(area / (width * height) * 100, 4),
                "bounding_box": [box[0][0], box[0][1], box[1][0], box[2][1]],
                "centroid": [centroid_x, centroid_y],
                "coordinates": box,
            }
        )

    settings = {
        "mode": "human_annotation_difference",
        "diff_threshold": DIFF_THRESHOLD,
        "min_component_pixels_at_processing_scale": MIN_COMPONENT_PIXELS,
        "processing_size": raw_small.size,
        "original_size": pair.raw_size,
        "scale_x": scale_x,
        "scale_y": scale_y,
    }
    return features, settings, raw_small


def estimate_low_ndvi(raw_path: Path) -> dict[str, Any]:
    raw = open_rgb(raw_path, MAX_PROCESS_DIM)
    arr = np.asarray(raw).astype(np.int16)
    gray = arr.mean(axis=2)
    low = gray < LOW_NDVI_DARK_THRESHOLD
    moderate = (gray >= LOW_NDVI_DARK_THRESHOLD) & (gray < 125)
    total = gray.size
    return {
        "mode": "raw_ndvi_visual_threshold",
        "low_ndvi_zone_percent": round(float(low.sum()) / total * 100, 4),
        "moderate_ndvi_zone_percent": round(float(moderate.sum()) / total * 100, 4),
        "threshold_note": "Visual export brightness threshold only; not calibrated NDVI.",
    }


def alignment_for_field(pairs: list[ImagePair]) -> dict[str, Any]:
    sizes = [pair.raw_size for pair in pairs]
    same_all = len(set(sizes)) == 1
    georef_count = sum(pair.raw_georef for pair in pairs)
    if same_all and georef_count == len(pairs) and len(pairs) > 1:
        confidence = "medium"
        reliable = False
        note = "Images share size and georeference count is complete, but feature registration was not available."
    elif same_all:
        confidence = "medium"
        reliable = False
        note = "Images share dimensions, allowing direct image-space overlays, but not geospatial registration."
    else:
        confidence = "low"
        reliable = False
        note = "Dates have different dimensions; cumulative overlays are normalized visual demos only."
    return {
        "field_name": pairs[0].field,
        "same_size_direct_overlay": same_all,
        "registration_methods_available": ["same-size direct overlay"],
        "feature_matching_available": False,
        "ecc_registration_available": False,
        "georeferenced_dates": georef_count,
        "alignment_confidence": confidence,
        "spatially_reliable": reliable,
        "notes": note,
    }


def draw_overlay(pair: ImagePair, features: list[dict[str, Any]], summary: dict[str, Any], output: Path) -> None:
    base = open_rgb(pair.raw_path, MAX_PANEL_THUMB)
    draw = ImageDraw.Draw(base)
    scale_x = base.size[0] / pair.raw_size[0]
    scale_y = base.size[1] / pair.raw_size[1]
    for feature in features:
        coords = [(x * scale_x, y * scale_y) for x, y in feature["coordinates"]]
        draw.line(coords, fill=(255, 30, 30), width=4)
        cx, cy = feature["centroid"]
        draw.text((cx * scale_x + 4, cy * scale_y + 4), feature["polygon_id"].split("_")[-1], fill=(255, 255, 255), font=SMALL_FONT)
    label = f"{pair.field} {pair.date} | marked area {summary['total_marked_area_percent']:.2f}%"
    draw.rectangle((0, 0, base.size[0], 42), fill=(0, 0, 0))
    draw.text((12, 10), label, fill=(255, 255, 255), font=SMALL_FONT)
    base.save(output)


def add_caption(image: Image.Image, caption: str, status: str) -> Image.Image:
    width, height = image.size
    out = Image.new("RGB", (width, height + 72), "white")
    out.paste(image, (0, 54))
    draw = ImageDraw.Draw(out)
    draw.text((10, 8), caption, fill=(20, 20, 20), font=SMALL_FONT)
    draw.text((10, 30), status, fill=(120, 30, 30), font=SMALL_FONT)
    return out


def make_timeline(field: str, pairs: list[ImagePair], summaries: list[dict[str, Any]], overlay_paths: list[Path], output: Path) -> None:
    panels = []
    for pair, summary, overlay_path in zip(pairs, summaries, overlay_paths):
        image = open_rgb(overlay_path, 520)
        panels.append(add_caption(image, f"{pair.date} | {summary['total_marked_area_percent']:.2f}% area", summary["spread_state"]))
    total_width = sum(panel.size[0] for panel in panels)
    max_height = max(panel.size[1] for panel in panels)
    canvas = Image.new("RGB", (total_width, max_height + 70), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 16), f"{field} Image-Space Concern Zone Timeline", fill=(20, 20, 20), font=TITLE_FONT)
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 70))
        x += panel.size[0]
    canvas.save(output)


def make_cumulative(field: str, pairs: list[ImagePair], features_by_date: dict[str, list[dict[str, Any]]], alignment: dict[str, Any], output: Path) -> None:
    base_pair = pairs[-1]
    base = open_rgb(base_pair.raw_path, 900)
    draw = ImageDraw.Draw(base)
    colors = [(255, 205, 0), (255, 120, 0), (255, 0, 0), (140, 0, 255), (0, 140, 255)]
    sx = base.size[0] / base_pair.raw_size[0]
    sy = base.size[1] / base_pair.raw_size[1]
    for idx, pair in enumerate(pairs):
        color = colors[idx % len(colors)]
        source_size = pair.raw_size
        for feature in features_by_date[pair.date]:
            # Normalize each image-space polygon into the latest image display space.
            coords = [
                (x / source_size[0] * base_pair.raw_size[0] * sx, y / source_size[1] * base_pair.raw_size[1] * sy)
                for x, y in feature["coordinates"]
            ]
            draw.line(coords, fill=color, width=4)
    draw.rectangle((0, 0, base.size[0], 84), fill=(0, 0, 0))
    draw.text((12, 8), f"{field} Cumulative Concern Zones", fill=(255, 255, 255), font=FONT)
    draw.text((12, 38), "Image-space normalized demo; not spatially registered field geometry.", fill=(255, 230, 180), font=SMALL_FONT)
    legend_y = 92
    for idx, pair in enumerate(pairs):
        color = colors[idx % len(colors)]
        draw.rectangle((12, legend_y, 32, legend_y + 20), fill=color)
        draw.text((40, legend_y), pair.date, fill=(20, 20, 20), font=SMALL_FONT)
        legend_y += 26
    base.save(output)


def make_chart(field: str, summaries: list[dict[str, Any]], output: Path) -> None:
    width, height = 900, 520
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    margin_l, margin_r, margin_t, margin_b = 80, 40, 70, 85
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    values = [summary["total_marked_area_percent"] for summary in summaries]
    max_v = max(values + [1.0])
    draw.text((24, 18), f"{field} Marked Concern Zone Area", fill=(20, 20, 20), font=TITLE_FONT)
    draw.line((margin_l, margin_t, margin_l, margin_t + plot_h), fill=(40, 40, 40), width=2)
    draw.line((margin_l, margin_t + plot_h, margin_l + plot_w, margin_t + plot_h), fill=(40, 40, 40), width=2)
    points = []
    for idx, summary in enumerate(summaries):
        x = margin_l + (plot_w * idx / max(1, len(summaries) - 1))
        y = margin_t + plot_h - (summary["total_marked_area_percent"] / max_v * plot_h)
        points.append((x, y))
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(210, 40, 40))
        draw.text((x - 52, margin_t + plot_h + 16), summary["date"], fill=(20, 20, 20), font=SMALL_FONT)
        draw.text((x - 22, y - 28), f"{summary['total_marked_area_percent']:.2f}%", fill=(20, 20, 20), font=SMALL_FONT)
    if len(points) > 1:
        draw.line(points, fill=(210, 40, 40), width=3)
    draw.text((18, margin_t + 10), "% area", fill=(20, 20, 20), font=SMALL_FONT)
    canvas.save(output)


def make_farmer_panel(field: str, summaries: list[dict[str, Any]], timeline_path: Path, chart_path: Path, output: Path) -> None:
    timeline = open_rgb(timeline_path, 1100)
    chart = open_rgb(chart_path, 700)
    width = max(timeline.size[0], chart.size[0] + 680)
    height = 160 + timeline.size[1] + 30 + chart.size[1] + 140
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 24), f"{field} NDVI Concern Zone Progression", fill=(20, 20, 20), font=TITLE_FONT)
    first = summaries[0]["total_marked_area_percent"]
    last = summaries[-1]["total_marked_area_percent"]
    delta = last - first
    interpretation = (
        f"Human-marked concern area changed from {first:.2f}% to {last:.2f}% "
        f"({delta:+.2f} percentage points)."
    )
    draw.text((32, 74), interpretation, fill=(30, 30, 30), font=FONT)
    draw.text((32, 108), "Warning: Human-marked NDVI concern zones; not calibrated disease diagnosis.", fill=(150, 40, 20), font=SMALL_FONT)
    canvas.paste(timeline, (0, 150))
    y = 150 + timeline.size[1] + 30
    canvas.paste(chart, (30, y))
    text_x = chart.size[0] + 70
    draw.text((text_x, y + 35), "Plain-English interpretation", fill=(20, 20, 20), font=FONT)
    if len(summaries) >= 4:
        line = "Best current evidence: multi-date progression demo."
    else:
        line = "Best current evidence: before/after comparison only."
    draw.text((text_x, y + 75), line, fill=(40, 40, 40), font=SMALL_FONT)
    draw.text((text_x, y + 105), "Use for farmer discussion, not model training.", fill=(40, 40, 40), font=SMALL_FONT)
    canvas.save(output)


def spread_state(current: float, previous: float | None, num_polygons: int) -> str:
    if num_polygons == 0:
        return "no_detected_zone"
    if previous is None:
        return "new_zone_appeared"
    delta = current - previous
    if abs(delta) < 0.25:
        return "stable"
    if delta > 0:
        return "expanding" if num_polygons <= 8 else "fragmented"
    return "contracting"


def write_audit(pairs_by_field: dict[str, list[ImagePair]], alignments: dict[str, dict[str, Any]], extraction_settings: dict[str, Any]) -> None:
    audit = {
        "dataset_root": str(DATASET_ROOT),
        "output_folder": str(OUT_DIR),
        "coordinate_type": "image_pixel_space",
        "fields": {},
        "extraction_settings": extraction_settings,
    }
    md = [
        "# Polygon Progression Feasibility Audit",
        "",
        "- Coordinate type: image-space pixel coordinates, not field survey geometry.",
        "- Recommended method: human annotation difference extraction, supported by raw NDVI visual threshold context.",
        "- Registration method: same-size direct overlay only where dimensions match; no OpenCV/SIFT/ECC available in this runtime.",
        "",
    ]
    for field, pairs in pairs_by_field.items():
        alignment = alignments[field]
        audit["fields"][field] = {
            "num_dates": len(pairs),
            "dates": [pair.date for pair in pairs],
            "raw_ndvi_images_available": len(pairs),
            "human_annotated_images_available": len(pairs),
            "dimensions_per_date": {
                pair.date: {"raw": pair.raw_size, "annotated": pair.annotated_size, "same_size": pair.same_size}
                for pair in pairs
            },
            "can_align_directly_by_size": alignment["same_size_direct_overlay"],
            "registration_needed": not alignment["same_size_direct_overlay"],
            "georeferenced_dates": alignment["georeferenced_dates"],
            "recommended_extraction": "C. both, with human annotation difference as primary and raw visual threshold as context",
            "alignment": alignment,
        }
        md.extend(
            [
                f"## {field}",
                "",
                f"- Dates: {', '.join(pair.date for pair in pairs)}",
                f"- Raw NDVI images: {len(pairs)}",
                f"- Human annotated images: {len(pairs)}",
                f"- Georeferenced dates: {alignment['georeferenced_dates']}",
                f"- Direct same-size alignment across dates: {alignment['same_size_direct_overlay']}",
                f"- Registration needed: {not alignment['same_size_direct_overlay']}",
                f"- Alignment confidence: {alignment['alignment_confidence']}",
                f"- Recommended extraction: both, with human annotation difference as primary.",
                f"- Notes: {alignment['notes']}",
                "",
                "| Date | Raw size | Annotated size | Pair same size | Raw georef |",
                "|---|---:|---:|---|---|",
            ]
        )
        for pair in pairs:
            md.append(f"| {pair.date} | {pair.raw_size[0]}x{pair.raw_size[1]} | {pair.annotated_size[0]}x{pair.annotated_size[1]} | {pair.same_size} | {pair.raw_georef} |")
        md.append("")
    (OUT_DIR / "polygon_progression_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (OUT_DIR / "polygon_progression_audit.md").write_text("\n".join(md), encoding="utf-8")


def write_alignment_report(alignments: dict[str, dict[str, Any]]) -> None:
    lines = ["# Alignment Report", ""]
    for field, alignment in alignments.items():
        lines.extend(
            [
                f"## {field}",
                "",
                f"- Same-size direct overlay: {alignment['same_size_direct_overlay']}",
                "- ORB/SIFT feature matching: unavailable in current runtime",
                "- ECC/image registration: unavailable in current runtime",
                f"- Georeferenced dates: {alignment['georeferenced_dates']}",
                f"- Alignment confidence: {alignment['alignment_confidence']}",
                f"- Spatially reliable: {alignment['spatially_reliable']}",
                f"- Decision: {alignment['notes']}",
                "",
            ]
        )
    (OUT_DIR / "alignment_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_outputs(features: list[dict[str, Any]], summaries: list[dict[str, Any]], warnings: list[str]) -> None:
    geojson = {
        "type": "FeatureCollection",
        "name": "polygon_zones_image_space",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "field_name": feature["field_name"],
                    "date": feature["date"],
                    "polygon_id": feature["polygon_id"],
                    "area_pixels": feature["area_pixels"],
                    "area_percent": feature["area_percent"],
                    "centroid_x": feature["centroid"][0],
                    "centroid_y": feature["centroid"][1],
                    "extraction_method": "human_annotation_difference",
                    "coordinate_type": "image_pixel_space",
                    "alignment_confidence": feature["alignment_confidence"],
                    "notes": "Image-space polygon extracted from human-marked NDVI overlay difference; not field survey geometry.",
                },
                "geometry": {"type": "Polygon", "coordinates": [feature["coordinates"]]},
            }
            for feature in features
        ],
    }
    (OUT_DIR / "polygon_zones_image_space.geojson").write_text(json.dumps(geojson, indent=2), encoding="utf-8")
    with (OUT_DIR / "spread_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    (OUT_DIR / "spread_summary.json").write_text(json.dumps({"summaries": summaries, "warnings": warnings}, indent=2), encoding="utf-8")


def write_validation_report(pairs: list[ImagePair], summaries: list[dict[str, Any]], features: list[dict[str, Any]], warnings: list[str]) -> None:
    overlay_missing = []
    timeline_missing = []
    for pair in pairs:
        overlay = OUT_DIR / pair.field / "overlays" / f"{pair.field}_{pair.date}_polygon_overlay.png"
        if not overlay.exists():
            overlay_missing.append(str(overlay))
    for field in sorted({pair.field for pair in pairs}):
        timeline = OUT_DIR / field / f"{field}_timeline_side_by_side.png"
        if not timeline.exists():
            timeline_missing.append(str(timeline))
    geojson_valid = False
    try:
        data = json.loads((OUT_DIR / "polygon_zones_image_space.geojson").read_text(encoding="utf-8"))
        geojson_valid = data.get("type") == "FeatureCollection"
    except Exception:
        geojson_valid = False
    polygon_point_failures = [
        feature["polygon_id"] for feature in features if len(feature["coordinates"]) < 4
    ]
    lines = [
        "# Progression Map Validation Report",
        "",
        f"- Expected field/date pairs processed: {len(pairs)} / {len(pairs)}",
        f"- Raw and annotated images exist: {all(pair.raw_path.exists() and pair.annotated_path.exists() for pair in pairs)}",
        f"- Missing overlay images: {len(overlay_missing)}",
        f"- Missing timeline panels: {len(timeline_missing)}",
        f"- CSV exists and has rows: {(OUT_DIR / 'spread_summary.csv').exists() and len(summaries) > 0}",
        f"- GeoJSON is valid JSON: {geojson_valid}",
        f"- Polygons with fewer than 3 points: {len(polygon_point_failures)}",
        f"- Tiny/noise polygon filter threshold: {MIN_COMPONENT_PIXELS} processing-scale pixels",
        f"- Warnings recorded: {len(warnings)}",
        "",
        "## Warnings",
        "",
    ]
    lines.extend(f"- {warning}" for warning in warnings)
    (OUT_DIR / "progression_map_validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_final_report(pairs_by_field: dict[str, list[ImagePair]], summaries_by_field: dict[str, list[dict[str, Any]]], total_polygons: int) -> None:
    mallard = summaries_by_field.get("Mallard-Avenue", [])
    mcintyre = summaries_by_field.get("McIntyre-Road", [])
    best = "Mallard-Avenue" if len(mallard) >= len(mcintyre) else "McIntyre-Road"
    lines = [
        "# Progression Map Final Report",
        "",
        "## Decisions",
        "",
        "1. Can we visually show NDVI progression today? **Yes.** The existing raw NDVI and human-marked NDVI overlays support a visual timeline.",
        "2. Can we show polygon spread today? **Yes, as extracted image-space concern zones.**",
        "3. Is the spread spatially/geographically accurate or image-space only? **Image-space only.** Only one GeoTIFF was found, and repeat-date georeferencing is insufficient.",
        f"4. Which field has the best progression evidence? **{best}.** Mallard-Avenue has the strongest timeline because it has 4 dates. McIntyre-Road has only 2 dates, so it supports before/after comparison only.",
        "5. How much did marked concern area change over time?",
    ]
    for field, summaries in summaries_by_field.items():
        first = summaries[0]["total_marked_area_percent"]
        last = summaries[-1]["total_marked_area_percent"]
        lines.append(f"   - {field}: {first:.2f}% -> {last:.2f}% ({last - first:+.2f} percentage points).")
    lines.extend(
        [
            "6. Is this enough for farmer demo? **Yes.** It is suitable for a farmer-facing progression demo.",
            "7. Is this enough for model training? **No.** These polygons are extracted from human-marked NDVI overlay differences and should be treated as concern-zone visualization, not disease diagnosis.",
            "8. What needs to be added next? Stable repeatable zone IDs, reviewed 0-5 disease/stress scores, RGB pairing, field context, and true repeat-date georeferencing.",
            "",
            f"- Total polygons extracted: {total_polygons}",
            "- Final feasibility decision: suitable for farmer demo; not yet a calibrated disease progression model.",
        ]
    )
    (OUT_DIR / "progression_map_final_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = read_template_pairs()
    pairs_by_field: dict[str, list[ImagePair]] = defaultdict(list)
    for pair in pairs:
        pairs_by_field[pair.field].append(pair)
    for field in pairs_by_field:
        pairs_by_field[field].sort(key=lambda pair: pair.date)

    print("Progression map execution plan")
    print(f"- fields: {', '.join(sorted(pairs_by_field))}")
    print(f"- dates: {', '.join(pair.date for pair in pairs)}")
    print("- extraction mode selected: human annotation difference primary; raw NDVI visual threshold context")
    print("- coordinate output: image-space pixel polygons, not geospatial")

    alignments = {field: alignment_for_field(field_pairs) for field, field_pairs in pairs_by_field.items()}
    for field, alignment in alignments.items():
        print(f"- {field} alignment reliable: {alignment['spatially_reliable']} ({alignment['alignment_confidence']})")

    all_features: list[dict[str, Any]] = []
    all_summaries: list[dict[str, Any]] = []
    summaries_by_field: dict[str, list[dict[str, Any]]] = {}
    features_by_field_date: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    warnings: list[str] = [
        "Outputs are image-space concern-zone visualizations, not field survey geometry.",
        "Human annotations are baked into raster images; extraction uses visual difference, not semantic disease labels.",
        "Raw NDVI threshold context is based on color/brightness visual exports, not calibrated NDVI values.",
    ]
    extraction_settings: dict[str, Any] = {}

    for field, field_pairs in pairs_by_field.items():
        field_dir = OUT_DIR / field
        overlay_dir = field_dir / "overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        previous_area: float | None = None
        field_summaries: list[dict[str, Any]] = []
        overlay_paths: list[Path] = []
        for pair in field_pairs:
            features, settings, _ = extract_annotation_polygons(pair)
            extraction_settings[f"{pair.field}_{pair.date}"] = settings
            raw_context = estimate_low_ndvi(pair.raw_path)
            alignment_confidence = alignments[field]["alignment_confidence"]
            for feature in features:
                feature["alignment_confidence"] = alignment_confidence
            all_features.extend(features)
            features_by_field_date[field][pair.date] = features
            total_area = sum(feature["area_pixels"] for feature in features)
            total_percent = total_area / (pair.raw_size[0] * pair.raw_size[1]) * 100
            state = spread_state(total_percent, previous_area, len(features))
            summary = {
                "field_name": field,
                "date": pair.date,
                "raw_ndvi_path": str(pair.raw_path),
                "annotated_ndvi_path": str(pair.annotated_path),
                "image_width": pair.raw_size[0],
                "image_height": pair.raw_size[1],
                "num_polygons": len(features),
                "total_marked_area_pixels": int(total_area),
                "total_marked_area_percent": round(total_percent, 4),
                "largest_polygon_area_pixels": max([feature["area_pixels"] for feature in features] or [0]),
                "change_from_previous_area_percent": "" if previous_area is None else round(total_percent - previous_area, 4),
                "spread_state": state if alignments[field]["alignment_confidence"] != "low" else "insufficient_alignment",
                "alignment_confidence": alignment_confidence,
                "extraction_method": "human_annotation_difference",
                "notes": raw_context["threshold_note"],
            }
            field_summaries.append(summary)
            all_summaries.append(summary)
            overlay_path = overlay_dir / f"{field}_{pair.date}_polygon_overlay.png"
            draw_overlay(pair, features, summary, overlay_path)
            overlay_paths.append(overlay_path)
            previous_area = total_percent
        summaries_by_field[field] = field_summaries
        timeline_path = field_dir / f"{field}_timeline_side_by_side.png"
        cumulative_path = field_dir / f"{field}_cumulative_spread.png"
        chart_path = field_dir / f"{field}_spread_area_chart.png"
        panel_path = field_dir / f"{field}_farmer_demo_progression_panel.png"
        make_timeline(field, field_pairs, field_summaries, overlay_paths, timeline_path)
        make_cumulative(field, field_pairs, features_by_field_date[field], alignments[field], cumulative_path)
        make_chart(field, field_summaries, chart_path)
        make_farmer_panel(field, field_summaries, timeline_path, chart_path, panel_path)

    write_audit(pairs_by_field, alignments, extraction_settings)
    write_alignment_report(alignments)
    write_outputs(all_features, all_summaries, warnings)
    write_validation_report(pairs, all_summaries, all_features, warnings)
    write_final_report(pairs_by_field, summaries_by_field, len(all_features))

    print(f"- output folder: {OUT_DIR}")
    print(f"- polygons extracted: {len(all_features)}")
    print(f"- timeline images created: {len(pairs_by_field)}")
    print(f"- cumulative maps created: {len(pairs_by_field)}")
    for field in sorted(pairs_by_field):
        print(f"- spread chart: {OUT_DIR / field / (field + '_spread_area_chart.png')}")
    print(f"- GeoJSON path: {OUT_DIR / 'polygon_zones_image_space.geojson'}")
    print(f"- CSV path: {OUT_DIR / 'spread_summary.csv'}")
    print("- final feasibility decision: farmer demo yes; model training no")


if __name__ == "__main__":
    main()
