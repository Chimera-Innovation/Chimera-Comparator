from __future__ import annotations

import csv
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DATASET_ROOT = Path(r"D:\NDVI-dataset")
MVP_DIR = DATASET_ROOT / "MVP - ndvi progression"
BAD_OUT_DIR = MVP_DIR / "progression_maps"
OUT_DIR = MVP_DIR / "human_annotation_progression"
TEMPLATE_CSV = MVP_DIR / "manual_annotation_template.csv"

MAX_PROCESS_DIM = 1800
MAX_PANEL_DIM = 620
DIFF_THRESHOLD = 42
MIN_COMPONENT_AREA = 260
MIN_REGION_WIDTH = 18
MIN_REGION_HEIGHT = 18
TEXT_LIKE_MAX_WIDTH = 135
TEXT_LIKE_MAX_HEIGHT = 55
MERGE_DISTANCE = 12


@dataclass
class ImagePair:
    field_name: str
    date: str
    raw_path: Path
    annotated_path: Path
    raw_rel: str
    annotated_rel: str
    raw_size: tuple[int, int]
    annotated_size: tuple[int, int]
    resized_for_comparison: bool


def font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


FONT = font(22)
SMALL = font(15)
TITLE = font(30)


def open_rgb(path: Path, max_dim: int | None = None) -> Image.Image:
    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(path).convert("RGB")
    if max_dim:
        image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    return image.copy()


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def read_pairs() -> list[ImagePair]:
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
                    field_name=row["field_name"],
                    date=row["date_observed"],
                    raw_path=raw_path,
                    annotated_path=annotated_path,
                    raw_rel=raw_rel,
                    annotated_rel=annotated_rel,
                    raw_size=raw_size,
                    annotated_size=annotated_size,
                    resized_for_comparison=raw_size != annotated_size,
                )
            )
    return sorted(pairs, key=lambda pair: (pair.field_name, pair.date))


def resize_for_processing(pair: ImagePair) -> tuple[Image.Image, Image.Image, float, float]:
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


def human_annotation_mask(raw: Image.Image, annotated: Image.Image) -> np.ndarray:
    raw_arr = np.asarray(raw).astype(np.int16)
    ann_arr = np.asarray(annotated).astype(np.int16)
    diff = np.abs(ann_arr - raw_arr).max(axis=2)
    mask = diff > DIFF_THRESHOLD
    return clean_mask(mask)


def clean_mask(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    for _ in range(1):
        mask = dilate(mask)
    for _ in range(1):
        mask = erode(mask)
    return mask


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
        pts: list[tuple[int, int]] = []
        min_x = max_x = start[0]
        min_y = max_y = start[1]
        while queue:
            x, y = queue.popleft()
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
                queue.append((nx, ny))
        found.append({"points": pts, "bbox": [min_x, min_y, max_x, max_y], "area": len(pts)})
    return found


def reject_component(comp: dict[str, Any], image_size: tuple[int, int]) -> bool:
    min_x, min_y, max_x, max_y = comp["bbox"]
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    image_w, image_h = image_size
    area = comp["area"]
    if area < MIN_COMPONENT_AREA:
        return True
    if width < MIN_REGION_WIDTH or height < MIN_REGION_HEIGHT:
        return True
    if width <= TEXT_LIKE_MAX_WIDTH and height <= TEXT_LIKE_MAX_HEIGHT and area < 900:
        return True
    touches_border = min_x <= 2 or min_y <= 2 or max_x >= image_w - 3 or max_y >= image_h - 3
    if touches_border and area > 5000:
        return True
    if touches_border and (width > image_w * 0.85 or height > image_h * 0.85):
        return True
    aspect = width / max(1, height)
    if aspect > 45 or aspect < 1 / 45:
        return True
    return False


def close_enough(a: list[int], b: list[int], distance: int) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 + distance < bx1 or bx2 + distance < ax1 or ay2 + distance < by1 or by2 + distance < ay1)


def merge_components(comps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = comps[:]
    changed = True
    while changed:
        changed = False
        result: list[dict[str, Any]] = []
        used = [False] * len(merged)
        for i, comp in enumerate(merged):
            if used[i]:
                continue
            box = comp["bbox"][:]
            area = comp["area"]
            used[i] = True
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                if close_enough(box, merged[j]["bbox"], MERGE_DISTANCE):
                    other = merged[j]["bbox"]
                    box = [min(box[0], other[0]), min(box[1], other[1]), max(box[2], other[2]), max(box[3], other[3])]
                    area += merged[j]["area"]
                    used[j] = True
                    changed = True
            result.append({"bbox": box, "area": area})
        merged = result
    return merged


def reject_merged_region(comp: dict[str, Any], image_size: tuple[int, int]) -> bool:
    x1, y1, x2, y2 = comp["bbox"]
    width = x2 - x1 + 1
    height = y2 - y1 + 1
    image_w, image_h = image_size
    touches_border = x1 <= 3 or y1 <= 3 or x2 >= image_w - 4 or y2 >= image_h - 4
    if width > image_w * 0.92 or height > image_h * 0.92:
        return True
    if touches_border and (width > image_w * 0.65 or height > image_h * 0.65):
        return True
    if comp["area"] < MIN_COMPONENT_AREA:
        return True
    return False


def polygon_from_box(box: list[int], sx: float, sy: float) -> list[list[float]]:
    x1, y1, x2, y2 = box
    coords = [
        [round(x1 * sx, 2), round(y1 * sy, 2)],
        [round(x2 * sx, 2), round(y1 * sy, 2)],
        [round(x2 * sx, 2), round(y2 * sy, 2)],
        [round(x1 * sx, 2), round(y2 * sy, 2)],
        [round(x1 * sx, 2), round(y1 * sy, 2)],
    ]
    return coords


def extract_human_regions(pair: ImagePair) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_small, ann_small, sx, sy = resize_for_processing(pair)
    mask = human_annotation_mask(raw_small, ann_small)
    image_size = raw_small.size
    kept = [comp for comp in components(mask) if not reject_component(comp, image_size)]
    merged = [comp for comp in merge_components(kept) if not reject_merged_region(comp, image_size)]
    features = []
    original_w, original_h = pair.raw_size
    for idx, comp in enumerate(merged, start=1):
        coords = polygon_from_box(comp["bbox"], sx, sy)
        x1, y1, x2, y2 = comp["bbox"]
        # Use filled region area after merging because human boxes/freehand regions denote zones, not line pixels.
        area_pixels = int(round((x2 - x1 + 1) * sx * (y2 - y1 + 1) * sy))
        centroid_x = round(((x1 + x2) / 2) * sx, 2)
        centroid_y = round(((y1 + y2) / 2) * sy, 2)
        features.append(
            {
                "field_name": pair.field_name,
                "date": pair.date,
                "polygon_id": f"{pair.field_name}_{pair.date}_H{idx:02d}",
                "label_type": "human_marked_perceived_vigour_loss",
                "source": "human_annotation_difference",
                "coordinate_type": "image_pixel_space",
                "area_pixels": area_pixels,
                "area_percent": round(area_pixels / (original_w * original_h) * 100, 4),
                "bbox": [round(x1 * sx, 2), round(y1 * sy, 2), round(x2 * sx, 2), round(y2 * sy, 2)],
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
                "confidence": "human_annotation_extracted",
                "notes": "Extracted only from raw-vs-human-annotated image differences. Raw NDVI was not thresholded.",
                "coordinates": coords,
            }
        )
    settings = {
        "diff_threshold": DIFF_THRESHOLD,
        "min_component_area_processing_pixels": MIN_COMPONENT_AREA,
        "merge_distance_processing_pixels": MERGE_DISTANCE,
        "processing_image_size": raw_small.size,
        "raw_size": pair.raw_size,
        "annotated_size": pair.annotated_size,
        "resized_annotated_to_raw_for_comparison": pair.resized_for_comparison,
        "raw_ndvi_thresholding_used": False,
    }
    return features, settings


def draw_overlay(pair: ImagePair, features: list[dict[str, Any]], output: Path) -> None:
    base = open_rgb(pair.raw_path, MAX_PANEL_DIM)
    draw = ImageDraw.Draw(base)
    sx = base.size[0] / pair.raw_size[0]
    sy = base.size[1] / pair.raw_size[1]
    for feature in features:
        coords = [(x * sx, y * sy) for x, y in feature["coordinates"]]
        draw.line(coords, fill=(255, 0, 0), width=5)
    area = sum(feature["area_pixels"] for feature in features) / (pair.raw_size[0] * pair.raw_size[1]) * 100
    draw.rectangle((0, 0, base.size[0], 62), fill=(0, 0, 0))
    draw.text((10, 8), f"{pair.field_name} {pair.date}", fill=(255, 255, 255), font=SMALL)
    draw.text((10, 34), f"Human-marked perceived vigour-loss zone | {area:.2f}% area", fill=(255, 235, 180), font=SMALL)
    output.parent.mkdir(parents=True, exist_ok=True)
    base.save(output)


def caption_panel(path: Path, label: str, area: float) -> Image.Image:
    img = open_rgb(path, 520)
    panel = Image.new("RGB", (img.size[0], img.size[1] + 72), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((8, 8), label, fill=(20, 20, 20), font=SMALL)
    draw.text((8, 32), f"human-marked area {area:.2f}%", fill=(120, 20, 20), font=SMALL)
    panel.paste(img, (0, 72))
    return panel


def make_timeline(field: str, summaries: list[dict[str, Any]], overlay_paths: list[Path], output: Path) -> None:
    panels = [caption_panel(path, summary["date"], summary["human_marked_area_percent"]) for summary, path in zip(summaries, overlay_paths)]
    width = sum(panel.size[0] for panel in panels)
    height = max(panel.size[1] for panel in panels) + 76
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 18), f"{field} Human Annotation Progression", fill=(20, 20, 20), font=TITLE)
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 76))
        x += panel.size[0]
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def interpretation(current: float, previous: float | None, weak_alignment: bool) -> str:
    if weak_alignment:
        return "visual comparison only"
    if previous is None:
        return "baseline human-marked concern area"
    delta = current - previous
    if abs(delta) < 0.25:
        return "human-marked concern area roughly stable"
    if delta > 0:
        return "human-marked concern area increased"
    return "human-marked concern area decreased"


def write_wrong_output_audit() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    old_summary = BAD_OUT_DIR / "spread_summary.csv"
    rows = []
    if old_summary.exists():
        with old_summary.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    old_polygons = sum(int(row.get("num_polygons", 0)) for row in rows)
    max_per_date = max([int(row.get("num_polygons", 0)) for row in rows] or [0])
    text = f"""# Wrong Output Audit

## Current Bad Output

- Folder audited: `{BAD_OUT_DIR}`
- Script identified: `build_progression_maps.py`
- Previous extraction label: `human_annotation_difference`, but the script also carried raw NDVI visual threshold context.
- Previous output polygon count: {old_polygons}
- Maximum polygons in one date: {max_per_date}

## Why It Is Not Acceptable

The current maps are not acceptable for the MVP if they detect raw NDVI color changes as concern zones or if they create many boxes from NDVI palette variation, labels, borders, and small artifacts.

Human annotation must be treated as the only valid concern-zone source. Raw NDVI may be used only as the background image.

## Corrected Principle

- Do not infer new concern zones from raw NDVI.
- Do not threshold raw NDVI to invent stress areas.
- Do not label red/orange/green/yellow NDVI palette regions as concern.
- Extract only pixels changed by the human annotated overlay.
- Label outputs as `human_marked_perceived_vigour_loss`, not disease and not automatic diagnosis.
"""
    (OUT_DIR / "wrong_output_audit.md").write_text(text, encoding="utf-8")


def write_geojson(features: list[dict[str, Any]]) -> None:
    data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {k: v for k, v in feature.items() if k != "coordinates"},
                "geometry": {"type": "Polygon", "coordinates": [feature["coordinates"]]},
            }
            for feature in features
        ],
    }
    (OUT_DIR / "human_marked_vigour_loss_polygons.geojson").write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_summary_csv(summaries: list[dict[str, Any]]) -> None:
    path = OUT_DIR / "human_annotation_spread_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)


def write_row_level_outputs(features: list[dict[str, Any]]) -> None:
    report = """# Row-Level Feasibility

- Are the human-marked zones aligned with crop rows? **Partially visible, but not consistently enough to automate from this dataset alone.**
- Can we manually assign each human-marked zone to one or more rows? **Yes.** Manual assignment is feasible after reviewing each clean human-marked polygon.
- Are row IDs stable across dates? **No stable row IDs were found in the current audit/template.**
- Is automatic row detection reliable? **No.** The current MVP should not make row-by-row progression claims yet.
- Should row-level work be manual first? **Yes.** Create stable row IDs and manually assign human-marked zones before any automated row progression.

Decision: do not generate row-by-row claims yet. Use the manual row-zone assignment template first.
"""
    (OUT_DIR / "row_level_feasibility.md").write_text(report, encoding="utf-8")
    template_path = OUT_DIR / "manual_row_zone_assignment_template.csv"
    columns = [
        "field_name",
        "date",
        "polygon_id",
        "zone_id",
        "human_marked_area_percent",
        "primary_row_id",
        "rows_touched",
        "row_span",
        "merge_with_adjacent_zone",
        "merged_zone_id",
        "human_note",
        "confidence",
        "reviewed_by",
    ]
    with template_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for feature in features:
            writer.writerow(
                {
                    "field_name": feature["field_name"],
                    "date": feature["date"],
                    "polygon_id": feature["polygon_id"],
                    "zone_id": "",
                    "human_marked_area_percent": feature["area_percent"],
                    "primary_row_id": "",
                    "rows_touched": "",
                    "row_span": "",
                    "merge_with_adjacent_zone": "",
                    "merged_zone_id": "",
                    "human_note": "",
                    "confidence": "",
                    "reviewed_by": "",
                }
            )


def write_final_report(total_polygons: int, fields: list[str], dates: list[str]) -> None:
    text = f"""# Human Annotation Progression Final Report

1. The corrected method follows human annotation only.
2. Raw NDVI is used only as the background image.
3. Concern polygons are extracted from differences between raw and human-annotated images.
4. No raw NDVI thresholding is used.
5. These are perceived vigour-loss zones, not calibrated disease diagnosis.
6. This is suitable for farmer visual discussion.
7. This is not yet suitable for model training until zones are reviewed and manually confirmed.
8. Row-by-row progression requires manual row-zone assignment or stable row detection.

## Summary

- Fields processed: {", ".join(fields)}
- Dates processed: {", ".join(dates)}
- Human annotation polygons extracted: {total_polygons}
- Coordinate type: image pixel space

Final decision: Corrected MVP now follows human annotation only. Raw NDVI is background context, not the source of detected concern polygons.
"""
    (OUT_DIR / "human_annotation_progression_final_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_wrong_output_audit()
    pairs = read_pairs()
    by_field: dict[str, list[ImagePair]] = defaultdict(list)
    for pair in pairs:
        by_field[pair.field_name].append(pair)
    for field in by_field:
        by_field[field].sort(key=lambda pair: pair.date)

    all_features: list[dict[str, Any]] = []
    all_summaries: list[dict[str, Any]] = []
    timeline_count = 0
    dates_processed: list[str] = []
    for field, field_pairs in sorted(by_field.items()):
        previous: float | None = None
        weak_alignment = len({pair.raw_size for pair in field_pairs}) > 1
        overlay_paths: list[Path] = []
        field_summaries: list[dict[str, Any]] = []
        for pair in field_pairs:
            features, settings = extract_human_regions(pair)
            all_features.extend(features)
            total_area = sum(feature["area_pixels"] for feature in features)
            area_percent = round(total_area / (pair.raw_size[0] * pair.raw_size[1]) * 100, 4)
            summary = {
                "field_name": field,
                "date": pair.date,
                "raw_ndvi_path": str(pair.raw_path),
                "annotated_ndvi_path": str(pair.annotated_path),
                "image_width": pair.raw_size[0],
                "image_height": pair.raw_size[1],
                "human_marked_area_pixels": total_area,
                "human_marked_area_percent": area_percent,
                "num_human_marked_regions": len(features),
                "change_from_previous_percent": "" if previous is None else round(area_percent - previous, 4),
                "progression_interpretation": interpretation(area_percent, previous, weak_alignment),
                "notes": json.dumps(settings),
            }
            field_summaries.append(summary)
            all_summaries.append(summary)
            overlay_path = OUT_DIR / field / "overlays" / f"{field}_{pair.date}_human_annotation_only_overlay.png"
            draw_overlay(pair, features, overlay_path)
            overlay_paths.append(overlay_path)
            previous = area_percent
            dates_processed.append(pair.date)
        timeline_path = OUT_DIR / field / f"{field}_human_annotation_timeline.png"
        make_timeline(field, field_summaries, overlay_paths, timeline_path)
        timeline_count += 1

    write_summary_csv(all_summaries)
    write_geojson(all_features)
    write_row_level_outputs(all_features)
    write_final_report(len(all_features), sorted(by_field), sorted(dates_processed))

    print(f"- corrected output folder: {OUT_DIR}")
    print(f"- fields processed: {', '.join(sorted(by_field))}")
    print(f"- dates processed: {', '.join(sorted(dates_processed))}")
    print(f"- number of human annotation polygons extracted: {len(all_features)}")
    print(f"- timeline images created: {timeline_count}")
    print(f"- CSV path: {OUT_DIR / 'human_annotation_spread_summary.csv'}")
    print(f"- GeoJSON path: {OUT_DIR / 'human_marked_vigour_loss_polygons.geojson'}")
    print(f"- row assignment template path: {OUT_DIR / 'manual_row_zone_assignment_template.csv'}")
    print("- final decision: Corrected MVP now follows human annotation only. Raw NDVI is background context, not the source of detected concern polygons.")


if __name__ == "__main__":
    main()
