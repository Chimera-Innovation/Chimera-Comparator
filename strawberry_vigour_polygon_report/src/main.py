from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

import cv2
import numpy as np

from .annotation_parser import (
    get_annotation_mask,
    human_annotation_colour_masks,
    rendered_green_and_ndvi_proxy,
    resize_annotated_to_raw,
)
from .growth_stage import growth_stage_for_date
from .human_guided_polygons import HumanGuidedConfig, build_human_guided_polygons
from .image_matching import match_image_pairs_with_audit
from .polygon_export import date_slug, field_slug, read_geojson_records, write_bed_summary, write_geojson, write_polygon_summary, write_row_summary
from .report_generator import (
    build_audit_rows,
    draw_polygon_overlay,
    make_chart,
    top_priority_beds,
    write_customer_release_audit,
    write_html_report,
    write_inspection_targets,
)
from .row_detection import RowRegion, detect_rows, write_row_lines_geojson


SMALL_ANNOTATION_PERCENT = 0.03
LARGE_ANNOTATION_PERCENT = 35.0
UNCLASSIFIED_CHANGED_PERCENT = 10.0

SEGMENTATION_COLORS_BGR = {
    "low_vigour": (35, 35, 235),
    "medium_vigour": (215, 90, 190),
    "high_vigour": (80, 235, 120),
}

SEGMENTATION_LABELS = {
    "low_vigour": "LOW VIGOUR",
    "medium_vigour": "MEDIUM VIGOUR",
    "high_vigour": "HIGH VIGOUR",
}


def outputs_are_current(output_paths: list[Path], source_paths: list[Path]) -> bool:
    if not output_paths or not all(path.exists() for path in output_paths):
        return False
    newest_source = max(path.stat().st_mtime for path in source_paths if path.exists())
    oldest_output = min(path.stat().st_mtime for path in output_paths)
    return oldest_output >= newest_source


def field_foreground_mask(raw_bgr) -> object:
    gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
    return (gray > 8) & (gray < 248)


def resize_pair_for_processing(raw_bgr, annotated_bgr, max_dimension: int) -> tuple[object, object, float]:
    if max_dimension <= 0:
        return raw_bgr, annotated_bgr, 1.0
    height, width = raw_bgr.shape[:2]
    largest = max(height, width)
    if largest <= max_dimension:
        return raw_bgr, annotated_bgr, 1.0
    scale = max_dimension / largest
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    raw_resized = cv2.resize(raw_bgr, new_size, interpolation=cv2.INTER_AREA)
    annotated_resized = cv2.resize(annotated_bgr, new_size, interpolation=cv2.INTER_AREA)
    return raw_resized, annotated_resized, scale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build human-annotated strawberry vigour polygon report.")
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--annotated-dir", required=True, type=Path)
    parser.add_argument("--field-name", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bed-aware", default="true", choices=("true", "false"), help="Deprecated alias; row-aware output is used for customer reports.")
    parser.add_argument("--bed-numbering", default="top_to_bottom", choices=("bottom_to_top", "top_to_bottom"), help="Deprecated alias. Customer row numbering is top_to_bottom.")
    parser.add_argument("--bed-count", default=None, type=int, help="Deprecated alias for --row-count.")
    parser.add_argument("--row-numbering", default="top_to_bottom", choices=("top_to_bottom",))
    parser.add_argument("--row-count", default=None, type=int)
    parser.add_argument("--human-buffer-pixels", default=50, type=int)
    parser.add_argument("--min-inspection-area-pixels", default=900, type=int)
    parser.add_argument("--max-component-area-pixels", default=None, type=int)
    parser.add_argument("--max-percent-field-area", default=25.0, type=float)
    parser.add_argument("--max-component-width-ratio", default=0.55, type=float)
    parser.add_argument("--min-human-overlap-ratio", default=0.70, type=float)
    parser.add_argument("--min-buffer-overlap-ratio", default=0.95, type=float)
    parser.add_argument("--field-area-m2", default=None, type=float)
    parser.add_argument("--max-processing-dimension", default=3600, type=int)
    parser.add_argument("--canopy-row-width-ratio", default=0.45, type=float)
    parser.add_argument("--min-canopy-row-width-pixels", default=6, type=int)
    parser.add_argument("--row-gap-length-width-ratio", default=20.0, type=float)
    parser.add_argument("--row-gap-width-fraction", default=0.5, type=float)
    parser.add_argument("--row-gap-rejection-score-threshold", default=0.55, type=float)
    parser.add_argument("--inter-row-gap-width-ratio", default=0.35, type=float)
    parser.add_argument("--inspection-region-merge-x-multiplier", default=8.0, type=float)
    parser.add_argument("--inspection-region-merge-y-multiplier", default=4.0, type=float)
    parser.add_argument("--max-inspection-regions", default=20, type=int)
    parser.add_argument("--date", action="append", default=None, help="Optional YYYY-MM-DD date filter. Can be supplied more than once.")
    parser.add_argument("--use-cache", default="true", choices=("true", "false"), help="Reuse current per-date outputs when source images have not changed.")
    parser.add_argument("--force-reprocess", action="store_true", help="Ignore cached per-date outputs and rebuild all selected dates.")
    parser.add_argument("--ground-notes-csv", default=None, type=Path)
    parser.add_argument("--event-log-csv", default=None, type=Path)
    return parser.parse_args()


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "polygons": output_dir / "polygons",
        "summaries": output_dir / "summaries",
        "overlays": output_dir / "overlays",
        "reports": output_dir / "reports",
        "debug": output_dir / "debug",
        "segmentation": output_dir / "segmentation",
    }
    return paths


def clean_segmentation_mask(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype("uint8") * 255


def combined_segmentation_image(class_masks: dict[str, np.ndarray]) -> np.ndarray:
    first = next(iter(class_masks.values()))
    combined = np.zeros((first.shape[0], first.shape[1], 3), dtype=np.uint8)
    for class_name in ("high_vigour", "medium_vigour", "low_vigour"):
        mask = class_masks.get(class_name)
        if mask is None:
            continue
        combined[mask > 0] = SEGMENTATION_COLORS_BGR[class_name]
    return combined


def transparent_overlay(raw_bgr: np.ndarray, layer_bgr: np.ndarray, alpha: float) -> np.ndarray:
    overlay = raw_bgr.copy()
    active = np.any(layer_bgr > 0, axis=2)
    blended = cv2.addWeighted(raw_bgr, 1.0 - alpha, layer_bgr, alpha, 0)
    overlay[active] = blended[active]
    return overlay


def segmentation_overlay(raw_bgr: np.ndarray, class_masks: dict[str, np.ndarray]) -> np.ndarray:
    return transparent_overlay(raw_bgr, combined_segmentation_image(class_masks), 0.65)


def repair_annotation_outlines(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8) * 255
    if not np.any(binary):
        return binary
    height, width = binary.shape
    thicken = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    repaired = cv2.dilate(binary, thicken, iterations=1)
    horizontal_width = max(21, int(round(width * 0.018)) | 1)
    horizontal_height = max(5, int(round(height * 0.004)) | 1)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_width, horizontal_height))
    repaired = cv2.morphologyEx(repaired, cv2.MORPH_CLOSE, horizontal_kernel)
    local_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    repaired = cv2.morphologyEx(repaired, cv2.MORPH_CLOSE, local_kernel)
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats((repaired > 0).astype(np.uint8), connectivity=8)
    cleaned = np.zeros_like(repaired)
    min_area = max(16, int(round(height * width * 0.000006)))
    for label_id in range(1, component_count):
        if int(stats[label_id, cv2.CC_STAT_AREA]) >= min_area:
            cleaned[labels == label_id] = 255
    return (cleaned > 0).astype(np.uint8) * 255


def fill_repaired_regions(class_name: str, repaired_mask: np.ndarray, original_outline_mask: np.ndarray) -> np.ndarray:
    if not np.any(repaired_mask):
        return repaired_mask
    height, width = repaired_mask.shape
    contours, _hierarchy = cv2.findContours((repaired_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(repaired_mask)
    image_area = float(max(1, height * width))
    min_area = max(24.0, image_area * 0.000002)
    max_area_by_class = {
        "low_vigour": image_area * 0.14,
        "medium_vigour": image_area * 0.22,
        "high_vigour": image_area * 0.30,
    }
    max_area = max_area_by_class[class_name]
    border = max(4, int(round(min(height, width) * 0.006)))
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        touches_border = x <= border or y <= border or x + w >= width - border or y + h >= height - border
        border_area_limit = {
            "low_vigour": image_area * 0.025,
            "medium_vigour": image_area * 0.045,
            "high_vigour": image_area * 0.30,
        }[class_name]
        if touches_border and area > border_area_limit:
            continue
        candidate = np.zeros_like(repaired_mask)
        cv2.drawContours(candidate, [contour], -1, 255, thickness=cv2.FILLED)
        if np.any((candidate > 0) & (original_outline_mask > 0)):
            filled = cv2.bitwise_or(filled, candidate)
    return filled


def filled_region_masks(class_masks: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    repaired_masks: dict[str, np.ndarray] = {}
    filled_masks: dict[str, np.ndarray] = {}
    for class_name, outline_mask in class_masks.items():
        repaired = repair_annotation_outlines(outline_mask)
        repaired_masks[class_name] = repaired
        filled_masks[class_name] = fill_repaired_regions(class_name, repaired, outline_mask)
    return repaired_masks, filled_masks


def ndvi_feature_stack(raw_bgr: np.ndarray) -> np.ndarray:
    green_index, visual_ndvi_proxy = rendered_green_and_ndvi_proxy(raw_bgr)
    return np.dstack([green_index, visual_ndvi_proxy]).astype(np.float32)


def smooth_ndvi_features(features: np.ndarray, canopy_mask: np.ndarray) -> np.ndarray:
    smoothed = np.zeros_like(features)
    canopy = canopy_mask > 0
    for channel in range(features.shape[2]):
        work = features[:, :, channel]
        blurred = cv2.GaussianBlur(work, (17, 17), 0)
        smoothed[:, :, channel] = np.where(canopy, blurred, work)
    return smoothed


def canopy_mask_from_raw(raw_bgr: np.ndarray) -> np.ndarray:
    field_mask = field_foreground_mask(raw_bgr)
    hsv = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    green_index, visual_ndvi_proxy = rendered_green_and_ndvi_proxy(raw_bgr)
    valid = field_mask & (saturation > 20) & (value > 25)
    if not np.any(valid):
        return field_mask.astype(np.uint8) * 255
    green_floor = float(np.percentile(green_index[valid], 5))
    ndvi_floor = float(np.percentile(visual_ndvi_proxy[valid], 5))
    canopy = valid & ((green_index >= green_floor) | (visual_ndvi_proxy >= ndvi_floor))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    canopy_u8 = canopy.astype(np.uint8) * 255
    canopy_u8 = cv2.morphologyEx(canopy_u8, cv2.MORPH_OPEN, kernel)
    canopy_u8 = cv2.morphologyEx(canopy_u8, cv2.MORPH_CLOSE, kernel)
    return canopy_u8


def seed_polygon_masks(class_masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    _repaired_masks, seeds = filled_region_masks(class_masks)
    return seeds


def class_statistics_from_seeds(
    features: np.ndarray,
    seed_masks: dict[str, np.ndarray],
    canopy_mask: np.ndarray,
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    stats: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    canopy = canopy_mask > 0
    for class_name in ("low_vigour", "medium_vigour", "high_vigour"):
        active = (seed_masks[class_name] > 0) & canopy
        values = features[active]
        if values.size == 0:
            mean = np.array([0.0, 0.0], dtype=np.float32)
            median_values = np.array([0.0, 0.0], dtype=np.float32)
            std = np.array([1.0, 1.0], dtype=np.float32)
        else:
            mean = values.mean(axis=0)
            median_values = np.median(values, axis=0)
            std = values.std(axis=0)
            std = np.maximum(std, np.array([0.025, 0.025], dtype=np.float32))
        stats[class_name] = {
            "mean": mean,
            "median": median_values,
            "std": std,
            "seed_pixels": int(values.shape[0]),
        }
        rows.append(
            {
                "class_name": class_name,
                "display_label": SEGMENTATION_LABELS[class_name],
                "seed_pixels": int(values.shape[0]),
                "green_index_mean": round(float(mean[0]), 6),
                "green_index_median": round(float(median_values[0]), 6),
                "green_index_std": round(float(std[0]), 6),
                "visual_ndvi_proxy_mean": round(float(mean[1]), 6),
                "visual_ndvi_proxy_median": round(float(median_values[1]), 6),
                "visual_ndvi_proxy_std": round(float(std[1]), 6),
            }
        )
    return stats, rows


def classify_canopy_by_ndvi(
    features: np.ndarray,
    canopy_mask: np.ndarray,
    stats: dict[str, dict[str, object]],
) -> np.ndarray:
    labels = np.zeros(features.shape[:2], dtype=np.uint8)
    canopy = canopy_mask > 0
    if not np.any(canopy):
        return labels
    centers = {
        "low_vigour": float(np.asarray(stats["low_vigour"]["mean"], dtype=np.float32)[1]),
        "medium_vigour": float(np.asarray(stats["medium_vigour"]["mean"], dtype=np.float32)[1]),
        "high_vigour": float(np.asarray(stats["high_vigour"]["mean"], dtype=np.float32)[1]),
    }
    ndvi = features[:, :, 1]
    distances = []
    ordered = ("low_vigour", "medium_vigour", "high_vigour")
    for class_name in ordered:
        distances.append(np.abs(ndvi - centers[class_name]))
    stack = np.dstack(distances)
    class_ids = np.argmin(stack, axis=2).astype(np.uint8) + 1
    labels[canopy] = class_ids[canopy]
    return labels


def smooth_classification(labels: np.ndarray, canopy_mask: np.ndarray) -> np.ndarray:
    canopy = canopy_mask > 0
    smoothed = cv2.medianBlur(labels, 17)
    smoothed[~canopy] = 0
    cleaned = np.zeros_like(smoothed)
    min_component_area = max(32, int(round(labels.size * 0.00002)))
    for class_id in (1, 2, 3):
        class_mask = np.where(smoothed == class_id, 255, 0).astype(np.uint8)
        count, component_labels, component_stats, _centroids = cv2.connectedComponentsWithStats((class_mask > 0).astype(np.uint8), 8)
        for label_id in range(1, count):
            if int(component_stats[label_id, cv2.CC_STAT_AREA]) >= min_component_area:
                cleaned[component_labels == label_id] = class_id
    cleaned[~canopy] = 0
    return cleaned


def labels_to_class_masks(labels: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "low_vigour": np.where(labels == 1, 255, 0).astype(np.uint8),
        "medium_vigour": np.where(labels == 2, 255, 0).astype(np.uint8),
        "high_vigour": np.where(labels == 3, 255, 0).astype(np.uint8),
    }


def panel_image(image: np.ndarray, title: str, size: tuple[int, int] = (520, 420)) -> np.ndarray:
    target_w, target_h = size
    title_h = 40
    canvas = np.full((target_h + title_h, target_w, 3), 255, dtype=np.uint8)
    h, w = image.shape[:2]
    scale = min(target_w / max(1, w), target_h / max(1, h))
    resized = cv2.resize(image, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    y = title_h + (target_h - resized.shape[0]) // 2
    x = (target_w - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    cv2.putText(canvas, title, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (30, 30, 30), 2, cv2.LINE_AA)
    return canvas


def write_visual_proof_contact_sheet(
    human_layer: np.ndarray,
    seed_layer: np.ndarray,
    ndvi_classification_layer: np.ndarray,
    final_region_map: np.ndarray,
    output_path: Path,
) -> None:
    top = np.hstack(
        [
            panel_image(human_layer, "A. Human annotations"),
            panel_image(seed_layer, "B. Seed polygons"),
        ]
    )
    bottom = np.hstack(
        [
            panel_image(ndvi_classification_layer, "C. NDVI classification"),
            panel_image(final_region_map, "D. Final vigour region map"),
        ]
    )
    contact_sheet = np.vstack([top, bottom])
    cv2.imwrite(str(output_path), contact_sheet)


def write_human_marked_contact_sheet(
    raw_bgr: np.ndarray,
    outline_layer: np.ndarray,
    filled_layer: np.ndarray,
    overlay: np.ndarray,
    output_path: Path,
) -> None:
    top = np.hstack(
        [
            panel_image(raw_bgr, "Raw NDVI/background"),
            panel_image(outline_layer, "Human annotation outlines"),
        ]
    )
    bottom = np.hstack(
        [
            panel_image(filled_layer, "Human-marked vigour regions"),
            panel_image(overlay, "Human marks over raw image"),
        ]
    )
    cv2.imwrite(str(output_path), np.vstack([top, bottom]))


def remove_stale_classifier_outputs(output_dir: Path) -> None:
    stale_names = [
        "canopy_mask.png",
        "ndvi_classification.png",
        "final_vigour_region_map.png",
        "final_vigour_region_overlay.png",
        "vigour_segmentation_overlay.png",
        "seed_polygons.png",
        "class_ndvi_statistics.csv",
        "mission_debrief.html",
    ]
    for name in stale_names:
        path = output_dir / name
        if path.exists():
            path.unlink()


def write_segmentation_outputs(
    output_dir: Path,
    date: str,
    field_name: str,
    raw_bgr: np.ndarray,
    annotated_bgr: np.ndarray,
    class_masks: dict[str, np.ndarray],
    processing_scale: float,
    original_width: int,
    original_height: int,
    processed_width: int,
    processed_height: int,
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    remove_stale_classifier_outputs(output_dir)
    summary_rows: list[dict[str, object]] = []
    total_pixels = float(max(1, raw_bgr.shape[0] * raw_bgr.shape[1]))
    _repaired_masks, filled_masks = filled_region_masks(class_masks)

    class_paths = {
        "low_vigour": output_dir / "low_vigour_marks.png",
        "medium_vigour": output_dir / "medium_vigour_marks.png",
        "high_vigour": output_dir / "high_vigour_marks.png",
    }
    filled_class_paths = {
        "low_vigour": output_dir / "low_vigour_human_marked_regions.png",
        "medium_vigour": output_dir / "medium_vigour_human_marked_regions.png",
        "high_vigour": output_dir / "high_vigour_human_marked_regions.png",
    }
    for class_name, path in class_paths.items():
        mask = class_masks[class_name]
        cv2.imwrite(str(path), mask)
        filled_mask = filled_masks[class_name]
        cv2.imwrite(str(filled_class_paths[class_name]), filled_mask)
        pixels = int(np.count_nonzero(filled_mask))
        summary_rows.append(
            {
                "field_name": field_name,
                "date": date,
                "class_name": class_name,
                "display_label": SEGMENTATION_LABELS[class_name],
                "area_pixels": pixels,
                "area_percent": round(pixels / total_pixels * 100.0, 6),
                "mask_path": str(filled_class_paths[class_name]),
                "processing_scale": processing_scale,
                "original_width": original_width,
                "original_height": original_height,
                "processed_width": processed_width,
                "processed_height": processed_height,
                "coordinate_space": "processed_image_pixels",
            }
        )

    outline_layer = combined_segmentation_image(class_masks)
    filled_layer = combined_segmentation_image(filled_masks)
    filled_overlay = transparent_overlay(raw_bgr, filled_layer, 0.45)
    outline_overlay = transparent_overlay(raw_bgr, outline_layer, 0.70)

    cv2.imwrite(str(output_dir / "annotation_outline_layer.png"), outline_layer)
    cv2.imwrite(str(output_dir / "human_marked_vigour_regions_filled.png"), filled_layer)
    cv2.imwrite(str(output_dir / "human_marked_vigour_regions_overlay.png"), filled_overlay)
    cv2.imwrite(str(output_dir / "human_marked_annotation_outline_overlay.png"), outline_overlay)
    write_human_marked_contact_sheet(
        raw_bgr,
        outline_layer,
        filled_layer,
        filled_overlay,
        output_dir / "visual_proof_contact_sheet.png",
    )
    source_audit = Path("polygon_semantics_audit.png")
    if source_audit.exists():
        audit_image = cv2.imread(str(source_audit), cv2.IMREAD_COLOR)
        if audit_image is not None:
            cv2.imwrite(str(output_dir / "polygon_semantics_audit.png"), audit_image)
    return summary_rows


def write_class_statistics(rows: list[dict[str, object]], output_path: Path, field_name: str, date: str) -> None:
    fieldnames = [
        "field_name",
        "date",
        "class_name",
        "display_label",
        "seed_pixels",
        "green_index_mean",
        "green_index_median",
        "green_index_std",
        "visual_ndvi_proxy_mean",
        "visual_ndvi_proxy_median",
        "visual_ndvi_proxy_std",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {"field_name": field_name, "date": date}
            out.update(row)
            writer.writerow(out)


def write_class_area_summary(rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "field_name",
        "date",
        "class_name",
        "display_label",
        "area_pixels",
        "area_percent",
        "mask_path",
        "processing_scale",
        "original_width",
        "original_height",
        "processed_width",
        "processed_height",
        "coordinate_space",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[WRITE] Class area summary: {output_path}")


def write_segmentation_html(output_dir: Path, field_name: str, latest_date: str | None) -> None:
    title = f"{field_name} &mdash; Human-Marked Vigour Regions"
    date_text = latest_date or "No date"
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #081018; color: #f5f7fb; }}
    header {{ padding: 24px 28px; background: #0c1722; border-bottom: 1px solid #243447; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    .sub {{ color: #a9b5c4; }}
    main {{ padding: 22px; display: grid; gap: 22px; }}
    .hero {{ background: #101b27; border: 1px solid #243447; padding: 16px; }}
    img {{ max-width: 100%; display: block; }}
    .legend {{ display: flex; gap: 18px; flex-wrap: wrap; margin-top: 12px; font-weight: 700; }}
    .chip {{ display: flex; align-items: center; gap: 8px; }}
    .swatch {{ width: 18px; height: 18px; border-radius: 3px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
    .card {{ background: #101b27; border: 1px solid #243447; padding: 14px; }}
    footer {{ color: #a9b5c4; padding: 20px 28px 30px; }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <div class="sub">Human annotation over NDVI background. For scouting prioritization only. Latest date: {date_text}</div>
  </header>
  <main>
    <section class="hero">
      <h2>Human-Marked Vigour Regions</h2>
      <img src="human_marked_vigour_regions_overlay.png" alt="Human-marked vigour regions over raw NDVI background">
      <div class="legend">
        <span class="chip"><span class="swatch" style="background:#eb2323"></span>Red = Low vigour</span>
        <span class="chip"><span class="swatch" style="background:#be5ad7"></span>Purple = Medium vigour</span>
        <span class="chip"><span class="swatch" style="background:#50eb78"></span>Green = High vigour</span>
      </div>
    </section>
    <section class="grid">
      <div class="card"><h2>QA Proof</h2><img src="visual_proof_contact_sheet.png" alt="Visual proof contact sheet"></div>
      <div class="card"><h2>Annotation Outlines</h2><img src="annotation_outline_layer.png" alt="Human annotation outline layer"></div>
      <div class="card"><h2>Filled Human Regions</h2><img src="human_marked_vigour_regions_filled.png" alt="Filled human-marked vigour regions"></div>
      <div class="card"><h2>Technical Audit</h2><img src="polygon_semantics_audit.png" alt="Polygon semantics audit"></div>
    </section>
    <section class="card">
      <h2>Technical Note</h2>
      <p>RGB-derived NDVI proxy was tested and failed class separability, so no full-field automatic classification is shown.</p>
    </section>
  </main>
  <footer>Ground inspection required to determine cause. Not a disease diagnosis.</footer>
</body>
</html>
"""
    path = output_dir / "customer_report.html"
    path.write_text(html, encoding="utf-8")
    print(f"[WRITE] Customer report: {path}")


def main() -> None:
    args = parse_args()
    row_aware = False
    row_numbering = args.row_numbering or args.bed_numbering or "top_to_bottom"
    if row_numbering != "top_to_bottom":
        print("[ROWS] Deprecated --bed-numbering/--row-numbering value was not top_to_bottom; using top_to_bottom.")
        row_numbering = "top_to_bottom"
    output_paths = ensure_output_dirs(args.output_dir)
    print(f"[START] Field: {args.field_name}")
    print(f"[START] Raw folder: {args.raw_dir}")
    print(f"[START] Annotated folder: {args.annotated_dir}")
    print(f"[START] Output folder: {args.output_dir}")
    print("[START] Human-marked vigour region mode: annotation outlines are truth")
    guided_config = HumanGuidedConfig(
        buffer_distance_pixels=args.human_buffer_pixels,
        min_area_pixels=args.min_inspection_area_pixels,
        max_component_area_pixels=args.max_component_area_pixels,
        max_percent_field_area=args.max_percent_field_area,
        max_component_width_ratio=args.max_component_width_ratio,
        min_human_overlap_ratio=args.min_human_overlap_ratio,
        min_buffer_overlap_ratio=args.min_buffer_overlap_ratio,
        field_area_m2=args.field_area_m2,
        canopy_row_width_ratio=args.canopy_row_width_ratio,
        min_canopy_row_width_pixels=args.min_canopy_row_width_pixels,
        row_gap_length_width_ratio=args.row_gap_length_width_ratio,
        row_gap_width_fraction=args.row_gap_width_fraction,
        row_gap_rejection_score_threshold=args.row_gap_rejection_score_threshold,
        inter_row_gap_width_ratio=args.inter_row_gap_width_ratio,
        inspection_region_merge_x_multiplier=args.inspection_region_merge_x_multiplier,
        inspection_region_merge_y_multiplier=args.inspection_region_merge_y_multiplier,
        max_inspection_regions=args.max_inspection_regions,
    )

    pairs, audit_rows = match_image_pairs_with_audit(args.raw_dir, args.annotated_dir)
    if args.date:
        requested_dates = set(args.date)
        pairs = [pair for pair in pairs if pair.date in requested_dates]
        print(f"[FILTER] Processing requested dates only: {', '.join(sorted(requested_dates))}")
    if not pairs:
        print("[WARN] No matched image pairs found. Nothing to process.")
        return

    all_records = []
    records_by_date = defaultdict(list)
    overlay_paths_by_date: dict[str, Path] = {}
    row_regions_by_date: dict[str, list[RowRegion]] = {}
    row_confidence_by_date: dict[str, float] = {}
    raw_paths_by_date: dict[str, Path] = {}
    image_shapes_by_date: dict[str, tuple[int, int]] = {}
    global_row_count = args.row_count or args.bed_count
    use_cache = False
    segmentation_summary_rows: list[dict[str, object]] = []
    latest_segmentation_date: str | None = None
    if False and row_aware and global_row_count is None:
        detected_counts: list[int] = []
        for pair in pairs:
            raw_bgr = cv2.imread(str(pair.raw_path), cv2.IMREAD_COLOR)
            if raw_bgr is None:
                continue
            rows, _confidence = detect_rows(raw_bgr, row_numbering, None)
            detected_counts.append(len(rows))
        if detected_counts:
            global_row_count = int(round(median(detected_counts)))
            print(f"[ROWS] Auto-selected stable seasonal row count: {global_row_count} from per-date detections {detected_counts}")

    for pair in pairs:
        print(f"[DATE] Processing {pair.date}")
        geojson_name = f"{field_slug(args.field_name)}_{date_slug(pair.date)}_vigour_polygons.geojson"
        geojson_path = output_paths["polygons"] / geojson_name
        overlay_name = f"{field_slug(args.field_name)}_{date_slug(pair.date)}_vigour_polygon_overlay.png"
        overlay_path = output_paths["overlays"] / overlay_name
        audit_json_path = output_paths["debug"] / pair.date / "human_guided_audit.json"
        if use_cache and outputs_are_current([geojson_path, overlay_path, audit_json_path], [pair.raw_path, pair.annotated_path]):
            cached_records = sorted(read_geojson_records(geojson_path), key=lambda record: (record.class_id, record.polygon_id))
            if cached_records:
                print(f"[CACHE] Reusing current outputs for {pair.date}: {len(cached_records)} polygons.")
                all_records.extend(cached_records)
                records_by_date[pair.date].extend(cached_records)
                overlay_paths_by_date[pair.date] = overlay_path
                raw_paths_by_date[pair.date] = pair.raw_path
                first_record = cached_records[0]
                image_shapes_by_date[pair.date] = (int(first_record.processed_height), int(first_record.processed_width))
                if row_aware:
                    raw_for_rows = cv2.imread(str(pair.raw_path), cv2.IMREAD_COLOR)
                    if raw_for_rows is not None and first_record.processed_width and first_record.processed_height:
                        target_size = (int(first_record.processed_width), int(first_record.processed_height))
                        current_size = (int(raw_for_rows.shape[1]), int(raw_for_rows.shape[0]))
                        if current_size != target_size:
                            raw_for_rows = cv2.resize(raw_for_rows, target_size, interpolation=cv2.INTER_AREA)
                        row_regions, row_confidence = detect_rows(raw_for_rows, row_numbering, global_row_count)
                    else:
                        row_regions, row_confidence = [], 0.0
                else:
                    row_regions, row_confidence = [], 0.0
                row_regions_by_date[pair.date] = row_regions
                row_confidence_by_date[pair.date] = row_confidence
                continue
            print(f"[CACHE] Cache read produced no polygons for {pair.date}; rebuilding.")

        raw_bgr = cv2.imread(str(pair.raw_path), cv2.IMREAD_COLOR)
        annotated_bgr = cv2.imread(str(pair.annotated_path), cv2.IMREAD_COLOR)
        if raw_bgr is None:
            print(f"[WARN] Could not read raw image, skipping: {pair.raw_path}")
            continue
        if annotated_bgr is None:
            print(f"[WARN] Could not read annotated image, skipping: {pair.annotated_path}")
            continue
        original_height, original_width = int(raw_bgr.shape[0]), int(raw_bgr.shape[1])
        image_shapes_by_date[pair.date] = (original_height, original_width)

        annotated_bgr, resized = resize_annotated_to_raw(raw_bgr, annotated_bgr)
        if resized:
            print(f"[WARN] Image sizes differ for {pair.date}; resized annotated image to raw image size.")
            audit_rows.append(
                {
                    "status": "REVIEW",
                    "check": "resized image pair",
                    "date": pair.date,
                    "detail": "Annotated image dimensions differed from raw image and were resized before comparison.",
                }
            )
        raw_bgr, annotated_bgr, processing_scale = resize_pair_for_processing(raw_bgr, annotated_bgr, args.max_processing_dimension)
        processed_height, processed_width = int(raw_bgr.shape[0]), int(raw_bgr.shape[1])
        image_shapes_by_date[pair.date] = (processed_height, processed_width)
        if processing_scale != 1.0:
            print(f"[WARN] Large image {pair.date}; processing at scale {processing_scale:.3f} to keep audit run tractable.")
            audit_rows.append(
                {
                    "status": "REVIEW",
                    "check": "large image downsampled for processing",
                    "date": pair.date,
                    "detail": f"Raw and annotated images were resized together by scale {processing_scale:.3f} before mask extraction.",
                }
            )

        annotation_mask = get_annotation_mask(raw_bgr, annotated_bgr)
        changed_pixels = annotation_mask > 0
        field_mask = field_foreground_mask(raw_bgr)
        changed_pixels_in_field = changed_pixels & field_mask
        changed_count_in_field = int(changed_pixels_in_field.sum())
        annotation_area_percent_in_field = changed_count_in_field / max(1, int(field_mask.sum())) * 100
        if annotation_area_percent_in_field < SMALL_ANNOTATION_PERCENT:
            audit_rows.append(
                {
                    "status": "REVIEW",
                    "check": "annotation area too small",
                    "date": pair.date,
                    "detail": f"In-field annotation area is {annotation_area_percent_in_field:.3f}% of the visible field foreground.",
                }
            )
        if annotation_area_percent_in_field > LARGE_ANNOTATION_PERCENT:
            audit_rows.append(
                {
                    "status": "WARNING",
                    "check": "annotation area too large",
                    "date": pair.date,
                    "detail": f"In-field annotation area is {annotation_area_percent_in_field:.1f}% of the visible field foreground.",
                }
            )
        raw_class_masks = human_annotation_colour_masks(raw_bgr, annotated_bgr, annotation_mask)
        classified_changed = raw_class_masks["low_vigour"] | raw_class_masks["medium_vigour"] | raw_class_masks["high_vigour"]
        if changed_count_in_field:
            unclassified_percent = int((changed_pixels_in_field & ~classified_changed).sum()) / changed_count_in_field * 100
            if unclassified_percent > UNCLASSIFIED_CHANGED_PERCENT:
                audit_rows.append(
                    {
                        "status": "REVIEW",
                        "check": "unclassified changed pixels > 10%",
                        "date": pair.date,
                        "detail": f"{unclassified_percent:.1f}% of in-field changed pixels did not match red, purple, or light-green annotation classes.",
                    }
                )
        class_masks = {
            name: clean_segmentation_mask(mask.astype("uint8") * 255)
            for name, mask in raw_class_masks.items()
        }
        segmentation_summary_rows.extend(
            write_segmentation_outputs(
                args.output_dir,
                pair.date,
                args.field_name,
                raw_bgr,
                annotated_bgr,
                class_masks,
                processing_scale,
                original_width,
                original_height,
                processed_width,
                processed_height,
            )
        )
        latest_segmentation_date = pair.date
        print(f"[HUMAN-MARKED] {pair.date}: wrote annotation outlines and filled human-marked vigour regions.")

    if segmentation_summary_rows:
        write_class_area_summary(segmentation_summary_rows, args.output_dir / "class_area_summary.csv")
        write_segmentation_html(args.output_dir, args.field_name, latest_segmentation_date)
        print("[DONE] Human-marked vigour region output complete.")
        return

    if not all_records:
        print("[WARN] No polygons were generated from any matched pair.")
        return

    summary_path = output_paths["summaries"] / f"{field_slug(args.field_name)}_vigour_polygon_summary.csv"
    write_polygon_summary(all_records, summary_path)

    if row_aware:
        bed_summary_path = output_paths["summaries"] / f"{field_slug(args.field_name)}_bed_vigour_summary.csv"
        write_bed_summary(all_records, bed_summary_path, args.ground_notes_csv, args.event_log_csv)
        row_summary_path = args.output_dir / "row_vigour_summary.csv"
        write_row_summary(all_records, row_summary_path)

    chart_path = output_paths["reports"] / f"{field_slug(args.field_name)}_vigour_class_chart.png"
    make_chart(records_by_date, chart_path)

    hero_path = None
    if overlay_paths_by_date:
        latest_date = sorted(overlay_paths_by_date)[-1]
        latest_raw_path = raw_paths_by_date.get(latest_date)
        latest_meta_record = records_by_date[latest_date][0] if records_by_date.get(latest_date) else None
        latest_raw_bgr = cv2.imread(str(latest_raw_path), cv2.IMREAD_COLOR) if latest_raw_path else None
        if latest_raw_bgr is not None:
            if latest_meta_record is not None:
                target_size = (int(latest_meta_record.processed_width), int(latest_meta_record.processed_height))
                current_size = (int(latest_raw_bgr.shape[1]), int(latest_raw_bgr.shape[0]))
                if all(value > 0 for value in target_size) and current_size != target_size:
                    latest_raw_bgr = cv2.resize(latest_raw_bgr, target_size, interpolation=cv2.INTER_AREA)
            hero_path = args.output_dir / "priority_rows_overlay.png"
            latest_rows = row_regions_by_date.get(latest_date, [])
            latest_confidence = row_confidence_by_date.get(latest_date, 0.0)
            if latest_rows:
                write_row_lines_geojson(
                    latest_rows,
                    args.output_dir / "row_lines.geojson",
                    latest_date,
                    str(latest_raw_path or ""),
                    processing_scale=float(latest_meta_record.processing_scale) if latest_meta_record else 1.0,
                    original_width=int(latest_meta_record.original_width) if latest_meta_record else 0,
                    original_height=int(latest_meta_record.original_height) if latest_meta_record else 0,
                    processed_width=int(latest_meta_record.processed_width) if latest_meta_record else 0,
                    processed_height=int(latest_meta_record.processed_height) if latest_meta_record else 0,
                    coordinate_space=(latest_meta_record.coordinate_space if latest_meta_record else "processed_image_pixels"),
                )
            priority_targets = top_priority_beds(all_records, latest_date, row_regions_by_date, row_confidence_by_date, image_shapes_by_date, limit=20)
            write_inspection_targets(
                priority_targets,
                args.output_dir / "inspection_targets.geojson",
                args.output_dir / "inspection_targets.csv",
            )
            draw_polygon_overlay(
                latest_raw_bgr,
                records_by_date[latest_date],
                hero_path,
                latest_rows if row_aware and latest_confidence >= 0.9 else None,
                priority_targets[:5],
            )

    html_path = args.output_dir / "mission_debrief.html"
    final_audit_rows = build_audit_rows(all_records, audit_rows)
    write_html_report(
        args.field_name,
        all_records,
        overlay_paths_by_date,
        chart_path,
        html_path,
        bed_aware=row_aware,
        audit_rows=final_audit_rows,
        hero_image_path=hero_path,
        row_regions_by_date=row_regions_by_date,
        row_confidence_by_date=row_confidence_by_date,
        image_shapes_by_date=image_shapes_by_date,
    )
    release_audit_path = output_paths["reports"] / f"{field_slug(args.field_name)}_customer_release_audit.md"
    write_customer_release_audit(
        args.field_name,
        all_records,
        final_audit_rows,
        release_audit_path,
        row_regions_by_date,
        row_confidence_by_date,
        image_shapes_by_date,
    )

    print("[DONE] Vigour polygon report complete.")


if __name__ == "__main__":
    main()
