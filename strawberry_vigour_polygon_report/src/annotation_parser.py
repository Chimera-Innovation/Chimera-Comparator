from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .bed_detection import BedRegion, assign_bed_id


CLASS_SCHEMA = {
    "background": 0,
    "low_vigour": 1,
    "medium_vigour": 2,
    "high_vigour": 3,
}

SCOUTING_PRIORITY = {
    "low_vigour": "highest priority",
    "medium_vigour": "monitor / secondary priority",
    "high_vigour": "reference zone",
}


@dataclass
class PolygonRecord:
    field_name: str
    date: str
    growth_stage: str
    class_name: str
    class_id: int
    polygon_id: str
    bed_id: str
    centroid_x: float
    centroid_y: float
    area_pixels: float
    area_percent: float
    bed_area_pixels: float
    area_percent_within_bed: float
    scouting_priority: str
    confidence_source: str
    note: str
    mean_green_index: float
    mean_visual_ndvi_proxy: float
    coordinates: list[list[int]]
    area_m2: float | None = None
    severity: str = ""
    confidence: float = 0.0
    affected_rows: list[int] | None = None
    affected_beds: list[str] | None = None
    vigour_loss_percent: float = 0.0
    source: list[str] | None = None
    row_segment_start_m: float | None = None
    row_segment_end_m: float | None = None
    row_overlap_percent: float = 0.0
    review_status: str = "needs_human_review"
    processing_scale: float = 1.0
    original_width: int = 0
    original_height: int = 0
    processed_width: int = 0
    processed_height: int = 0
    coordinate_space: str = "processed_image_pixels"


def get_annotation_mask(raw_bgr: np.ndarray, annotated_bgr: np.ndarray, diff_threshold: int = 25) -> np.ndarray:
    diff = cv2.absdiff(raw_bgr, annotated_bgr)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    mask = gray > diff_threshold
    return mask.astype(np.uint8) * 255


def resize_annotated_to_raw(raw_bgr: np.ndarray, annotated_bgr: np.ndarray) -> tuple[np.ndarray, bool]:
    raw_h, raw_w = raw_bgr.shape[:2]
    ann_h, ann_w = annotated_bgr.shape[:2]
    if (raw_w, raw_h) == (ann_w, ann_h):
        return annotated_bgr, False
    resized = cv2.resize(annotated_bgr, (raw_w, raw_h), interpolation=cv2.INTER_LINEAR)
    return resized, True


def class_masks_from_annotation(annotated_bgr: np.ndarray, annotation_mask: np.ndarray) -> dict[str, np.ndarray]:
    masks = raw_class_masks_from_annotation(annotated_bgr, annotation_mask)
    return {name: clean_mask(mask.astype(np.uint8) * 255) for name, mask in masks.items()}


def raw_class_masks_from_annotation(annotated_bgr: np.ndarray, annotation_mask: np.ndarray) -> dict[str, np.ndarray]:
    hsv = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2HSV)
    annotation_bool = annotation_mask > 0

    red_1 = cv2.inRange(hsv, (0, 70, 70), (12, 255, 255)) > 0
    red_2 = cv2.inRange(hsv, (170, 70, 70), (180, 255, 255)) > 0
    purple = cv2.inRange(hsv, (125, 40, 40), (165, 255, 255)) > 0
    light_green = cv2.inRange(hsv, (35, 40, 80), (90, 255, 255)) > 0

    masks = {
        "low_vigour": (red_1 | red_2) & annotation_bool,
        "medium_vigour": purple & annotation_bool,
        "high_vigour": light_green & annotation_bool,
    }
    return masks


def clean_mask(mask: np.ndarray) -> np.ndarray:
    open_kernel = np.ones((5, 5), np.uint8)
    close_kernel = np.ones((7, 7), np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)
    return fill_closed_regions(cleaned)


def fill_closed_regions(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return mask
    flood = mask.copy()
    h, w = flood.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(mask, holes)
    return filled


def masks_to_polygons(
    class_masks: dict[str, np.ndarray],
    raw_bgr: np.ndarray,
    field_name: str,
    date: str,
    growth_stage: str,
    bed_regions: list[BedRegion] | None = None,
    min_area_pixels: int = 1000,
) -> list[PolygonRecord]:
    total_pixels = next(iter(class_masks.values())).size
    green_index, visual_ndvi_proxy = rendered_green_and_ndvi_proxy(raw_bgr)
    records: list[PolygonRecord] = []
    for class_name, mask in class_masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        kept = 0
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area_pixels:
                continue
            epsilon = 0.004 * cv2.arcLength(contour, True)
            simplified = cv2.approxPolyDP(contour, epsilon, True)
            coords = [[int(point[0][0]), int(point[0][1])] for point in simplified]
            if len(coords) < 3:
                continue
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            moments = cv2.moments(contour)
            if moments["m00"]:
                centroid_x = float(moments["m10"] / moments["m00"])
                centroid_y = float(moments["m01"] / moments["m00"])
            else:
                xy = contour.reshape(-1, 2)
                centroid_x = float(xy[:, 0].mean())
                centroid_y = float(xy[:, 1].mean())
            bed = assign_bed_id(centroid_y, bed_regions or [])
            bed_id = bed.bed_id if bed else "unassigned"
            bed_area = float(bed.area_pixels if bed else total_pixels)
            area_percent_within_bed = area / max(1.0, bed_area) * 100
            contour_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
            active = contour_mask > 0
            mean_green = float(green_index[active].mean()) if np.any(active) else 0.0
            mean_ndvi_proxy = float(visual_ndvi_proxy[active].mean()) if np.any(active) else 0.0
            kept += 1
            records.append(
                PolygonRecord(
                    field_name=field_name,
                    date=date,
                    growth_stage=growth_stage,
                    class_name=class_name,
                    class_id=CLASS_SCHEMA[class_name],
                    polygon_id=f"{date}_{class_name}_{kept:03d}",
                    bed_id=bed_id,
                    centroid_x=centroid_x,
                    centroid_y=centroid_y,
                    area_pixels=area,
                    area_percent=area / total_pixels * 100,
                    bed_area_pixels=bed_area,
                    area_percent_within_bed=area_percent_within_bed,
                    scouting_priority=SCOUTING_PRIORITY[class_name],
                    confidence_source="human_annotation",
                    note="For scouting prioritization only. Not a disease diagnosis.",
                    mean_green_index=mean_green,
                    mean_visual_ndvi_proxy=mean_ndvi_proxy,
                    coordinates=coords,
                )
            )
        print(f"[POLYGONS] {date} {class_name}: {kept}")
    return records


def rendered_green_and_ndvi_proxy(raw_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = raw_bgr.astype(np.float32) / 255.0
    blue = raw[:, :, 0]
    green = raw[:, :, 1]
    red = raw[:, :, 2]
    green_index = np.clip((2.0 * green - red - blue + 2.0) / 4.0, 0.0, 1.0)
    visual_ndvi_proxy = np.clip((green - red) / (green + red + 1e-6), -1.0, 1.0)
    return green_index, visual_ndvi_proxy


def class_coverage_from_polygons(records: list[PolygonRecord]) -> dict[str, float]:
    coverage = {"low_vigour": 0.0, "medium_vigour": 0.0, "high_vigour": 0.0}
    for record in records:
        coverage[record.class_name] += record.area_percent
    return coverage
