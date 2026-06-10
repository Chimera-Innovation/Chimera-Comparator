from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .annotation_parser import CLASS_SCHEMA, PolygonRecord, rendered_green_and_ndvi_proxy
from .bed_detection import make_field_mask
from .row_detection import RowRegion, row_span_for_bounds


@dataclass(frozen=True)
class HumanGuidedConfig:
    buffer_distance_pixels: int = 50
    min_area_pixels: int = 900
    max_component_area_pixels: int | None = None
    max_percent_field_area: float = 25.0
    max_component_width_ratio: float = 0.55
    min_human_overlap_ratio: float = 0.01
    min_buffer_overlap_ratio: float = 0.95
    row_chunk_size: int = 5
    column_count: int = 18
    field_area_m2: float | None = None
    write_debug_outputs: bool = True


def raw_vigour_masks(raw_bgr: np.ndarray, field_mask: np.ndarray) -> dict[str, np.ndarray]:
    """Create supporting NDVI-rendered vigour masks from the raw evidence image."""
    green_index, visual_ndvi_proxy = rendered_green_and_ndvi_proxy(raw_bgr)
    valid = field_mask > 0
    if not np.any(valid):
        valid = np.ones(raw_bgr.shape[:2], dtype=bool)

    valid_green = green_index[valid]
    valid_ndvi = visual_ndvi_proxy[valid]
    green_low = float(np.percentile(valid_green, 32))
    green_mid = float(np.percentile(valid_green, 52))
    ndvi_low = float(np.percentile(valid_ndvi, 34))
    ndvi_mid = float(np.percentile(valid_ndvi, 55))

    low = ((green_index <= green_low) | (visual_ndvi_proxy <= ndvi_low)) & valid
    medium = (((green_index > green_low) & (green_index <= green_mid)) | ((visual_ndvi_proxy > ndvi_low) & (visual_ndvi_proxy <= ndvi_mid))) & valid
    healthy = ((green_index > green_mid) & (visual_ndvi_proxy > ndvi_mid)) & valid

    kernel = np.ones((3, 3), np.uint8)
    return {
        "healthy_mask": cv2.morphologyEx(healthy.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel),
        "medium_vigour_mask": cv2.morphologyEx(medium.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel),
        "low_vigour_mask": cv2.morphologyEx(low.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel),
        "stress_mask": cv2.morphologyEx((low | medium).astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel),
    }


def build_human_mask(class_masks: dict[str, np.ndarray], include_high: bool = False) -> np.ndarray:
    names = ["low_vigour", "medium_vigour"]
    if include_high:
        names.append("high_vigour")
    first = next(iter(class_masks.values()))
    mask = np.zeros(first.shape[:2], dtype=np.uint8)
    for name in names:
        if name in class_masks:
            mask = cv2.bitwise_or(mask, (class_masks[name] > 0).astype(np.uint8) * 255)
    return mask


def buffer_mask(mask: np.ndarray, distance_pixels: int) -> np.ndarray:
    radius = max(1, int(distance_pixels))
    size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(mask, kernel, iterations=1)


def cleanup_confirmation(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned


def row_valid_mask(shape: tuple[int, int], rows: list[RowRegion]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for row in rows:
        y_min = max(0, min(shape[0] - 1, row.y_min))
        y_max = max(0, min(shape[0] - 1, row.y_max))
        x_min = max(0, min(shape[1] - 1, row.x_min))
        x_max = max(0, min(shape[1] - 1, row.x_max))
        if y_max > y_min and x_max > x_min:
            cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, cv2.FILLED)
    return mask


def row_clip_masks(shape: tuple[int, int], rows: list[RowRegion], config: HumanGuidedConfig) -> list[tuple[list[RowRegion], np.ndarray]]:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda row: row.row_number)
    clips = []
    for index in range(0, len(ordered), max(1, config.row_chunk_size)):
        group = ordered[index : index + max(1, config.row_chunk_size)]
        if not group:
            continue
        mask = np.zeros(shape, dtype=np.uint8)
        y_min = max(0, min(row.y_min for row in group))
        y_max = min(shape[0] - 1, max(row.y_max for row in group))
        x_min = max(0, min(row.x_min for row in group))
        x_max = min(shape[1] - 1, max(row.x_max for row in group))
        cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, cv2.FILLED)
        clips.append((group, mask))
    return clips


def segment_clip_masks(shape: tuple[int, int], group: list[RowRegion], column_count: int) -> list[np.ndarray]:
    y_min = max(0, min(row.y_min for row in group))
    y_max = min(shape[0] - 1, max(row.y_max for row in group))
    x_min = max(0, min(row.x_min for row in group))
    x_max = min(shape[1] - 1, max(row.x_max for row in group))
    edges = np.linspace(x_min, x_max, max(2, column_count + 1)).round().astype(int)
    masks = []
    for index in range(len(edges) - 1):
        left = int(edges[index])
        right = int(edges[index + 1])
        if right <= left:
            continue
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.rectangle(mask, (left, y_min), (right, y_max), 255, cv2.FILLED)
        masks.append(mask)
    return masks


def affected_rows_for_mask(mask: np.ndarray, rows: list[RowRegion]) -> list[int]:
    affected = []
    for row in rows:
        y_min = max(0, min(mask.shape[0] - 1, row.y_min))
        y_max = max(0, min(mask.shape[0] - 1, row.y_max))
        x_min = max(0, min(mask.shape[1] - 1, row.x_min))
        x_max = max(0, min(mask.shape[1] - 1, row.x_max))
        if y_max <= y_min or x_max <= x_min:
            continue
        if np.count_nonzero(mask[y_min : y_max + 1, x_min : x_max + 1]) > 0:
            affected.append(row.row_number)
    return affected


def severity_from_overlap(low_overlap: float, medium_overlap: float) -> tuple[str, int]:
    if low_overlap > 0 and low_overlap >= medium_overlap * 0.4:
        return "high", CLASS_SCHEMA["low_vigour"]
    if low_overlap > 0:
        return "moderate_high", CLASS_SCHEMA["low_vigour"]
    return "medium", CLASS_SCHEMA["medium_vigour"]


def contours_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    contours, _hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def vigour_map(raw_bgr: np.ndarray, vigour_masks: dict[str, np.ndarray]) -> np.ndarray:
    output = np.zeros_like(raw_bgr)
    output[vigour_masks["healthy_mask"] > 0] = (70, 225, 95)
    output[vigour_masks["medium_vigour_mask"] > 0] = (220, 120, 170)
    output[vigour_masks["low_vigour_mask"] > 0] = (35, 35, 235)
    return cv2.addWeighted(raw_bgr, 0.55, output, 0.45, 0)


def component_debug_overlay(raw_bgr: np.ndarray, labels: np.ndarray, component_rows: list[dict[str, object]]) -> np.ndarray:
    overlay = raw_bgr.copy()
    palette = [(35, 35, 235), (220, 120, 170), (70, 225, 95), (40, 190, 245), (255, 255, 255)]
    for row in component_rows:
        label_id = int(row["component_id"])
        mask = labels == label_id
        color = palette[label_id % len(palette)]
        tint = overlay.copy()
        tint[mask] = color
        overlay = cv2.addWeighted(tint, 0.2, overlay, 0.8, 0)
        x = int(row["x"])
        y = int(row["y"])
        w = int(row["width"])
        h = int(row["height"])
        status = str(row["status"])
        rect_color = (70, 225, 95) if status == "accepted" else (35, 35, 235)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), rect_color, 3, cv2.LINE_AA)
        cv2.putText(overlay, f"{label_id}:{status}", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, rect_color, 2, cv2.LINE_AA)
    return overlay


def final_polygon_overlay(raw_bgr: np.ndarray, records: list[PolygonRecord]) -> np.ndarray:
    output = raw_bgr.copy()
    for record in records:
        color = (35, 35, 235) if record.class_name == "low_vigour" else (220, 120, 170)
        polygon = np.array(record.coordinates, dtype=np.int32).reshape((-1, 1, 2))
        cv2.drawContours(output, [polygon], -1, color, 4, cv2.LINE_AA)
        cv2.putText(
            output,
            record.polygon_id,
            (int(record.centroid_x), int(record.centroid_y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
    return output


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask)


def write_component_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "component_id",
        "status",
        "reason",
        "area",
        "field_percent",
        "human_overlap_ratio",
        "buffer_overlap_ratio",
        "width_ratio",
        "x",
        "y",
        "width",
        "height",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_debug_bundle(
    debug_root: Path,
    date: str,
    raw_bgr: np.ndarray,
    vigour_masks: dict[str, np.ndarray],
    human_mask: np.ndarray,
    human_buffer_mask: np.ndarray,
    stress_mask: np.ndarray,
    confirmation_mask: np.ndarray,
    labels: np.ndarray,
    component_rows: list[dict[str, object]],
    records: list[PolygonRecord],
    audit: dict[str, object],
) -> None:
    date_dir = debug_root / date
    date_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(date_dir / "01_raw_image.png"), raw_bgr)
    cv2.imwrite(str(date_dir / "02_vigour_map.png"), vigour_map(raw_bgr, vigour_masks))
    write_mask(date_dir / "03_human_annotation_mask.png", human_mask)
    write_mask(date_dir / "04_human_buffer_mask.png", human_buffer_mask)
    write_mask(date_dir / "05_stress_mask.png", stress_mask)
    write_mask(date_dir / "06_confirmation_mask.png", confirmation_mask)
    cv2.imwrite(str(date_dir / "07_connected_component_debug_overlay.png"), component_debug_overlay(raw_bgr, labels, component_rows))
    cv2.imwrite(str(date_dir / "08_final_polygon_overlay.png"), final_polygon_overlay(raw_bgr, records))
    write_component_csv(date_dir / "connected_components.csv", component_rows)
    (date_dir / "human_guided_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


def build_human_guided_polygons(
    class_masks: dict[str, np.ndarray],
    raw_bgr: np.ndarray,
    field_name: str,
    date: str,
    growth_stage: str,
    rows: list[RowRegion] | None = None,
    config: HumanGuidedConfig | None = None,
    debug_root: Path | None = None,
) -> list[PolygonRecord]:
    config = config or HumanGuidedConfig()
    rows = rows or []
    shape = raw_bgr.shape[:2]
    field_mask = make_field_mask(raw_bgr)
    human_mask = build_human_mask(class_masks)
    if not np.any(human_mask):
        print(f"[GUIDED] {date}: no low/medium human annotation seeds; no inspection polygons.")
        return []

    human_buffer_mask = buffer_mask(human_mask, config.buffer_distance_pixels)
    vigour_masks = raw_vigour_masks(raw_bgr, field_mask)
    stress_mask = vigour_masks["stress_mask"]
    valid_mask = (field_mask > 0).astype(np.uint8) * 255
    if rows:
        valid_mask = cv2.bitwise_and(valid_mask, row_valid_mask(shape, rows))
    confirmation_mask = cv2.bitwise_and(stress_mask, human_buffer_mask)
    confirmation_mask = cv2.bitwise_and(confirmation_mask, valid_mask)
    confirmation_mask = cleanup_confirmation(confirmation_mask)

    max_area = (config.max_percent_field_area / 100.0) * float(shape[0] * shape[1])
    max_component_area = float(config.max_component_area_pixels or max_area)
    green_index, visual_ndvi_proxy = rendered_green_and_ndvi_proxy(raw_bgr)
    low_mask = (class_masks.get("low_vigour", np.zeros(shape, dtype=np.uint8)) > 0).astype(np.uint8) * 255
    medium_mask = (class_masks.get("medium_vigour", np.zeros(shape, dtype=np.uint8)) > 0).astype(np.uint8) * 255

    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats((confirmation_mask > 0).astype(np.uint8), connectivity=8)
    component_rows: list[dict[str, object]] = []
    accepted_components: list[np.ndarray] = []
    image_area = float(shape[0] * shape[1])
    for label_id in range(1, component_count):
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        area = float(stats[label_id, cv2.CC_STAT_AREA])
        field_percent = area / max(1.0, image_area) * 100.0
        component_mask = (labels == label_id).astype(np.uint8) * 255
        human_overlap = float(np.count_nonzero(cv2.bitwise_and(component_mask, human_mask)))
        buffer_overlap = float(np.count_nonzero(cv2.bitwise_and(component_mask, human_buffer_mask)))
        human_overlap_ratio = human_overlap / max(1.0, area)
        buffer_overlap_ratio = buffer_overlap / max(1.0, area)
        width_ratio = width / max(1.0, float(shape[1]))
        reason = ""
        if area < config.min_area_pixels:
            reason = "area_below_min"
        elif area > max_component_area:
            reason = "area_above_max_component"
        elif field_percent > config.max_percent_field_area:
            reason = "too_much_field_area"
        elif buffer_overlap_ratio < config.min_buffer_overlap_ratio:
            reason = "weak_buffer_overlap"
        elif human_overlap_ratio < config.min_human_overlap_ratio:
            reason = "weak_human_overlap"
        elif width_ratio > config.max_component_width_ratio:
            reason = "too_wide_across_field"
        status = "accepted" if not reason else "rejected"
        component_rows.append(
            {
                "component_id": label_id,
                "status": status,
                "reason": reason,
                "area": round(area, 2),
                "field_percent": round(field_percent, 6),
                "human_overlap_ratio": round(human_overlap_ratio, 6),
                "buffer_overlap_ratio": round(buffer_overlap_ratio, 6),
                "width_ratio": round(width_ratio, 6),
                "x": x,
                "y": y,
                "width": width,
                "height": height,
            }
        )
        if status == "accepted":
            accepted_components.append(component_mask)

    extraction_masks: list[np.ndarray] = []
    for component_mask in accepted_components:
        if rows:
            for group, row_mask in row_clip_masks(shape, rows, config):
                row_component = cv2.bitwise_and(component_mask, row_mask)
                if not np.any(row_component):
                    continue
                for segment_mask in segment_clip_masks(shape, group, config.column_count):
                    segment = cv2.bitwise_and(row_component, segment_mask)
                    if np.count_nonzero(segment) >= config.min_area_pixels:
                        extraction_masks.append(segment)
        else:
            extraction_masks.append(component_mask)

    records: list[PolygonRecord] = []
    polygon_index = 0
    total_pixels = float(shape[0] * shape[1])
    for extraction_mask in extraction_masks:
        for contour in contours_from_mask(extraction_mask):
            area = float(cv2.contourArea(contour))
            if area < config.min_area_pixels or area > max_area:
                continue
            contour_mask = np.zeros(shape, dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
            if np.count_nonzero(cv2.bitwise_and(contour_mask, human_buffer_mask)) <= 0:
                continue
            human_overlap = float(np.count_nonzero(cv2.bitwise_and(contour_mask, human_mask)))
            buffer_overlap = float(np.count_nonzero(cv2.bitwise_and(contour_mask, human_buffer_mask)))
            if buffer_overlap / max(1.0, area) < config.min_buffer_overlap_ratio:
                continue
            if human_overlap / max(1.0, area) < config.min_human_overlap_ratio:
                continue
            epsilon = 0.01 * cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, epsilon, True)
            coords = [[int(point[0][0]), int(point[0][1])] for point in polygon]
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
            affected_rows = affected_rows_for_mask(contour_mask, rows)
            if affected_rows:
                row_id = f"rows_{min(affected_rows):03d}_{max(affected_rows):03d}"
                bed_area = sum(row.area_pixels for row in rows if row.row_number in set(affected_rows))
            else:
                row_start, row_end = row_span_for_bounds(int(centroid_y), int(centroid_y), rows) if rows else (0, 0)
                affected_rows = [value for value in (row_start, row_end) if value]
                row_id = "human_guided_zone"
                bed_area = total_pixels
            active = contour_mask > 0
            low_overlap = float(np.count_nonzero(cv2.bitwise_and(contour_mask, low_mask)))
            medium_overlap = float(np.count_nonzero(cv2.bitwise_and(contour_mask, medium_mask)))
            severity, class_id = severity_from_overlap(low_overlap, medium_overlap)
            class_name = "low_vigour" if class_id == CLASS_SCHEMA["low_vigour"] else "medium_vigour"
            ndvi_overlap = float(np.count_nonzero(cv2.bitwise_and(contour_mask, vigour_masks["stress_mask"])))
            confidence = 0.55 + 0.25 * min(1.0, human_overlap / max(1.0, area)) + 0.2 * min(1.0, ndvi_overlap / max(1.0, area))
            area_m2 = None
            if config.field_area_m2 is not None:
                area_m2 = area / max(1.0, total_pixels) * config.field_area_m2
            vigour_loss_percent = (low_overlap + medium_overlap) / max(1.0, area) * 100.0
            mean_green = float(green_index[active].mean()) if np.any(active) else 0.0
            mean_ndvi_proxy = float(visual_ndvi_proxy[active].mean()) if np.any(active) else 0.0
            polygon_index += 1
            records.append(
                PolygonRecord(
                    field_name=field_name,
                    date=date,
                    growth_stage=growth_stage,
                    class_name=class_name,
                    class_id=class_id,
                    polygon_id=f"P{polygon_index:03d}",
                    bed_id=row_id,
                    centroid_x=centroid_x,
                    centroid_y=centroid_y,
                    area_pixels=area,
                    area_percent=area / total_pixels * 100.0,
                    bed_area_pixels=float(max(1.0, bed_area)),
                    area_percent_within_bed=area / max(1.0, float(bed_area)) * 100.0,
                    scouting_priority="highest priority" if severity.startswith("high") else "monitor / secondary priority",
                    confidence_source="human_annotation_constrained_ndvi_support",
                    note="Generated only inside buffered human annotation influence. For scouting prioritization only.",
                    mean_green_index=mean_green,
                    mean_visual_ndvi_proxy=mean_ndvi_proxy,
                    coordinates=coords,
                    area_m2=area_m2,
                    severity=severity,
                    confidence=float(min(0.99, confidence)),
                    affected_rows=affected_rows,
                    affected_beds=[row_id],
                    vigour_loss_percent=float(min(100.0, vigour_loss_percent)),
                    source=["human_annotation", "ndvi_stress"],
                )
            )
    audit = {
        "date": date,
        "component_count": component_count - 1,
        "accepted_components": sum(1 for row in component_rows if row["status"] == "accepted"),
        "rejected_components": sum(1 for row in component_rows if row["status"] == "rejected"),
        "polygon_count": len(records),
        "confirmation_formula": "confirmation_mask = stress_mask AND human_buffer_mask AND valid_crop_or_row_mask",
        "contour_source": "cleaned_binary_component_or_row_segment_mask",
        "contour_mode": "cv2.RETR_EXTERNAL",
        "uses_canny": False,
        "buffer_distance_pixels": config.buffer_distance_pixels,
        "min_area_pixels": config.min_area_pixels,
        "max_percent_field_area": config.max_percent_field_area,
    }
    if config.write_debug_outputs and debug_root is not None:
        write_debug_bundle(
            debug_root,
            date,
            raw_bgr,
            vigour_masks,
            human_mask,
            human_buffer_mask,
            stress_mask,
            confirmation_mask,
            labels,
            component_rows,
            records,
            audit,
        )
    print(
        f"[GUIDED] {date}: {len(records)} human-guided inspection polygons; "
        f"components accepted={audit['accepted_components']} rejected={audit['rejected_components']}."
    )
    return records
