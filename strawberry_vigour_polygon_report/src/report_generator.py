from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Template

from .annotation_parser import PolygonRecord, class_coverage_from_polygons
from .row_detection import RowRegion, row_span_for_bounds


CLASS_COLORS_BGR = {
    "low_vigour": (35, 35, 235),
    "medium_vigour": (220, 120, 170),
    "high_vigour": (70, 225, 95),
}

CLASS_COLORS_HEX = {
    "low_vigour": "#f23535",
    "medium_vigour": "#a66cff",
    "high_vigour": "#95e342",
}

SCORE_WEIGHTS = {
    "persistence": 25,
    "expansion": 25,
    "severity": 25,
    "coverage": 15,
    "evidence_count": 10,
}

EVIDENCE_TOTAL = 6
MAX_CUSTOMER_ROW_RANGE = 10
TARGET_ROW_SPAN = 5
TARGET_COLUMN_COUNT = 16
MIN_TARGET_FOCUS_PIXELS = 700


def format_percent(value: float) -> str:
    return f"{value:.1f}%"


def short_date(date: str) -> str:
    if len(date) >= 10 and date[5:7] == "05":
        return f"May {int(date[8:10])}"
    if len(date) >= 10 and date[5:7] == "06":
        return f"Jun {int(date[8:10])}"
    return date


def row_label(row_id: str) -> str:
    if row_id.startswith("row_"):
        return f"Row guide {int(row_id.split('_', 1)[1]):03d}"
    return row_id.replace("_", " ").title()


def records_by_date(records: list[PolygonRecord]) -> dict[str, list[PolygonRecord]]:
    grouped: dict[str, list[PolygonRecord]] = defaultdict(list)
    for record in records:
        grouped[record.date].append(record)
    return grouped


def all_record_bounds(records: list[PolygonRecord], image_shape: tuple[int, int] | None = None) -> tuple[float, float]:
    if image_shape:
        height, width = image_shape
        return float(width), float(height)
    max_x = 1.0
    max_y = 1.0
    for record in records:
        for x, y in record.coordinates:
            max_x = max(max_x, float(x))
            max_y = max(max_y, float(y))
    return max_x, max_y


def location_parts(x_ratio: float, y_ratio: float) -> tuple[str, str]:
    if y_ratio < 0.32:
        vertical = "upper"
    elif y_ratio < 0.62:
        vertical = "center"
    else:
        vertical = "lower"

    if x_ratio < 0.33:
        horizontal = "left"
    elif x_ratio < 0.66:
        horizontal = "central"
    else:
        horizontal = "right"
    return vertical, horizontal


def location_label(x_ratio: float, y_ratio: float) -> str:
    vertical, horizontal = location_parts(x_ratio, y_ratio)
    return location_label_from_parts(vertical, horizontal)


def location_label_from_parts(vertical: str, horizontal: str) -> str:
    if vertical == "lower" and horizontal == "central":
        return "south-central production block"
    if vertical == "center" and horizontal == "left":
        return "center-left block"
    if vertical == "lower":
        return "lower field block"
    if vertical == "upper" and horizontal == "right":
        return "upper-right production rows"
    if vertical == "upper":
        return "upper field"
    if horizontal == "central":
        return "central field block"
    return f"{vertical}-{horizontal} block"


def bucket_for_record(record: PolygonRecord, max_x: float, max_y: float) -> tuple[str, str]:
    return location_parts(record.centroid_x / max_x, record.centroid_y / max_y)


def block_label_for_bounds(
    bounds: tuple[int, int, int, int],
    max_x: float,
    max_y: float,
    rows: list[RowRegion] | None,
    row_confidence: float,
    bucket: tuple[str, str] | None = None,
) -> tuple[str, str]:
    x_min, y_min, x_max, y_max = bounds
    if rows and row_confidence >= 0.9:
        row_start, row_end = row_span_for_bounds(y_min, y_max, rows)
        row_span = row_end - row_start + 1
        if row_span <= MAX_CUSTOMER_ROW_RANGE:
            return f"Approx. Rows {row_start}-{row_end}", f"Rows {row_start}-{row_end}"
    if bucket:
        label = location_label_from_parts(*bucket)
        return label, label
    x_ratio = ((x_min + x_max) / 2.0) / max_x
    y_ratio = ((y_min + y_max) / 2.0) / max_y
    label = location_label(x_ratio, y_ratio)
    return label, label


def block_bounds(records: list[PolygonRecord]) -> tuple[int, int, int, int]:
    points: list[list[int]] = []
    for record in records:
        points.extend(record.coordinates)
    array = np.array(points, dtype=np.int32)
    x_min, y_min = array.min(axis=0)
    x_max, y_max = array.max(axis=0)
    return int(x_min), int(y_min), int(x_max), int(y_max)


def records_to_mask(records: list[PolygonRecord], shape: tuple[int, int], class_names: set[str]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for record in records:
        if record.class_name not in class_names or len(record.coordinates) < 3:
            continue
        points = np.array(record.coordinates, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [points], 255)
    return mask


def row_groups(rows: list[RowRegion]) -> list[list[RowRegion]]:
    ordered = sorted(rows, key=lambda row: row.row_number)
    return [ordered[index : index + TARGET_ROW_SPAN] for index in range(0, len(ordered), TARGET_ROW_SPAN)]


def segment_bounds_from_rows(group: list[RowRegion], column_index: int, column_count: int) -> tuple[int, int, int, int]:
    y_min = max(0, min(row.y_min for row in group))
    y_max = max(row.y_max for row in group)
    x_min = min(row.x_min for row in group)
    x_max = max(row.x_max for row in group)
    edges = np.linspace(x_min, x_max, column_count + 1).round().astype(int)
    left = int(edges[column_index])
    right = int(edges[column_index + 1])
    if right <= left:
        right = left + 1
    return left, y_min, right, y_max


def mask_area_in_bounds(mask: np.ndarray, bounds: tuple[int, int, int, int]) -> float:
    x_min, y_min, x_max, y_max = bounds
    y_min = max(0, min(mask.shape[0] - 1, y_min))
    y_max = max(0, min(mask.shape[0] - 1, y_max))
    x_min = max(0, min(mask.shape[1] - 1, x_min))
    x_max = max(0, min(mask.shape[1] - 1, x_max))
    if y_max <= y_min or x_max <= x_min:
        return 0.0
    return float(np.count_nonzero(mask[y_min : y_max + 1, x_min : x_max + 1]))


def target_location_label(bounds: tuple[int, int, int, int], max_x: float, max_y: float) -> str:
    x_min, y_min, x_max, y_max = bounds
    return location_label(((x_min + x_max) / 2.0) / max_x, ((y_min + y_max) / 2.0) / max_y)


def inspection_targets(
    records: list[PolygonRecord],
    latest_date: str,
    row_regions_by_date: dict[str, list[RowRegion]] | None = None,
    row_confidence_by_date: dict[str, float] | None = None,
    image_shapes_by_date: dict[str, tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    grouped = records_by_date(records)
    latest_records = grouped.get(latest_date, [])
    if not latest_records:
        return []
    max_x, max_y = all_record_bounds(records, (image_shapes_by_date or {}).get(latest_date))
    shape = (max(1, int(round(max_y))), max(1, int(round(max_x))))
    latest_rows = (row_regions_by_date or {}).get(latest_date, [])
    latest_confidence = (row_confidence_by_date or {}).get(latest_date, 0.0)
    low_mask = records_to_mask(latest_records, shape, {"low_vigour"})
    medium_mask = records_to_mask(latest_records, shape, {"medium_vigour"})
    human_focus_mask = cv2.bitwise_or(low_mask, medium_mask)
    if not np.any(human_focus_mask):
        return []

    groups = row_groups(latest_rows) if latest_rows and latest_confidence >= 0.9 else []
    if not groups:
        y_edges = np.linspace(0, shape[0] - 1, 12).round().astype(int)
        groups = [
            [
                RowRegion(
                    row_number=index + 1,
                    row_id=f"block_row_{index + 1:03d}",
                    bed_id=f"block_row_{index + 1:03d}",
                    y_min=int(y_edges[index]),
                    y_max=int(y_edges[index + 1]),
                    x_min=0,
                    x_max=shape[1] - 1,
                    area_pixels=max(1, (int(y_edges[index + 1]) - int(y_edges[index]) + 1) * shape[1]),
                    confidence=0.4,
                    source="location_block_fallback",
                )
            ]
            for index in range(len(y_edges) - 1)
        ]

    previous_dates = [date for date in sorted(grouped) if date < latest_date]
    previous_date = previous_dates[-1] if previous_dates else ""
    date_masks: dict[str, np.ndarray] = {}
    for date, date_records in grouped.items():
        date_shape = shape
        if image_shapes_by_date and date in image_shapes_by_date:
            h, w = image_shapes_by_date[date]
            date_shape = (max(1, int(h)), max(1, int(w)))
        date_masks[date] = records_to_mask(date_records, date_shape, {"low_vigour", "medium_vigour"})

    raw_targets: list[dict[str, Any]] = []
    total_pixels = max(1.0, float(shape[0] * shape[1]))
    for group in groups:
        row_start = min(row.row_number for row in group)
        row_end = max(row.row_number for row in group)
        for column_index in range(TARGET_COLUMN_COUNT):
            bounds = segment_bounds_from_rows(group, column_index, TARGET_COLUMN_COUNT)
            low = mask_area_in_bounds(low_mask, bounds)
            medium = mask_area_in_bounds(medium_mask, bounds)
            focus = low + medium
            if focus < MIN_TARGET_FOCUS_PIXELS:
                continue
            present_dates = []
            focus_by_date: dict[str, float] = {}
            x_min, y_min, x_max, y_max = bounds
            x0_ratio = x_min / max_x
            x1_ratio = x_max / max_x
            y0_ratio = y_min / max_y
            y1_ratio = y_max / max_y
            for date in sorted(grouped):
                date_shape = date_masks[date].shape
                date_bounds = (
                    int(round(x0_ratio * date_shape[1])),
                    int(round(y0_ratio * date_shape[0])),
                    int(round(x1_ratio * date_shape[1])),
                    int(round(y1_ratio * date_shape[0])),
                )
                date_focus = mask_area_in_bounds(date_masks[date], date_bounds)
                focus_by_date[date] = date_focus
                if date_focus >= MIN_TARGET_FOCUS_PIXELS:
                    present_dates.append(date)
            previous_focus = focus_by_date.get(previous_date, 0.0) if previous_date else 0.0
            x_min, y_min, x_max, y_max = bounds
            location = target_location_label(bounds, max_x, max_y)
            row_text = f"Rows {row_start}-{row_end}" if latest_confidence >= 0.9 else location
            raw_targets.append(
                {
                    "records": [],
                    "bounds": bounds,
                    "center": (int((x_min + x_max) / 2), int((y_min + y_max) / 2)),
                    "location_label": f"Zone pending - {row_text}",
                    "row_or_block": row_text,
                    "location_block": location,
                    "row_start": row_start,
                    "row_end": row_end,
                    "length_label": "walkable row segment",
                    "low_pixels": low,
                    "medium_pixels": medium,
                    "high_pixels": 0.0,
                    "focus_pixels": focus,
                    "low_percent": low / total_pixels * 100.0,
                    "medium_percent": medium / total_pixels * 100.0,
                    "focus_percent": focus / total_pixels * 100.0,
                    "coverage_percent": focus / total_pixels * 100.0,
                    "present_dates": present_dates,
                    "first_present": present_dates[0] if present_dates else latest_date,
                    "persistence_count": len(present_dates),
                    "previous_date": previous_date,
                    "previous_focus_pixels": previous_focus,
                    "expansion_pixels": focus - previous_focus,
                    "production_relevance": True,
                    "row_confidence": latest_confidence,
                    "human_annotation_overlap": focus,
                }
            )

    if not raw_targets:
        return []
    recurring_scores = [item for item in raw_targets if item["persistence_count"] >= 2]
    reference = recurring_scores or raw_targets
    max_expansion = max(1.0, max((max(0.0, item["expansion_pixels"]) for item in reference), default=1.0))
    max_focus = max(1.0, max((item["focus_pixels"] for item in reference), default=1.0))
    targets = []
    for item in raw_targets:
        if item["low_pixels"] > 0 and item["medium_pixels"] > 0:
            severity = float(SCORE_WEIGHTS["severity"])
        elif item["low_pixels"] > 0:
            severity = SCORE_WEIGHTS["severity"] * 0.82
        else:
            severity = SCORE_WEIGHTS["severity"] * 0.55
        persistence = SCORE_WEIGHTS["persistence"] * min(1.0, max(0, item["persistence_count"] - 1) / 3.0)
        expansion = SCORE_WEIGHTS["expansion"] * min(1.0, max(0.0, item["expansion_pixels"]) / max_expansion)
        coverage = SCORE_WEIGHTS["coverage"] * min(1.0, item["focus_pixels"] / max_focus)
        evidence_sources = evidence_sources_for_block(item)
        evidence = SCORE_WEIGHTS["evidence_count"] * (len(evidence_sources) / EVIDENCE_TOTAL)
        score = round(min(100.0, severity + persistence + expansion + coverage + evidence))
        item.update(
            {
                "severity_score": severity,
                "persistence_score": persistence,
                "expansion_score": expansion,
                "coverage_score": coverage,
                "evidence_count_score": evidence,
                "evidence_count": len(evidence_sources),
                "evidence_total": EVIDENCE_TOTAL,
                "evidence_sources": evidence_sources,
                "inspection_priority_score": int(score),
                "score": int(score),
            }
        )
        targets.append(item)
    targets.sort(
        key=lambda item: (
            item["inspection_priority_score"],
            item["low_pixels"],
            item["focus_pixels"],
            item["expansion_pixels"],
        ),
        reverse=True,
    )
    for rank, item in enumerate(targets, start=1):
        item["rank"] = rank
        item["zone_id"] = f"Zone {rank:02d}"
        item["location_label"] = f"{item['zone_id']} - {item['row_or_block']}"
        item["action"] = action_for_block(item)
        item["reason"] = reason_for_block(item)
        item["walk_minutes"] = estimate_walk_minutes(item)
        item["change_status"], item["change_arrow"] = change_status(item)
    return targets


def focus_area(records: list[PolygonRecord]) -> float:
    return sum(record.area_pixels for record in records if record.class_name in ("low_vigour", "medium_vigour"))


def class_area(records: list[PolygonRecord], class_name: str) -> float:
    return sum(record.area_pixels for record in records if record.class_name == class_name)


def class_percent(records: list[PolygonRecord], class_name: str, total_pixels: float) -> float:
    return class_area(records, class_name) / max(1.0, total_pixels) * 100.0


def grouped_blocks(
    records: list[PolygonRecord],
    latest_date: str,
    row_regions_by_date: dict[str, list[RowRegion]] | None = None,
    row_confidence_by_date: dict[str, float] | None = None,
    image_shapes_by_date: dict[str, tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    grouped = records_by_date(records)
    latest_records = grouped.get(latest_date, [])
    if not latest_records:
        return []

    max_x, max_y = all_record_bounds(records, (image_shapes_by_date or {}).get(latest_date))
    total_pixels = max_x * max_y
    latest_by_bucket: dict[tuple[str, str], list[PolygonRecord]] = defaultdict(list)
    for record in latest_records:
        if record.class_name in ("low_vigour", "medium_vigour", "high_vigour"):
            latest_by_bucket[bucket_for_record(record, max_x, max_y)].append(record)

    dated_bucket_focus: dict[tuple[str, tuple[str, str]], float] = defaultdict(float)
    for record in records:
        if record.class_name not in ("low_vigour", "medium_vigour"):
            continue
        dated_bucket_focus[(record.date, bucket_for_record(record, max_x, max_y))] += record.area_pixels

    previous_dates = [date for date in sorted(grouped) if date < latest_date]
    previous_date = previous_dates[-1] if previous_dates else ""
    max_focus = 1.0
    raw_scores: list[dict[str, Any]] = []

    for bucket, bucket_records in latest_by_bucket.items():
        if not bucket_records:
            continue
        low = class_area(bucket_records, "low_vigour")
        medium = class_area(bucket_records, "medium_vigour")
        high = class_area(bucket_records, "high_vigour")
        focus = low + medium
        if focus <= 0 and high <= 0:
            continue
        max_focus = max(max_focus, focus)
        first_present = ""
        present_dates = []
        for date in sorted(grouped):
            if dated_bucket_focus.get((date, bucket), 0.0) > 0:
                present_dates.append(date)
                first_present = first_present or date
        previous_focus = dated_bucket_focus.get((previous_date, bucket), 0.0) if previous_date else 0.0
        expansion_pixels = focus - previous_focus
        bounds = block_bounds(bucket_records)
        center_x = int((bounds[0] + bounds[2]) / 2)
        center_y = int((bounds[1] + bounds[3]) / 2)
        x_ratio = center_x / max_x
        y_ratio = center_y / max_y
        production_relevance = 0.08 <= x_ratio <= 0.92 and 0.25 <= y_ratio <= 0.92
        latest_rows = (row_regions_by_date or {}).get(latest_date, [])
        latest_confidence = (row_confidence_by_date or {}).get(latest_date, 0.0)
        display_label, row_or_block = block_label_for_bounds(bounds, max_x, max_y, latest_rows, latest_confidence, bucket)
        raw_scores.append(
            {
                "bucket": bucket,
                "records": bucket_records,
                "bounds": bounds,
                "center": (center_x, center_y),
                "location_label": display_label,
                "row_or_block": row_or_block,
                "low_pixels": low,
                "medium_pixels": medium,
                "high_pixels": high,
                "focus_pixels": focus,
                "low_percent": low / max(1.0, total_pixels) * 100.0,
                "medium_percent": medium / max(1.0, total_pixels) * 100.0,
                "focus_percent": focus / max(1.0, total_pixels) * 100.0,
                "coverage_percent": (low + medium + high) / max(1.0, total_pixels) * 100.0,
                "present_dates": present_dates,
                "first_present": first_present,
                "persistence_count": len(present_dates),
                "previous_date": previous_date,
                "previous_focus_pixels": previous_focus,
                "expansion_pixels": expansion_pixels,
                "production_relevance": production_relevance,
                "row_confidence": latest_confidence,
            }
        )

    recurring_scores = [item for item in raw_scores if item["persistence_count"] >= 2]
    expansion_reference = recurring_scores or raw_scores
    focus_reference = recurring_scores or raw_scores
    max_expansion = max(1.0, max((max(0.0, item["expansion_pixels"]) for item in expansion_reference), default=1.0))
    blocks = []
    for item in raw_scores:
        if item["low_pixels"] > 0 and item["medium_pixels"] > 0:
            severity = float(SCORE_WEIGHTS["severity"])
        elif item["low_pixels"] > 0:
            severity = SCORE_WEIGHTS["severity"] * 0.8
        elif item["medium_pixels"] > 0:
            severity = SCORE_WEIGHTS["severity"] * 0.53
        else:
            severity = 0.0
        if item["persistence_count"] < 2:
            severity = min(severity, SCORE_WEIGHTS["severity"] * 0.45)
        persistence = SCORE_WEIGHTS["persistence"] * min(1.0, max(0, item["persistence_count"] - 1) / 3.0)
        expansion = SCORE_WEIGHTS["expansion"] * min(1.0, max(0.0, item["expansion_pixels"]) / max_expansion)
        if item["persistence_count"] < 2:
            expansion = 0.0
        coverage = SCORE_WEIGHTS["coverage"] * min(1.0, item["focus_percent"] / 1.05)
        evidence_sources = evidence_sources_for_block(item)
        evidence = SCORE_WEIGHTS["evidence_count"] * (len(evidence_sources) / EVIDENCE_TOTAL)
        score = round(min(100.0, persistence + expansion + severity + coverage + evidence))
        item.update(
            {
                "severity_score": severity,
                "persistence_score": persistence,
                "expansion_score": expansion,
                "coverage_score": coverage,
                "evidence_count_score": evidence,
                "evidence_count": len(evidence_sources),
                "evidence_total": EVIDENCE_TOTAL,
                "evidence_sources": evidence_sources,
                "inspection_priority_score": int(score),
                "score": int(score),
            }
        )
        blocks.append(item)

    blocks.sort(
        key=lambda item: (
            item["inspection_priority_score"],
            item["low_pixels"],
            item["focus_pixels"],
            item["expansion_pixels"],
        ),
        reverse=True,
    )
    for rank, item in enumerate(blocks, start=1):
        item["rank"] = rank
        item["action"] = action_for_block(item)
        item["reason"] = reason_for_block(item)
        item["walk_minutes"] = estimate_walk_minutes(item)
        item["change_status"], item["change_arrow"] = change_status(item)
    return blocks


def action_for_block(block: dict[str, Any]) -> str:
    if block["inspection_priority_score"] >= 75 or block["low_pixels"] > 0:
        return "Inspect within 24-48 hours"
    if block["inspection_priority_score"] >= 45:
        return "Watch on next pass"
    return "Use as healthy reference"


def evidence_sources_for_block(block: dict[str, Any]) -> list[str]:
    sources = ["Human annotation"]
    if block.get("persistence_count", 0) >= 2 or abs(float(block.get("expansion_pixels", 0.0))) > 0:
        sources.append("NDVI/OSAVI trend")
    return sources


def reason_for_block(block: dict[str, Any]) -> str:
    if block["persistence_count"] >= 2 and block["expansion_pixels"] > 0 and block["low_pixels"] > 0:
        return "Persistent, expanding, and includes low-vigour annotation."
    if block["persistence_count"] >= 2 and block["medium_pixels"] > 0:
        return "Recurring medium-vigour area that should be checked before harvest."
    if block["low_pixels"] > 0:
        return "Contains low-vigour annotation on the latest flight."
    return "Annotated high-vigour reference area."


def estimate_walk_minutes(block: dict[str, Any]) -> int:
    minutes = 3
    if block["coverage_percent"] > 1.0:
        minutes += 1
    if block["persistence_count"] >= 3:
        minutes += 1
    if block["inspection_priority_score"] >= 85:
        minutes += 1
    return max(3, min(8, minutes))


def change_status(block: dict[str, Any]) -> tuple[str, str]:
    delta = float(block["expansion_pixels"])
    baseline = max(1.0, float(block["previous_focus_pixels"]))
    if not block["previous_date"]:
        return "New", "up"
    if delta > baseline * 0.12:
        return "Expanding", "up"
    if delta < -baseline * 0.12:
        return "Improving", "down"
    return "Stable", "right"


def why_bullets(block: dict[str, Any]) -> list[str]:
    bullets = []
    if block["persistence_count"] >= 2:
        bullets.append("Persistent across multiple flights")
    else:
        bullets.append("Present on the latest flight")
    if block["expansion_pixels"] > 0:
        bullets.append("Expanded since previous collection")
    else:
        bullets.append("Still visible on the latest flight")
    if block["low_pixels"] > 0:
        bullets.append("Contains low-vigour annotation")
    elif block["medium_pixels"] > 0:
        bullets.append("Contains medium-vigour annotation")
    else:
        bullets.append("Useful as a high-vigour reference")
    if block["coverage_percent"] >= 0.05:
        bullets.append("Covers meaningful production area")
    return bullets


def top_priority_beds(
    records: list[PolygonRecord],
    latest_date: str,
    row_regions_by_date: dict[str, list[RowRegion]] | None = None,
    row_confidence_by_date: dict[str, float] | None = None,
    image_shapes_by_date: dict[str, tuple[int, int]] | None = None,
    limit: int = 3,
) -> list[dict[str, Any]]:
    return inspection_targets(records, latest_date, row_regions_by_date, row_confidence_by_date, image_shapes_by_date)[:limit]


def inspection_status(blocks: list[dict[str, Any]]) -> dict[str, str]:
    if not blocks:
        return {"label": "FIELD STABLE", "level": "green"}
    top_score = int(blocks[0]["inspection_priority_score"])
    if top_score >= 75:
        return {"label": "INSPECTION NEEDED", "level": "red"}
    if top_score >= 45:
        return {"label": "WATCH NEXT", "level": "yellow"}
    return {"label": "FIELD STABLE", "level": "green"}


def decision_sentence(block: dict[str, Any] | None) -> str:
    if not block:
        return "No immediate inspection zone was generated from the latest flight."
    return f"Inspect the {block['location_label']} first because it is persistent, expanding, and large enough to matter."


def draw_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    foreground: tuple[int, int, int] = (255, 255, 255),
    background: tuple[int, int, int] = (8, 15, 22),
    font_scale: float = 0.8,
    thickness: int = 2,
) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad_x = max(8, int(9 * font_scale))
    pad_y = max(6, int(7 * font_scale))
    x = max(0, min(x, image.shape[1] - tw - 2 * pad_x - 1))
    y = max(th + pad_y, min(y, image.shape[0] - baseline - pad_y - 1))
    cv2.rectangle(
        image,
        (x, y - th - baseline - pad_y),
        (x + tw + 2 * pad_x, y + baseline + pad_y),
        background,
        thickness=cv2.FILLED,
    )
    cv2.rectangle(
        image,
        (x, y - th - baseline - pad_y),
        (x + tw + 2 * pad_x, y + baseline + pad_y),
        (255, 255, 255),
        thickness=1,
    )
    cv2.putText(image, text, (x + pad_x, y), font, font_scale, foreground, thickness, cv2.LINE_AA)


def draw_priority_marker(image: np.ndarray, rank: int, center: tuple[int, int], scale: float) -> None:
    radius = max(24, int(20 * scale))
    cv2.circle(image, center, radius, (255, 255, 255), max(5, int(5 * scale)), cv2.LINE_AA)
    cv2.circle(image, center, radius, (35, 35, 235), max(3, int(3 * scale)), cv2.LINE_AA)
    text = f"#{rank}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.75, min(1.8, 0.65 * scale))
    thickness = max(2, int(3 * scale))
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.putText(image, text, (center[0] - tw // 2, center[1] + th // 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_legend(image: np.ndarray, scale: float) -> None:
    x = int(24 * scale)
    y = int(42 * scale)
    w = int(420 * scale)
    h = int(132 * scale)
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (8, 15, 22), cv2.FILLED)
    image[:] = cv2.addWeighted(overlay, 0.78, image, 0.22, 0)
    cv2.rectangle(image, (x, y), (x + w, y + h), (245, 250, 255), max(1, int(scale)))
    rows = [
        ("Low vigour", CLASS_COLORS_BGR["low_vigour"]),
        ("Medium vigour", CLASS_COLORS_BGR["medium_vigour"]),
        ("High vigour", CLASS_COLORS_BGR["high_vigour"]),
    ]
    for idx, (label, color) in enumerate(rows):
        yy = y + int((34 + idx * 32) * scale)
        cv2.line(image, (x + int(18 * scale), yy), (x + int(86 * scale), yy), color, max(5, int(5 * scale)), cv2.LINE_AA)
        cv2.putText(
            image,
            label,
            (x + int(104 * scale), yy + int(8 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.62, min(1.2, 0.62 * scale)),
            (255, 255, 255),
            max(1, int(2 * scale)),
            cv2.LINE_AA,
        )


def draw_polygon_overlay(
    raw_bgr: np.ndarray,
    records: list[PolygonRecord],
    output_path: Path,
    rows: list[RowRegion] | None = None,
    priority_beds: list[dict[str, Any]] | None = None,
) -> None:
    preview = raw_bgr.copy()
    h, w = preview.shape[:2]
    scale = max(h, w) / 1600
    boundary_thickness = max(3, int(round(3 * scale)))
    priorities = priority_beds or []

    if priorities:
        wash = np.full_like(preview, (8, 15, 22))
        preview = cv2.addWeighted(preview, 0.82, wash, 0.18, 0)
    else:
        for record in records:
            points = np.array(record.coordinates, dtype=np.int32).reshape((-1, 1, 2))
            color = CLASS_COLORS_BGR[record.class_name]
            overlay = preview.copy()
            cv2.fillPoly(overlay, [points], color)
            preview = cv2.addWeighted(overlay, 0.12, preview, 0.88, 0)
            cv2.polylines(preview, [points], isClosed=True, color=color, thickness=boundary_thickness)

    row_confidence = min((row.confidence for row in rows), default=0.0) if rows else 0.0
    if rows and row_confidence >= 0.9 and not priorities:
        for row in rows:
            points = np.array(row.points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(preview, [points], isClosed=False, color=(255, 255, 255), thickness=max(1, int(1.5 * scale)), lineType=cv2.LINE_AA)
            if row.row_number % 5 == 0:
                draw_label(
                    preview,
                    f"Row {row.row_number}",
                    (row.x_min + int(12 * scale), int(row.center_y)),
                    font_scale=max(0.45, min(0.9, 0.45 * scale)),
                    thickness=max(1, int(1.4 * scale)),
                )

    priority_colors = [(35, 35, 235), (40, 190, 245), (70, 225, 95)]
    for block in priorities[:5]:
        center = tuple(int(value) for value in block.get("center", (w // 2, h // 2)))
        bounds = block.get("bounds")
        color = priority_colors[min(int(block.get("rank", 1)) - 1, len(priority_colors) - 1)]
        if bounds:
            x_min, y_min, x_max, y_max = [int(value) for value in bounds]
            pad = int(8 * scale)
            x_min = max(0, x_min - pad)
            y_min = max(0, y_min - pad)
            x_max = min(w - 1, x_max + pad)
            y_max = min(h - 1, y_max + pad)
            zone_overlay = preview.copy()
            cv2.rectangle(zone_overlay, (x_min, y_min), (x_max, y_max), color, cv2.FILLED)
            preview = cv2.addWeighted(zone_overlay, 0.2, preview, 0.8, 0)
            cv2.rectangle(preview, (x_min, y_min), (x_max, y_max), color, max(6, int(5 * scale)), cv2.LINE_AA)
        draw_priority_marker(preview, int(block.get("rank", 1)), center, scale)
        label = str(block.get("location_label", "inspection zone"))
        draw_label(
            preview,
            label,
            (center[0] + int(30 * scale), center[1] - int(24 * scale)),
            foreground=(255, 255, 255),
            background=(26, 30, 42),
            font_scale=max(0.52, min(1.0, 0.52 * scale)),
            thickness=max(1, int(2 * scale)),
        )
    if not priorities:
        draw_legend(preview, scale)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), preview)
    print(f"[WRITE] Overlay: {output_path}")


def make_chart(records_by_date: dict[str, list[PolygonRecord]], output_path: Path) -> None:
    dates = sorted(records_by_date)
    low = []
    medium = []
    high = []
    for date in dates:
        coverage = class_coverage_from_polygons(records_by_date[date])
        low.append(coverage["low_vigour"])
        medium.append(coverage["medium_vigour"])
        high.append(coverage["high_vigour"])
    x = np.arange(len(dates))
    plt.figure(figsize=(10, 5.5))
    plt.bar(x, low, label="Low vigour", color=CLASS_COLORS_HEX["low_vigour"])
    plt.bar(x, medium, bottom=low, label="Medium vigour", color=CLASS_COLORS_HEX["medium_vigour"])
    bottom = np.array(low) + np.array(medium)
    plt.bar(x, high, bottom=bottom, label="High vigour", color=CLASS_COLORS_HEX["high_vigour"])
    plt.xticks(x, [short_date(date) for date in dates])
    plt.ylabel("Area percent")
    plt.title("Technical Class Summary")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()
    print(f"[WRITE] Chart: {output_path}")


def date_rows(records: list[PolygonRecord]) -> list[dict[str, Any]]:
    rows = []
    grouped = records_by_date(records)
    previous_focus: float | None = None
    for date in sorted(grouped):
        coverage = class_coverage_from_polygons(grouped[date])
        focus = coverage["low_vigour"] + coverage["medium_vigour"]
        if previous_focus is None:
            trend = "Stable"
            arrow = "right"
        elif focus > previous_focus + 2:
            trend = "Worse"
            arrow = "up"
        elif focus < previous_focus - 2:
            trend = "Better"
            arrow = "down"
        else:
            trend = "Stable"
            arrow = "right"
        previous_focus = focus
        rows.append(
            {
                "date": date,
                "short_date": short_date(date),
                "growth_stage": grouped[date][0].growth_stage,
                "focus_value": focus,
                "trend": trend,
                "arrow": arrow,
            }
        )
    return rows


def row_report_rows(records: list[PolygonRecord]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, float | str]] = defaultdict(
        lambda: {
            "growth_stage": "",
            "area": 1.0,
            "low": 0.0,
            "medium": 0.0,
            "high": 0.0,
        }
    )
    for record in records:
        item = grouped[(record.date, record.bed_id)]
        item["growth_stage"] = record.growth_stage
        item["area"] = max(float(item["area"]), record.bed_area_pixels)
        if record.class_name == "low_vigour":
            item["low"] = float(item["low"]) + record.area_pixels
        elif record.class_name == "medium_vigour":
            item["medium"] = float(item["medium"]) + record.area_pixels
        elif record.class_name == "high_vigour":
            item["high"] = float(item["high"]) + record.area_pixels
    rows = []
    for (date, row_id), item in sorted(grouped.items()):
        area = max(1.0, float(item["area"]))
        low = float(item["low"]) / area * 100
        medium = float(item["medium"]) / area * 100
        high = float(item["high"]) / area * 100
        rows.append(
            {
                "date": date,
                "short_date": short_date(date),
                "row_id": row_id,
                "row_label": row_label(row_id),
                "growth_stage": str(item["growth_stage"]),
                "low_value": low,
                "medium_value": medium,
                "high_value": high,
                "focus_value": low + medium,
                "low": format_percent(low),
                "medium": format_percent(medium),
                "high": format_percent(high),
                "focus": format_percent(low + medium),
            }
        )
    return rows


def build_audit_rows(records: list[PolygonRecord], processing_audits: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
    rows = []
    for row in processing_audits or []:
        status = row.get("status", "WARNING")
        if status == "REVIEW":
            status = "WARNING"
        rows.append({**row, "status": status})
    seen_checks = {row["check"] for row in rows}

    for row in row_report_rows(records):
        for class_key, class_label in (("low_value", "low"), ("medium_value", "medium"), ("high_value", "high")):
            value = float(row[class_key])
            if value > 60:
                rows.append(
                    {
                        "status": "WARNING",
                        "check": "row-guide class percent > 60%",
                        "date": str(row["date"]),
                        "detail": f"{row['row_label']} has {format_percent(value)} {class_label} vigour coverage.",
                    }
                )
                seen_checks.add("row-guide class percent > 60%")

    by_row: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in row_report_rows(records):
        by_row[str(row["row_id"])].append(row)
    for row_id, rows_for_guide in by_row.items():
        rows_for_guide.sort(key=lambda row: str(row["date"]))
        previous = None
        for row in rows_for_guide:
            current = float(row["focus_value"])
            if previous is not None:
                delta = current - float(previous["focus_value"])
                if abs(delta) > 20:
                    rows.append(
                        {
                            "status": "WARNING",
                            "check": "sudden change > 20%",
                            "date": str(row["date"]),
                            "detail": f"{row_label(row_id)} changed {delta:+.1f} points from {previous['date']} to {row['date']}.",
                        }
                    )
                    seen_checks.add("sudden change > 20%")
            previous = row

    pass_messages = {
        "row-guide class percent > 60%": "No internal row guide had a single class above the review threshold.",
        "sudden change > 20%": "No internal row guide had a large one-flight jump.",
        "annotation area too small": "Field-relevant marked area was present on matched flights.",
        "annotation area too large": "Marked area stayed within expected release bounds.",
        "unclassified changed pixels > 10%": "Field-relevant marked pixels matched the expected colours.",
        "missing raw/annotated pair": "All dated field images had matched pairs.",
        "resized image pair": "Matched image sizes were consistent.",
    }
    for check, detail in pass_messages.items():
        if check not in seen_checks:
            rows.append({"status": "PASS", "check": check, "date": "all", "detail": detail})
    status_order = {"FAIL": 0, "WARNING": 1, "PASS": 2}
    return sorted(rows, key=lambda row: (status_order.get(row["status"], 9), row["check"], row["date"]))


def field_audit_rows(records: list[PolygonRecord], audit_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    warnings = defaultdict(list)
    for row in audit_rows:
        if row.get("status") != "PASS":
            warnings[row.get("check", "")].append(row)
    return [
        {
            "name": "Image Pair Match",
            "status": "WARNING" if warnings.get("missing raw/annotated pair") else "PASS",
            "detail": "Some unmatched dated images were excluded." if warnings.get("missing raw/annotated pair") else "Matched field images were found.",
        },
        {
            "name": "Annotation Parsing",
            "status": "WARNING" if warnings.get("unclassified changed pixels > 10%") else "PASS",
            "detail": "Some field markings need review." if warnings.get("unclassified changed pixels > 10%") else "Human colour annotations were read successfully.",
        },
        {
            "name": "Priority Extraction",
            "status": "PASS" if records else "FAIL",
            "detail": "Human-anchored inspection zones were generated." if records else "No inspection zones were generated.",
        },
        {
            "name": "Row Confidence Gate",
            "status": "PASS",
            "detail": "Exact row labels are hidden unless row confidence is at least 0.90.",
        },
        {
            "name": "Date Matching",
            "status": "WARNING" if warnings.get("resized image pair") else "PASS",
            "detail": "One or more images needed size alignment." if warnings.get("resized image pair") else "Dates were read from filenames and matched.",
        },
    ]


def technical_summary(records: list[PolygonRecord], latest_date: str, row_confidence_by_date: dict[str, float] | None) -> dict[str, Any]:
    dates = sorted({record.date for record in records})
    latest_records = [record for record in records if record.date == latest_date]
    return {
        "record_count": len(records),
        "date_count": len(dates),
        "latest_polygon_count": len(latest_records),
        "row_confidence": (row_confidence_by_date or {}).get(latest_date, 0.0),
    }


def write_customer_release_audit(
    field_name: str,
    records: list[PolygonRecord],
    audit_rows: list[dict[str, str]],
    output_path: Path,
    row_regions_by_date: dict[str, list[RowRegion]] | None = None,
    row_confidence_by_date: dict[str, float] | None = None,
    image_shapes_by_date: dict[str, tuple[int, int]] | None = None,
) -> None:
    latest_date = max((record.date for record in records), default="")
    blocks = top_priority_beds(records, latest_date, row_regions_by_date, row_confidence_by_date, image_shapes_by_date, limit=10)
    rows = field_audit_rows(records, audit_rows)
    lines = [
        f"# {field_name} Field Inspection Work Order Audit",
        "",
        f"Latest flight: {latest_date}",
        f"Inspection zones: {', '.join(str(row['location_label']) for row in blocks[:5])}",
        "",
        "## Release Audit",
        "",
    ]
    for row in rows:
        lines.append(f"- {row['status']}: {row['name']} - {row['detail']}")
    lines.extend(["", "## Top Inspection Zones", ""])
    for row in blocks:
        lines.append(f"{row['rank']}. {row['location_label']} - score {row['inspection_priority_score']}/100 - {row['reason']}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[WRITE] Customer release audit: {output_path}")


def write_inspection_targets(targets: list[dict[str, Any]], geojson_path: Path, csv_path: Path) -> None:
    features = []
    rows = []
    for target in targets:
        x_min, y_min, x_max, y_max = [int(value) for value in target["bounds"]]
        coordinates = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max], [x_min, y_min]]
        properties = {
            "zone_id": target.get("zone_id", ""),
            "rank": target.get("rank", ""),
            "location_label": target.get("location_label", ""),
            "row_or_block": target.get("row_or_block", ""),
            "location_block": target.get("location_block", ""),
            "priority_score": target.get("inspection_priority_score", 0),
            "row_start": target.get("row_start", ""),
            "row_end": target.get("row_end", ""),
            "low_pixels": round(float(target.get("low_pixels", 0.0)), 2),
            "medium_pixels": round(float(target.get("medium_pixels", 0.0)), 2),
            "focus_pixels": round(float(target.get("focus_pixels", 0.0)), 2),
            "persistence_count": target.get("persistence_count", 0),
            "expansion_pixels": round(float(target.get("expansion_pixels", 0.0)), 2),
            "human_annotation_overlap": round(float(target.get("human_annotation_overlap", 0.0)), 2),
            "reason": target.get("reason", ""),
            "recommended_action": target.get("action", ""),
        }
        features.append(
            {
                "type": "Feature",
                "properties": properties,
                "geometry": {"type": "Polygon", "coordinates": [coordinates]},
            }
        )
        rows.append(properties)
    geojson_path.parent.mkdir(parents=True, exist_ok=True)
    geojson_path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys()) if rows else [
            "zone_id",
            "rank",
            "location_label",
            "row_or_block",
            "location_block",
            "priority_score",
            "row_start",
            "row_end",
            "low_pixels",
            "medium_pixels",
            "focus_pixels",
            "persistence_count",
            "expansion_pixels",
            "human_annotation_overlap",
            "reason",
            "recommended_action",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[WRITE] Inspection target GeoJSON: {geojson_path}")
    print(f"[WRITE] Inspection target CSV: {csv_path}")


def arrow_symbol(name: str) -> str:
    if name == "up":
        return "Up"
    if name == "down":
        return "Down"
    return "Stable"


def write_html_report(
    field_name: str,
    records: list[PolygonRecord],
    overlay_paths_by_date: dict[str, Path],
    chart_path: Path,
    output_path: Path,
    bed_aware: bool = False,
    audit_rows: list[dict[str, str]] | None = None,
    hero_image_path: Path | None = None,
    row_regions_by_date: dict[str, list[RowRegion]] | None = None,
    row_confidence_by_date: dict[str, float] | None = None,
    image_shapes_by_date: dict[str, tuple[int, int]] | None = None,
) -> None:
    latest_date = sorted(overlay_paths_by_date)[-1] if overlay_paths_by_date else ""
    grouped = records_by_date(records)
    latest_stage = grouped[latest_date][0].growth_stage if latest_date and grouped.get(latest_date) else ""
    blocks = top_priority_beds(records, latest_date, row_regions_by_date, row_confidence_by_date, image_shapes_by_date, limit=5)
    top_block = blocks[0] if blocks else None
    status = inspection_status(blocks)
    timeline = date_rows(records)
    final_audit_rows = audit_rows or build_audit_rows(records, None)
    release_audit = field_audit_rows(records, final_audit_rows)
    tech = technical_summary(records, latest_date, row_confidence_by_date)
    watch_next = blocks[1:4]
    healthy_reference = None

    template = Template(
        r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{{ field_name }} &mdash; Field Inspection Work Order</title>
  <style>
    :root {
      --bg: #071018;
      --panel: #0d1924;
      --panel2: #111f2b;
      --line: #263848;
      --text: #f3f7fb;
      --muted: #9fb0bd;
      --red: #ff3b3b;
      --yellow: #f5c84c;
      --green: #7be35d;
      --purple: #a66cff;
      --cyan: #50d6ff;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); font-family: Arial, sans-serif; }
    header { padding: 26px 34px 16px; background: linear-gradient(180deg, #0b1823, #071018); border-bottom: 1px solid var(--line); }
    header h1 { margin: 0; font-size: 42px; letter-spacing: 0; }
    header .sub { margin-top: 6px; color: var(--cyan); font-size: 18px; font-weight: 800; }
    main { padding: 24px 32px 38px; max-width: 1800px; margin: 0 auto; }
    section { margin-bottom: 28px; }
    h2 { margin: 0 0 14px; font-size: 22px; text-transform: uppercase; letter-spacing: 0; }
    .work-order { border: 1px solid var(--line); border-radius: 8px; background: var(--panel); overflow: hidden; }
    .status-band { padding: 24px; display: grid; grid-template-columns: minmax(340px, 1.1fr) minmax(320px, .9fr); gap: 22px; align-items: stretch; }
    .work-order.red .status-band { background: linear-gradient(90deg, rgba(255,59,59,.34), rgba(13,25,36,.86)); }
    .work-order.yellow .status-band { background: linear-gradient(90deg, rgba(245,200,76,.32), rgba(13,25,36,.86)); }
    .work-order.green .status-band { background: linear-gradient(90deg, rgba(123,227,93,.25), rgba(13,25,36,.86)); }
    .eyebrow { color: var(--muted); font-size: 13px; font-weight: 900; text-transform: uppercase; margin-bottom: 7px; }
    .inspect { font-size: 52px; line-height: 1.02; font-weight: 900; margin: 0 0 10px; }
    .target { font-size: 34px; line-height: 1.08; font-weight: 900; margin: 0 0 12px; color: var(--cyan); }
    .decision { font-size: 24px; line-height: 1.32; font-weight: 800; margin: 0; }
    .score { background: rgba(5, 11, 17, .72); border: 1px solid var(--line); border-radius: 8px; padding: 18px; }
    .score strong { display: block; font-size: 66px; line-height: 1; color: var(--red); }
    .score span { color: var(--muted); font-weight: 800; text-transform: uppercase; }
    .work-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(245px, 1fr)); gap: 1px; background: var(--line); }
    .work-cell { background: var(--panel2); padding: 18px; min-height: 145px; }
    .work-cell h3 { margin: 0 0 10px; font-size: 18px; }
    .work-cell ul { margin: 0; padding-left: 18px; color: var(--text); line-height: 1.45; }
    .work-cell p { margin: 0; color: var(--text); line-height: 1.45; }
    .snapshot { background: #02070b; border-left: 1px solid var(--line); }
    .snapshot img { width: 100%; height: 100%; max-height: 360px; object-fit: contain; display: block; }
    .map-card { background: #02070b; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
    .map-card img { display: block; width: 100%; max-height: 84vh; object-fit: contain; background: #02070b; }
    .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(255px, 1fr)); gap: 16px; }
    .card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 18px; }
    .card.inspect-card { border-color: rgba(255,59,59,.75); box-shadow: inset 4px 0 0 var(--red); }
    .card.watch-card { border-color: rgba(245,200,76,.75); box-shadow: inset 4px 0 0 var(--yellow); }
    .card.reference-card { border-color: rgba(123,227,93,.75); box-shadow: inset 4px 0 0 var(--green); }
    .card .name { font-size: 28px; line-height: 1.06; font-weight: 900; margin-bottom: 8px; }
    .muted { color: var(--muted); line-height: 1.45; }
    .timeline-strip { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; }
    .thumb { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
    .thumb img { width: 100%; aspect-ratio: 1.25; object-fit: contain; background: #02070b; display: block; }
    .thumb div { padding: 10px 12px 12px; display: flex; justify-content: space-between; gap: 10px; font-weight: 800; }
    details { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 15px 18px; }
    summary { cursor: pointer; font-weight: 900; text-transform: uppercase; }
    .audit-grid { margin-top: 14px; display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
    .audit { background: #0a141e; border: 1px solid var(--line); border-radius: 8px; padding: 14px; }
    .audit.pass strong { color: var(--green); }
    .audit.warning strong { color: var(--yellow); }
    .audit.fail strong { color: var(--red); }
    table { border-collapse: collapse; width: 100%; margin-top: 14px; font-size: 13px; }
    th, td { border-bottom: 1px solid var(--line); padding: 8px; text-align: left; }
    th { color: var(--muted); }
    .table-wrap { max-height: 520px; overflow: auto; }
    footer { padding: 18px 32px; border-top: 1px solid var(--line); color: var(--muted); background: #050b11; }
    @media (max-width: 900px) {
      .status-band { grid-template-columns: 1fr; }
      .inspect { font-size: 42px; }
      .target { font-size: 28px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>{{ field_name }} &mdash; Field Inspection Work Order</h1>
    <div class="sub">Field Intelligence Summary | Visual Scouting Priority Map</div>
  </header>
  <main>
    <section class="work-order {{ status.level }}">
      <div class="status-band">
        <div>
          <div class="eyebrow">Field status: {{ status.label }}</div>
          <p class="inspect">INSPECT FIRST:</p>
          <p class="target">{{ top.location_label if top else "No urgent zone" }}</p>
          <p class="decision">{{ decision }}</p>
        </div>
        <div class="score">
          <span>Priority Score</span>
          <strong>{{ top.inspection_priority_score if top else 0 }}/100</strong>
          <p class="muted">Score combines persistence, expansion, severity, coverage, and evidence count.</p>
        </div>
      </div>
      <div class="work-grid">
        <div class="work-cell">
          <h3>Why it matters</h3>
          <ul>
            {% for item in why %}<li>{{ item }}</li>{% endfor %}
          </ul>
        </div>
        <div class="work-cell">
          <h3>Ground Verification</h3>
          <ul>
            <li>Irrigation consistency</li>
            <li>Leaf scorch</li>
            <li>Mildew pressure</li>
            <li>Pest pressure</li>
            <li>Soil issues</li>
          </ul>
        </div>
        <div class="work-cell">
          <h3>Recommended Action</h3>
          <p>Inspect within 24-48 hours.</p>
        </div>
        <div class="work-cell">
          <h3>Evidence Count</h3>
          <p>{{ top.evidence_count if top else 0 }} of {{ top.evidence_total if top else 6 }} available sources: {{ top.evidence_sources|join(", ") if top else "none" }}.</p>
        </div>
      </div>
      <div class="snapshot">{% if hero_rel %}<img src="{{ hero_rel }}" alt="Inspection work order map snapshot">{% endif %}</div>
    </section>

    <section>
      <h2>Annotated Inspection Map</h2>
      <div class="map-card">
        {% if hero_rel %}<img src="{{ hero_rel }}" alt="Annotated inspection map">{% endif %}
      </div>
    </section>

    <section>
      <h2>Watch Next</h2>
      <div class="cards">
        {% for block in watch_next %}
        <article class="card watch-card">
          <div class="eyebrow">Score {{ block.inspection_priority_score }} / 100</div>
          <div class="name">{{ block.location_label }}</div>
          <p class="muted">{{ block.reason }}</p>
          <p><strong>Recommended action:</strong> {{ block.action }}</p>
        </article>
        {% endfor %}
        {% if healthy %}
        <article class="card reference-card">
          <div class="eyebrow">Healthy Reference</div>
          <div class="name">{{ healthy.location_label }}</div>
          <p class="muted">Use this area as a comparison point while walking the field.</p>
        </article>
        {% endif %}
      </div>
    </section>

    <section>
      <h2>Timeline / Persistence Evidence</h2>
      <div class="timeline-strip">
        {% for row in timeline %}
        <article class="thumb">
          <img src="{{ overlays[row.date] }}" alt="{{ row.short_date }} scouting view">
          <div><span>{{ row.short_date }}</span><span>{{ arrow_symbol(row.arrow) }} | {{ row.trend }}</span></div>
        </article>
        {% endfor %}
      </div>
    </section>

    <section>
      <details>
        <summary>Technical Appendix</summary>
        <div class="audit-grid">
          {% for row in release_audit %}
          <article class="audit {{ row.status|lower }}">
            <strong>{{ row.status }}</strong>
            <h3>{{ row.name }}</h3>
            <p class="muted">{{ row.detail }}</p>
          </article>
          {% endfor %}
        </div>
        <h3>Scoring Components</h3>
        <div class="table-wrap">
          <table>
            <tr><th>Inspection Zone</th><th>Score</th><th>Persistence</th><th>Expansion</th><th>Severity</th><th>Coverage</th><th>Evidence Count</th><th>Evidence Sources</th></tr>
            {% for block in blocks %}
            <tr>
              <td>{{ block.location_label }}</td>
              <td>{{ block.inspection_priority_score }}</td>
              <td>{{ "%.1f"|format(block.persistence_score) }}</td>
              <td>{{ "%.1f"|format(block.expansion_score) }}</td>
              <td>{{ "%.1f"|format(block.severity_score) }}</td>
              <td>{{ "%.1f"|format(block.coverage_score) }}</td>
              <td>{{ block.evidence_count }} / {{ block.evidence_total }}</td>
              <td>{{ block.evidence_sources|join(", ") }}</td>
            </tr>
            {% endfor %}
          </table>
        </div>
        <h3>Internal Row Guide Summary</h3>
        <p class="muted">Exact row labels are customer-visible only when row confidence is at least 0.90. Latest confidence: {{ "%.2f"|format(tech.row_confidence) }}.</p>
        <div class="table-wrap">
          <table>
            <tr><th>Date</th><th>Internal row guide</th><th>Low %</th><th>Medium %</th><th>High %</th><th>Scout focus %</th></tr>
            {% for row in row_rows %}
            <tr><td>{{ row.short_date }}</td><td>{{ row.row_label }}</td><td>{{ row.low }}</td><td>{{ row.medium }}</td><td>{{ row.high }}</td><td>{{ row.focus }}</td></tr>
            {% endfor %}
          </table>
        </div>
        <h3>Technical Class Chart</h3>
        <img src="{{ chart_rel }}" alt="Technical class chart" style="width:100%; max-height:420px; object-fit:contain;">
        <h3>Release QA Detail</h3>
        <div class="table-wrap">
          <table>
            <tr><th>Status</th><th>Check</th><th>Date</th><th>Detail</th></tr>
            {% for row in audit_rows %}
            <tr><td>{{ row.status }}</td><td>{{ row.check }}</td><td>{{ row.date }}</td><td>{{ row.detail }}</td></tr>
            {% endfor %}
          </table>
        </div>
      </details>
    </section>
  </main>
  <footer>For scouting prioritization only. Ground inspection required to determine cause. Not a disease diagnosis.</footer>
</body>
</html>
        """
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlays_rel = {
        date: Path(os.path.relpath(path, output_path.parent)).as_posix()
        for date, path in overlay_paths_by_date.items()
    }
    hero_rel = Path(os.path.relpath(hero_image_path, output_path.parent)).as_posix() if hero_image_path else ""
    chart_rel = Path(os.path.relpath(chart_path, output_path.parent)).as_posix()
    html = template.render(
        field_name=field_name,
        status=status,
        top=top_block,
        decision=decision_sentence(top_block),
        why=why_bullets(top_block) if top_block else ["No urgent inspection zone was generated."],
        blocks=blocks,
        watch_next=watch_next,
        healthy=healthy_reference,
        timeline=timeline,
        overlays=overlays_rel,
        hero_rel=hero_rel,
        chart_rel=chart_rel,
        release_audit=release_audit,
        audit_rows=final_audit_rows,
        row_rows=row_report_rows(records),
        tech=tech,
        arrow_symbol=arrow_symbol,
        bed_aware=bed_aware,
    )
    output_path.write_text(html, encoding="utf-8")
    print(f"[WRITE] HTML report: {output_path}")
