from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .annotation_parser import PolygonRecord


def date_slug(date: str) -> str:
    month = int(date[5:7])
    day = int(date[8:10])
    if month == 5:
        return f"may{day}"
    return date.replace("-", "")


def field_slug(field_name: str) -> str:
    return field_name.lower().replace(" ", "_").replace("-", "_")


def processing_metadata(record: PolygonRecord) -> dict[str, int | float | str]:
    return {
        "processing_scale": float(record.processing_scale),
        "original_width": int(record.original_width),
        "original_height": int(record.original_height),
        "processed_width": int(record.processed_width),
        "processed_height": int(record.processed_height),
        "coordinate_space": record.coordinate_space,
    }


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_list(value: object) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return []


def read_geojson_records(input_path: Path) -> list[PolygonRecord]:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    records: list[PolygonRecord] = []
    for index, feature in enumerate(data.get("features", []), start=1):
        properties = feature.get("properties", {})
        geometry = feature.get("geometry", {})
        coordinates = geometry.get("coordinates", [[]])[0]
        coords = [[int(round(point[0])), int(round(point[1]))] for point in coordinates if len(point) >= 2]
        if len(coords) < 3:
            continue
        records.append(
            PolygonRecord(
                field_name=str(properties.get("field_name", "")),
                date=str(properties.get("date", "")),
                growth_stage=str(properties.get("growth_stage", "")),
                class_name=str(properties.get("class_name", "")),
                class_id=_as_int(properties.get("class_id"), 0),
                polygon_id=str(properties.get("polygon_id", f"P{index:03d}")),
                bed_id=str(properties.get("bed_id", "")),
                centroid_x=_as_float(properties.get("centroid_x")),
                centroid_y=_as_float(properties.get("centroid_y")),
                area_pixels=_as_float(properties.get("area_pixels")),
                area_percent=_as_float(properties.get("area_percent")),
                bed_area_pixels=_as_float(properties.get("bed_area_pixels"), 1.0),
                area_percent_within_bed=min(100.0, _as_float(properties.get("area_percent_within_bed"))),
                scouting_priority=str(properties.get("scouting_priority", "")),
                confidence_source=str(properties.get("confidence_source", "")),
                note=str(properties.get("note", "")),
                mean_green_index=_as_float(properties.get("mean_green_index")),
                mean_visual_ndvi_proxy=_as_float(properties.get("mean_visual_ndvi_proxy")),
                coordinates=coords,
                area_m2=None if properties.get("area_m2") is None else _as_float(properties.get("area_m2")),
                severity=str(properties.get("severity", "")),
                confidence=_as_float(properties.get("confidence")),
                affected_rows=[_as_int(value) for value in _as_list(properties.get("affected_rows"))],
                affected_beds=[str(value) for value in _as_list(properties.get("affected_beds"))],
                vigour_loss_percent=min(100.0, _as_float(properties.get("vigour_loss_percent"))),
                source=[str(value) for value in _as_list(properties.get("source"))],
                row_segment_start_m=None if properties.get("row_segment_start_m") is None else _as_float(properties.get("row_segment_start_m")),
                row_segment_end_m=None if properties.get("row_segment_end_m") is None else _as_float(properties.get("row_segment_end_m")),
                row_overlap_percent=min(100.0, _as_float(properties.get("row_overlap_percent"))),
                review_status=str(properties.get("review_status", "needs_human_review")),
                processing_scale=_as_float(properties.get("processing_scale"), 1.0),
                original_width=_as_int(properties.get("original_width")),
                original_height=_as_int(properties.get("original_height")),
                processed_width=_as_int(properties.get("processed_width")),
                processed_height=_as_int(properties.get("processed_height")),
                coordinate_space=str(properties.get("coordinate_space", "processed_image_pixels")),
            )
        )
    return records


def write_geojson(records: list[PolygonRecord], output_path: Path) -> None:
    features = []
    for record in records:
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "field_name": record.field_name,
                    "date": record.date,
                    "growth_stage": record.growth_stage,
                    "class_name": record.class_name,
                    "class_id": record.class_id,
                    "polygon_id": record.polygon_id,
                    "bed_id": record.bed_id,
                    "centroid_x": record.centroid_x,
                    "centroid_y": record.centroid_y,
                    "area_pixels": record.area_pixels,
                    "area_percent": record.area_percent,
                    "bed_area_pixels": record.bed_area_pixels,
                    "area_percent_within_bed": record.area_percent_within_bed,
                    "scouting_priority": record.scouting_priority,
                    "confidence_source": record.confidence_source,
                    "note": record.note,
                    "mean_green_index": record.mean_green_index,
                    "mean_visual_ndvi_proxy": record.mean_visual_ndvi_proxy,
                    "area_m2": record.area_m2,
                    "severity": record.severity,
                    "confidence": record.confidence,
                    "affected_rows": record.affected_rows or [],
                    "affected_beds": record.affected_beds or [],
                    "vigour_loss_percent": record.vigour_loss_percent,
                    "source": record.source or [],
                    "row_segment_start_m": record.row_segment_start_m,
                    "row_segment_end_m": record.row_segment_end_m,
                    "row_overlap_percent": record.row_overlap_percent,
                    "review_status": record.review_status,
                    **processing_metadata(record),
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [record.coordinates],
                },
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2), encoding="utf-8")
    print(f"[WRITE] GeoJSON: {output_path}")


def write_polygon_summary(records: list[PolygonRecord], output_path: Path) -> None:
    rows = [
        {
            "field_name": record.field_name,
            "date": record.date,
            "growth_stage": record.growth_stage,
            "class_name": record.class_name,
            "class_id": record.class_id,
            "polygon_id": record.polygon_id,
            "bed_id": record.bed_id,
            "centroid_x": round(record.centroid_x, 2),
            "centroid_y": round(record.centroid_y, 2),
            "area_pixels": round(record.area_pixels, 2),
            "area_percent": round(record.area_percent, 6),
            "bed_area_pixels": round(record.bed_area_pixels, 2),
            "area_percent_within_bed": round(record.area_percent_within_bed, 6),
            "scouting_priority": record.scouting_priority,
            "confidence_source": record.confidence_source,
            "mean_green_index": round(record.mean_green_index, 6),
            "mean_visual_ndvi_proxy": round(record.mean_visual_ndvi_proxy, 6),
            "area_m2": "" if record.area_m2 is None else round(record.area_m2, 4),
            "severity": record.severity,
            "confidence": round(record.confidence, 4),
            "affected_rows": ",".join(str(row) for row in (record.affected_rows or [])),
            "affected_beds": ",".join(record.affected_beds or []),
            "vigour_loss_percent": round(record.vigour_loss_percent, 4),
            "source": ",".join(record.source or []),
            "row_segment_start_m": "" if record.row_segment_start_m is None else round(record.row_segment_start_m, 4),
            "row_segment_end_m": "" if record.row_segment_end_m is None else round(record.row_segment_end_m, 4),
            "row_overlap_percent": round(record.row_overlap_percent, 4),
            "review_status": record.review_status,
            **processing_metadata(record),
        }
        for record in records
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"[WRITE] CSV summary: {output_path}")


def load_notes(path: Path | None) -> dict[tuple[str, str], list[str]]:
    if not path:
        return {}
    if not path.exists():
        print(f"[WARN] Optional notes CSV does not exist: {path}")
        return {}
    frame = pd.read_csv(path)
    notes: dict[tuple[str, str], list[str]] = {}
    for _, row in frame.iterrows():
        date = str(row.get("date", "")).strip()
        bed_id = str(row.get("bed_id", "")).strip() or "all"
        text = str(row.get("note", row.get("notes", row.to_dict()))).strip()
        if not date or not text:
            continue
        notes.setdefault((date, bed_id), []).append(text)
    return notes


def notes_for(notes: dict[tuple[str, str], list[str]], date: str, bed_id: str) -> str:
    values = []
    values.extend(notes.get((date, bed_id), []))
    values.extend(notes.get((date, "all"), []))
    return " | ".join(values)


def write_bed_summary(
    records: list[PolygonRecord],
    output_path: Path,
    ground_notes_csv: Path | None = None,
    event_log_csv: Path | None = None,
) -> None:
    ground_notes = load_notes(ground_notes_csv)
    event_notes = load_notes(event_log_csv)
    grouped: dict[tuple[str, str, str, str], dict[str, float | str]] = {}
    for record in records:
        key = (record.field_name, record.date, record.growth_stage, record.bed_id)
        item = grouped.setdefault(
            key,
            {
                "bed_area_pixels": record.bed_area_pixels,
                **processing_metadata(record),
                "low_vigour_area_pixels": 0.0,
                "medium_vigour_area_pixels": 0.0,
                "high_vigour_area_pixels": 0.0,
            },
        )
        item[f"{record.class_name}_area_pixels"] = float(item[f"{record.class_name}_area_pixels"]) + record.area_pixels

    rows = []
    previous_focus_by_bed: dict[str, float] = {}
    for (field_name, date, growth_stage, bed_id), item in sorted(grouped.items(), key=lambda pair: (pair[0][1], pair[0][3])):
        bed_area = max(1.0, float(item["bed_area_pixels"]))
        low_pct = min(100.0, float(item["low_vigour_area_pixels"]) / bed_area * 100)
        medium_pct = min(100.0, float(item["medium_vigour_area_pixels"]) / bed_area * 100)
        high_pct = min(100.0, float(item["high_vigour_area_pixels"]) / bed_area * 100)
        focus = min(100.0, low_pct + medium_pct)
        previous = previous_focus_by_bed.get(bed_id)
        if previous is None:
            change = ""
            trend = "first_observation"
        elif abs(focus - previous) < 0.25:
            change = round(focus - previous, 6)
            trend = "stable"
        elif focus > previous:
            change = round(focus - previous, 6)
            trend = "increased"
        else:
            change = round(focus - previous, 6)
            trend = "decreased"
        previous_focus_by_bed[bed_id] = focus
        rows.append(
            {
                "field_name": field_name,
                "date": date,
                "growth_stage": growth_stage,
                "bed_id": bed_id,
                "bed_area_pixels": round(bed_area, 2),
                "low_vigour_area_pixels": round(float(item["low_vigour_area_pixels"]), 2),
                "medium_vigour_area_pixels": round(float(item["medium_vigour_area_pixels"]), 2),
                "high_vigour_area_pixels": round(float(item["high_vigour_area_pixels"]), 2),
                "low_vigour_percent_within_bed": round(low_pct, 6),
                "medium_vigour_percent_within_bed": round(medium_pct, 6),
                "high_vigour_percent_within_bed": round(high_pct, 6),
                "scout_focus_percent_within_bed": round(focus, 6),
                "change_from_previous_date": change,
                "trend_from_previous_date": trend,
                "scouting_priority": priority_from_focus(low_pct, medium_pct),
                "ground_notes": notes_for(ground_notes, date, bed_id),
                "event_notes": notes_for(event_notes, date, bed_id),
                "processing_scale": item["processing_scale"],
                "original_width": item["original_width"],
                "original_height": item["original_height"],
                "processed_width": item["processed_width"],
                "processed_height": item["processed_height"],
                "coordinate_space": item["coordinate_space"],
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"[WRITE] Bed CSV summary: {output_path}")


def write_row_summary(records: list[PolygonRecord], output_path: Path) -> None:
    grouped: dict[tuple[str, str], dict[str, float | str]] = {}
    for record in records:
        key = (record.date, record.bed_id)
        item = grouped.setdefault(
            key,
            {
                "row_id": record.bed_id,
                "row_area_pixels": record.bed_area_pixels,
                **processing_metadata(record),
                "low_vigour_overlap": 0.0,
                "medium_vigour_overlap": 0.0,
                "high_vigour_overlap": 0.0,
            },
        )
        item["row_area_pixels"] = max(float(item["row_area_pixels"]), record.bed_area_pixels)
        if record.class_name == "low_vigour":
            item["low_vigour_overlap"] = float(item["low_vigour_overlap"]) + record.area_pixels
        elif record.class_name == "medium_vigour":
            item["medium_vigour_overlap"] = float(item["medium_vigour_overlap"]) + record.area_pixels
        elif record.class_name == "high_vigour":
            item["high_vigour_overlap"] = float(item["high_vigour_overlap"]) + record.area_pixels

    previous_focus_by_row: dict[str, float] = {}
    rows = []
    for (date, row_id), item in sorted(grouped.items(), key=lambda pair: (pair[0][1], pair[0][0])):
        row_area = max(1.0, float(item["row_area_pixels"]))
        low = float(item["low_vigour_overlap"])
        medium = float(item["medium_vigour_overlap"])
        high = float(item["high_vigour_overlap"])
        focus_percent = min(100.0, (low + medium) / row_area * 100)
        previous_focus = previous_focus_by_row.get(row_id)
        if previous_focus is None:
            change = ""
        else:
            change = round(focus_percent - previous_focus, 6)
        previous_focus_by_row[row_id] = focus_percent
        row_number = ""
        if row_id.startswith("row_"):
            try:
                row_number = int(row_id.split("_", 1)[1])
            except ValueError:
                row_number = ""
        rows.append(
            {
                "date": date,
                "row_id": row_id,
                "row_number": row_number,
                "low_vigour_overlap": round(low, 2),
                "medium_vigour_overlap": round(medium, 2),
                "high_vigour_overlap": round(high, 2),
                "scout_priority": priority_from_focus(low / row_area * 100, medium / row_area * 100),
                "change_from_previous_flight": change,
                "processing_scale": item["processing_scale"],
                "original_width": item["original_width"],
                "original_height": item["original_height"],
                "processed_width": item["processed_width"],
                "processed_height": item["processed_height"],
                "coordinate_space": item["coordinate_space"],
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"[WRITE] Row CSV summary: {output_path}")


def priority_from_focus(low_pct: float, medium_pct: float) -> str:
    if low_pct >= 5:
        return "highest priority"
    if low_pct + medium_pct >= 5:
        return "monitor / secondary priority"
    return "reference / low immediate concern"
