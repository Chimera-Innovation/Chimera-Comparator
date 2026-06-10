from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import median

import cv2

from .annotation_parser import (
    class_masks_from_annotation,
    get_annotation_mask,
    raw_class_masks_from_annotation,
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


def outputs_are_current(output_paths: list[Path], source_paths: list[Path]) -> bool:
    if not output_paths or not all(path.exists() for path in output_paths):
        return False
    newest_source = max(path.stat().st_mtime for path in source_paths if path.exists())
    oldest_output = min(path.stat().st_mtime for path in output_paths)
    return oldest_output >= newest_source


def field_foreground_mask(raw_bgr) -> object:
    gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
    foreground = (gray > 8) & (gray < 248)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    foreground_u8 = foreground.astype("uint8") * 255
    foreground_u8 = cv2.morphologyEx(foreground_u8, cv2.MORPH_CLOSE, kernel)
    foreground_u8 = cv2.morphologyEx(foreground_u8, cv2.MORPH_OPEN, kernel)
    return foreground_u8 > 0


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
    parser.add_argument("--min-human-overlap-ratio", default=0.01, type=float)
    parser.add_argument("--min-buffer-overlap-ratio", default=0.95, type=float)
    parser.add_argument("--field-area-m2", default=None, type=float)
    parser.add_argument("--max-processing-dimension", default=3600, type=int)
    parser.add_argument("--date", action="append", default=None, help="Optional YYYY-MM-DD date filter. Can be supplied more than once.")
    parser.add_argument("--use-cache", default="true", choices=("true", "false"), help="Reuse current per-date outputs when source images have not changed.")
    parser.add_argument("--force-reprocess", action="store_true", help="Ignore cached per-date outputs and rebuild all selected dates.")
    parser.add_argument("--ground-notes-csv", default=None, type=Path)
    parser.add_argument("--event-log-csv", default=None, type=Path)
    return parser.parse_args()


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "polygons": output_dir / "polygons",
        "summaries": output_dir / "summaries",
        "overlays": output_dir / "overlays",
        "reports": output_dir / "reports",
        "debug": output_dir / "debug",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def main() -> None:
    args = parse_args()
    row_aware = args.bed_aware.lower() == "true"
    row_numbering = args.row_numbering or args.bed_numbering or "top_to_bottom"
    if row_numbering != "top_to_bottom":
        print("[ROWS] Deprecated --bed-numbering/--row-numbering value was not top_to_bottom; using top_to_bottom.")
        row_numbering = "top_to_bottom"
    output_paths = ensure_output_dirs(args.output_dir)
    print(f"[START] Field: {args.field_name}")
    print(f"[START] Raw folder: {args.raw_dir}")
    print(f"[START] Annotated folder: {args.annotated_dir}")
    print(f"[START] Output folder: {args.output_dir}")
    print(f"[START] Row-aware inspection mode: {row_aware} ({row_numbering})")
    guided_config = HumanGuidedConfig(
        buffer_distance_pixels=args.human_buffer_pixels,
        min_area_pixels=args.min_inspection_area_pixels,
        max_component_area_pixels=args.max_component_area_pixels,
        max_percent_field_area=args.max_percent_field_area,
        max_component_width_ratio=args.max_component_width_ratio,
        min_human_overlap_ratio=args.min_human_overlap_ratio,
        min_buffer_overlap_ratio=args.min_buffer_overlap_ratio,
        field_area_m2=args.field_area_m2,
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
    use_cache = args.use_cache.lower() == "true" and not args.force_reprocess
    if row_aware and global_row_count is None:
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
        raw_class_masks = raw_class_masks_from_annotation(annotated_bgr, annotation_mask)
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
        class_masks = class_masks_from_annotation(annotated_bgr, annotation_mask)
        growth_stage = growth_stage_for_date(pair.date)
        if row_aware:
            row_regions, row_confidence = detect_rows(raw_bgr, row_numbering, global_row_count)
        else:
            row_regions, row_confidence = [], 0.0
        row_regions_by_date[pair.date] = row_regions
        row_confidence_by_date[pair.date] = row_confidence
        records = build_human_guided_polygons(
            class_masks,
            raw_bgr,
            args.field_name,
            pair.date,
            growth_stage,
            row_regions,
            guided_config,
            output_paths["debug"],
            processing_scale=processing_scale,
            original_width=original_width,
            original_height=original_height,
            processed_width=processed_width,
            processed_height=processed_height,
            coordinate_space="processed_image_pixels",
        )
        if not records:
            print(f"[WARN] No human-guided inspection polygons generated for {pair.date}; no polygons exported for this date.")
            continue

        date_records = sorted(records, key=lambda record: (record.class_id, record.polygon_id))
        all_records.extend(date_records)
        records_by_date[pair.date].extend(date_records)

        write_geojson(date_records, geojson_path)

        draw_polygon_overlay(raw_bgr, date_records, overlay_path, None)
        overlay_paths_by_date[pair.date] = overlay_path
        raw_paths_by_date[pair.date] = pair.raw_path

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
