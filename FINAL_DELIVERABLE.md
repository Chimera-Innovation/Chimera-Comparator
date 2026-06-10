# Chimera Comparator Final Deliverable

## Deliverable Status

This is the usable MVP deliverable for converting human-marked strawberry NDVI annotations into reviewable scouting outputs.

Final rule: human brush marks are truth. Raw NDVI is background only. No NDVI thresholding, stress inference, row detection, SAM, or DINO is used.

The newest final package is the bed-aware Strawberry 1 Field Intelligence Summary. It treats Mallard and McIntyre source filenames as the same field, `Strawberry 1`, and turns human-derived vigour zones into a customer-facing walk-priority view.

## Current Result

- Fields processed: Mallard-Avenue and McIntyre-Road
- Dates processed: 6
- Successful extractions: 6
- Partial extractions: 0
- Manual review required: 0
- Failed extractions: 0

## Reproduce

Run from the MVP folder:

```powershell
python build_brush_mark_comparator.py
python build_farmer_progression_output.py
python build_june_bearing_vigour_report.py
cd strawberry_vigour_polygon_report
python -m src.main --raw-dir "D:\NDVI-dataset\raw-ndvi\Strawberry_1" --annotated-dir "C:\Users\Chimera\Downloads\Strawberry 1 - annotated" --field-name "Strawberry 1" --output-dir "C:\Users\Chimera\Downloads\Strawberry 1 - vigour polygon report" --bed-aware true --bed-numbering bottom_to_top
```

Expected dataset layout:

```text
D:\NDVI-dataset\raw-ndvi
D:\NDVI-dataset\human-Annotated-ndvi
D:\NDVI-dataset\MVP - ndvi progression
```

## Primary Files

- Comparator script: [build_brush_mark_comparator.py](build_brush_mark_comparator.py)
- Farmer progression script: [build_farmer_progression_output.py](build_farmer_progression_output.py)
- June-bearing vigour class script: [build_june_bearing_vigour_report.py](build_june_bearing_vigour_report.py)
- Strawberry vigour polygon project: [strawberry_vigour_polygon_report/README.md](strawberry_vigour_polygon_report/README.md)
- Bed detection and assignment layer: [strawberry_vigour_polygon_report/src/bed_detection.py](strawberry_vigour_polygon_report/src/bed_detection.py)
- Audit CSV: [brush_mark_comparator/brush_mark_extraction_audit.csv](brush_mark_comparator/brush_mark_extraction_audit.csv)
- Final report: [brush_mark_comparator/brush_mark_comparator_final_report.md](brush_mark_comparator/brush_mark_comparator_final_report.md)
- Farmer-ready report: [farmer_progression_output/farmer_ready_report.md](farmer_progression_output/farmer_ready_report.md)
- Farmer progression CSV: [farmer_progression_output/field_progression_summary.csv](farmer_progression_output/field_progression_summary.csv)
- June-bearing vigour report: [june_bearing_vigour_report/june_bearing_farmer_ready_report.md](june_bearing_vigour_report/june_bearing_farmer_ready_report.md)
- June-bearing vigour CSV: [june_bearing_vigour_report/vigour_class_summary.csv](june_bearing_vigour_report/vigour_class_summary.csv)
- Artifact manifest: [deliverable_manifest.csv](deliverable_manifest.csv)

## June-Bearing Vigour Class Output

The updated annotation package treats red as low vigour, purple as medium vigour, and light green as high vigour. Dates are parsed from the annotation filenames and mapped to approximate June-bearing strawberry growth stages.

| Field | Report | Timeline | Trend Chart | Dashboard |
| --- | --- | --- | --- | --- |
| Mallard-Avenue | [report](june_bearing_vigour_report/Mallard-Avenue/june_bearing_field_report.md) | [timeline](june_bearing_vigour_report/Mallard-Avenue/june_bearing_vigour_timeline.png) | [trend](june_bearing_vigour_report/Mallard-Avenue/vigour_class_percent_over_time.png) | [dashboard](june_bearing_vigour_report/Mallard-Avenue/june_bearing_farmer_dashboard.png) |
| McIntyre-Road | [report](june_bearing_vigour_report/McIntyre-Road/june_bearing_field_report.md) | [timeline](june_bearing_vigour_report/McIntyre-Road/june_bearing_vigour_timeline.png) | [trend](june_bearing_vigour_report/McIntyre-Road/vigour_class_percent_over_time.png) | [dashboard](june_bearing_vigour_report/McIntyre-Road/june_bearing_farmer_dashboard.png) |

The June-bearing report still uses raw NDVI as context only. It classifies the added human annotation colors after raw-vs-annotated comparison.

## Strawberry Scouting Priority Dashboard

The standalone dashboard pipeline treats Mallard and McIntyre as source labels for the same field, `Strawberry 1`.

It exports:

- Customer-facing Field Intelligence Summary
- Latest-flight visual priority map
- Customer release audit
- GeoJSON polygon classes
- CSV polygon summary
- Bed-level CSV summary
- PNG polygon overlay previews

Requested output folder:

```text
C:\Users\Chimera\Downloads\Strawberry 1 - vigour polygon report
```

The polygon pipeline adds rendered green-index and visual-NDVI-proxy context metrics to each human-derived polygon. Those indices affect overlay emphasis only; they do not create polygon boundaries or assign vigour class.

Validated bed-aware output:

```text
C:\Users\Chimera\Downloads\Strawberry 1 - vigour polygon report
```

Key files in that output:

- `polygons\strawberry_1_may29_vigour_polygons.geojson`
- `summaries\strawberry_1_vigour_polygon_summary.csv`
- `summaries\strawberry_1_bed_vigour_summary.csv`
- `reports\strawberry_1_latest_priority_map.png`
- `overlays\strawberry_1_may29_vigour_polygon_overlay.png`
- `reports\strawberry_1_vigour_polygon_report.html`
- `reports\strawberry_1_customer_release_audit.md`

The bed-aware run selected a stable seasonal count of 10 beds from the matched May image set and numbered them bottom-to-top.

The HTML report now opens as a Field Intelligence Summary: field status, top walk priorities, estimated inspection time, full-width scouting map, "Walk Here First", "What Changed", field timeline, collapsed field audit, and collapsed technical appendix.

Latest customer-facing result:

```text
ACTION REQUIRED
Start with Bed 09, Bed 07, Bed 03.
Field audit: 4 PASS, 1 WARNING, 0 FAIL.
```

## Farmer-Facing Progression Output

The farmer-facing package converts the validated brush comparator masks into filled human-marked vigour-loss zones and operational summaries.

| Field | Summary | Timeline | Trend Chart | Growth Map | Dashboard |
| --- | --- | --- | --- | --- | --- |
| Mallard-Avenue | [summary](farmer_progression_output/Mallard-Avenue/field_farmer_summary.md) | [timeline](farmer_progression_output/Mallard-Avenue/field_farmer_timeline.png) | [trend](farmer_progression_output/Mallard-Avenue/vigour_loss_percent_over_time.png) | [growth map](farmer_progression_output/Mallard-Avenue/field_concern_growth_map.png) | [dashboard](farmer_progression_output/Mallard-Avenue/field_farmer_dashboard.png) |
| McIntyre-Road | [summary](farmer_progression_output/McIntyre-Road/field_farmer_summary.md) | [timeline](farmer_progression_output/McIntyre-Road/field_farmer_timeline.png) | [trend](farmer_progression_output/McIntyre-Road/vigour_loss_percent_over_time.png) | [growth map](farmer_progression_output/McIntyre-Road/field_concern_growth_map.png) | [dashboard](farmer_progression_output/McIntyre-Road/field_farmer_dashboard.png) |

Farmer-facing wording uses human-observed concern, persistent low-vigour area, vigour-loss zone, and requires investigation. It does not diagnose disease.

## Visual Review

Each debug image contains five panels:

1. Raw NDVI
2. Human annotated reference
3. Difference heatmap
4. Extracted brush mask
5. Raw NDVI with transferred brush marks

| Field | Date | Debug Panel | Final Overlay | Mask |
| --- | --- | --- | --- | --- |
| Mallard-Avenue | 2026-05-12 | [debug](brush_mark_comparator/Mallard-Avenue/debug/Mallard-Avenue_2026-05-12_brush_debug.png) | [overlay](brush_mark_comparator/Mallard-Avenue/overlays/Mallard-Avenue_2026-05-12_raw_with_brush_marks.png) | [mask](brush_mark_comparator/Mallard-Avenue/masks/Mallard-Avenue_2026-05-12_brush_mask.png) |
| Mallard-Avenue | 2026-05-18 | [debug](brush_mark_comparator/Mallard-Avenue/debug/Mallard-Avenue_2026-05-18_brush_debug.png) | [overlay](brush_mark_comparator/Mallard-Avenue/overlays/Mallard-Avenue_2026-05-18_raw_with_brush_marks.png) | [mask](brush_mark_comparator/Mallard-Avenue/masks/Mallard-Avenue_2026-05-18_brush_mask.png) |
| Mallard-Avenue | 2026-05-22 | [debug](brush_mark_comparator/Mallard-Avenue/debug/Mallard-Avenue_2026-05-22_brush_debug.png) | [overlay](brush_mark_comparator/Mallard-Avenue/overlays/Mallard-Avenue_2026-05-22_raw_with_brush_marks.png) | [mask](brush_mark_comparator/Mallard-Avenue/masks/Mallard-Avenue_2026-05-22_brush_mask.png) |
| Mallard-Avenue | 2026-05-29 | [debug](brush_mark_comparator/Mallard-Avenue/debug/Mallard-Avenue_2026-05-29_brush_debug.png) | [overlay](brush_mark_comparator/Mallard-Avenue/overlays/Mallard-Avenue_2026-05-29_raw_with_brush_marks.png) | [mask](brush_mark_comparator/Mallard-Avenue/masks/Mallard-Avenue_2026-05-29_brush_mask.png) |
| McIntyre-Road | 2026-05-08 | [debug](brush_mark_comparator/McIntyre-Road/debug/McIntyre-Road_2026-05-08_brush_debug.png) | [overlay](brush_mark_comparator/McIntyre-Road/overlays/McIntyre-Road_2026-05-08_raw_with_brush_marks.png) | [mask](brush_mark_comparator/McIntyre-Road/masks/McIntyre-Road_2026-05-08_brush_mask.png) |
| McIntyre-Road | 2026-05-27 | [debug](brush_mark_comparator/McIntyre-Road/debug/McIntyre-Road_2026-05-27_brush_debug.png) | [overlay](brush_mark_comparator/McIntyre-Road/overlays/McIntyre-Road_2026-05-27_raw_with_brush_marks.png) | [mask](brush_mark_comparator/McIntyre-Road/masks/McIntyre-Road_2026-05-27_brush_mask.png) |

## Review Checklist

- Open each debug panel and compare panel 2 against panel 5.
- Confirm the transferred marks follow the human brush strokes.
- Confirm there are no extra algorithmic rectangles, NDVI-derived stress regions, or fake 0% labels.
- Treat any visible mismatch as an extraction bug, because the human marking is the source of truth.

## Git State At Delivery

Branch:

```text
codex/brush-mark-comparator-usable
```

Published repository:

```text
https://github.com/Chimera-Innovation/Chimera-Comparator.git
```
