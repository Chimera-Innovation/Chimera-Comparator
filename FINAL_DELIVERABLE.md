# Chimera Comparator Final Deliverable

## Deliverable Status

This is the usable MVP deliverable for transferring human Paint/brush annotations from marked NDVI images onto the matching raw NDVI background.

Final rule: human brush marks are truth. Raw NDVI is background only. No NDVI thresholding, stress inference, row detection, SAM, or DINO is used.

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
- Audit CSV: [brush_mark_comparator/brush_mark_extraction_audit.csv](brush_mark_comparator/brush_mark_extraction_audit.csv)
- Final report: [brush_mark_comparator/brush_mark_comparator_final_report.md](brush_mark_comparator/brush_mark_comparator_final_report.md)
- Farmer-ready report: [farmer_progression_output/farmer_ready_report.md](farmer_progression_output/farmer_ready_report.md)
- Farmer progression CSV: [farmer_progression_output/field_progression_summary.csv](farmer_progression_output/field_progression_summary.csv)
- Artifact manifest: [deliverable_manifest.csv](deliverable_manifest.csv)

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
