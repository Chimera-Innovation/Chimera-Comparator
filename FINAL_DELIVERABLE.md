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
```

Expected dataset layout:

```text
D:\NDVI-dataset\raw-ndvi
D:\NDVI-dataset\human-Annotated-ndvi
D:\NDVI-dataset\MVP - ndvi progression
```

## Primary Files

- Comparator script: [build_brush_mark_comparator.py](build_brush_mark_comparator.py)
- Audit CSV: [brush_mark_comparator/brush_mark_extraction_audit.csv](brush_mark_comparator/brush_mark_extraction_audit.csv)
- Final report: [brush_mark_comparator/brush_mark_comparator_final_report.md](brush_mark_comparator/brush_mark_comparator_final_report.md)
- Artifact manifest: [deliverable_manifest.csv](deliverable_manifest.csv)

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
