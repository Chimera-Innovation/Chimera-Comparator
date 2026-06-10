# Strawberry Vigour Polygon Validation Report

Validation date: 2026-06-09

## Implementation Check

The package compiled successfully with:

```powershell
cd "D:\NDVI-dataset\MVP - ndvi progression\strawberry_vigour_polygon_report"
C:\Users\Chimera\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m compileall .
```

## Requested Input Folders

These requested Downloads input folders were not present during validation:

```text
C:\Users\Chimera\Downloads\Strawberry 1 - raw images
C:\Users\Chimera\Downloads\Strawberry 1 - annotated - annotated output
```

The pipeline itself supports those paths when they exist. For validation, the available equivalent inputs were used:

```text
D:\NDVI-dataset\raw-ndvi\Strawberry_1
C:\Users\Chimera\Downloads\Strawberry 1 - annotated
```

## Validated Command

```powershell
cd "D:\NDVI-dataset\MVP - ndvi progression\strawberry_vigour_polygon_report"
C:\Users\Chimera\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m src.main --raw-dir "D:\NDVI-dataset\raw-ndvi\Strawberry_1" --annotated-dir "C:\Users\Chimera\Downloads\Strawberry 1 - annotated" --field-name "Strawberry 1" --output-dir "C:\Users\Chimera\Downloads\Strawberry 1 - vigour polygon report" --bed-aware true --bed-numbering bottom_to_top
```

## Validation Result

- Matched dates: 2026-05-08, 2026-05-12, 2026-05-18, 2026-05-22, 2026-05-27, 2026-05-29
- Unified field name: Strawberry 1
- Stable seasonal bed count: 10
- Bed numbering: bottom_to_top
- Polygon summary rows: 290
- Bed summary rows: 48
- HTML layout: customer-facing Field Intelligence Summary
- Field status: ACTION REQUIRED
- Top walk priorities: Bed 09, Bed 07, Bed 03, Bed 08, Bed 02
- Latest priority map labels: #1 to #5 beds on raw-image overlay
- Field audit: 4 PASS, 1 WARNING, 0 FAIL

## Primary Output Folder

```text
C:\Users\Chimera\Downloads\Strawberry 1 - vigour polygon report
```

Primary files:

- `polygons\strawberry_1_may29_vigour_polygons.geojson`
- `summaries\strawberry_1_vigour_polygon_summary.csv`
- `summaries\strawberry_1_bed_vigour_summary.csv`
- `overlays\strawberry_1_may29_vigour_polygon_overlay.png`
- `reports\strawberry_1_latest_priority_map.png`
- `reports\strawberry_1_vigour_polygon_report.html`
- `reports\strawberry_1_customer_release_audit.md`

## Classification Boundary

Raw NDVI was used only as evidence/background and as the comparison source for detecting changed annotation pixels.

Vigour class labels came only from human-added annotation colors:

- Red: low_vigour
- Purple: medium_vigour
- Light green: high_vigour

Rendered green-index and visual-NDVI-proxy values are exported only as context metrics and overlay emphasis helpers. They do not assign class labels and do not create polygons.
