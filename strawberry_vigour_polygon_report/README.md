# Strawberry Scouting Priority Dashboard

This project converts human-annotated NDVI images into a customer-facing Field Intelligence Summary for a single field.

It is a human-annotation parser, not an NDVI threshold classifier.

Raw NDVI is used only as the background comparison image. Polygon classes are generated only from annotation colors added by the human reviewer:

- Red = low vigour
- Purple = medium vigour
- Light green = high vigour

McIntyre and Mallard source labels are treated as image/source names for the same field when `--field-name "Strawberry 1"` is used.

## Install

```powershell
pip install -r requirements.txt
```

## Run

```powershell
python -m src.main ^
  --raw-dir "C:\Users\Chimera\Downloads\Strawberry 1" ^
  --annotated-dir "C:\Users\Chimera\Downloads\Strawberry 1 - annotated" ^
  --field-name "Strawberry 1" ^
  --output-dir "C:\Users\Chimera\Downloads\Strawberry 1 - vigour polygon report" ^
  --bed-aware true ^
  --row-numbering top_to_bottom ^
  --row-count 56
```

Optional:

```powershell
  --ground-notes-csv "path\to\ground_notes.csv" ^
  --event-log-csv "path\to\event_log.csv" ^
  --date 2026-05-29 ^
  --use-cache true ^
  --force-reprocess
```

`--bed-aware` and `--bed-count` are retained as deprecated aliases for compatibility. Customer-facing output uses row-aware inspection language. Row 1 is the top-most visible strawberry row, and row numbers increase downward.

The default `--use-cache true` reuses per-date GeoJSON, overlay, and debug audit outputs when they are newer than the matching raw and annotated source images. Use `--force-reprocess` after changing extraction thresholds.

## Outputs

```text
polygons/
summaries/
overlays/
reports/
```

Primary outputs:

- `polygons/<field>_<date>_vigour_polygons.geojson`
- `summaries/strawberry_1_vigour_polygon_summary.csv`
- `row_vigour_summary.csv`
- `overlays/<field>_<date>_vigour_polygon_overlay.png`
- `priority_rows_overlay.png`
- `mission_debrief.html`
- `reports/strawberry_1_customer_release_audit.md`

The GeoJSON uses image pixel coordinates only. It does not pretend to be GPS or GIS data unless georeferencing is added later.

See `VALIDATION_REPORT.md` for the last validated command, output row counts, and the note about which local input folders were present during validation.

## Row-Aware Work Orders

Rows are the field navigation layer. Every polygon is constrained by human annotation influence, then associated with row guides for internal summaries.

The row-aware summaries include:

- class area percentage
- class area percentage within row guide
- scout focus percentage: low vigour + medium vigour
- change from previous date
- optional ground notes and event log notes when CSVs are supplied

The HTML report is a Field Inspection Work Order. It opens with the recommended first walk, priority score, why it matters, what to verify on the ground, and a large visual scouting map. Raw image, vigour map, and inferred polygons are shown as the latest evidence stack. The full audit and technical tables are collapsed by default.

The priority engine ranks inspection targets by:

- low-vigour share
- scout-focus share: low + medium
- increase since the previous flight
- largest connected low-vigour zone

Expected optional notes CSV columns are flexible, but `date`, `bed_id`, and `note` are preferred for now. Use `bed_id=all` for date-level notes.

## Tests

```powershell
python -m unittest discover -s tests
```

## Notes

The report is for scouting prioritization only. It is not a disease diagnosis and does not claim yield loss.
