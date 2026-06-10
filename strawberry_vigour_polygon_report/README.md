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
  --raw-dir "C:\Users\Chimera\Downloads\Strawberry 1 - raw images" ^
  --annotated-dir "C:\Users\Chimera\Downloads\Strawberry 1 - annotated - annotated output" ^
  --field-name "Strawberry 1" ^
  --output-dir "C:\Users\Chimera\Downloads\Strawberry 1 - vigour polygon report" ^
  --bed-aware true ^
  --bed-numbering bottom_to_top
```

Optional:

```powershell
  --ground-notes-csv "path\to\ground_notes.csv" ^
  --event-log-csv "path\to\event_log.csv" ^
  --bed-count 10
```

If `--bed-aware true` is enabled and `--bed-count` is not supplied, the pipeline detects a seasonal bed count across matched dates, then applies stable bed IDs consistently. `bottom_to_top` means `bed_01` starts at the bottom of the image.

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
- `summaries/strawberry_1_bed_vigour_summary.csv`
- `overlays/<field>_<date>_vigour_polygon_overlay.png`
- `reports/strawberry_1_latest_priority_map.png`
- `reports/strawberry_1_vigour_polygon_report.html`
- `reports/strawberry_1_customer_release_audit.md`

The GeoJSON uses image pixel coordinates only. It does not pretend to be GPS or GIS data unless georeferencing is added later.

See `VALIDATION_REPORT.md` for the last validated command, output row counts, and the note about which local input folders were present during validation.

## Bed Awareness

Bed IDs are a field navigation layer. Every polygon is assigned to a bed by polygon centroid.

The bed-aware summaries include:

- class area percentage
- class area percentage within bed
- scout focus percentage: low vigour + medium vigour
- change from previous date
- optional ground notes and event log notes when CSVs are supplied

The HTML report is a Field Intelligence Summary. It opens with field status, top walk priorities, estimated inspection time, and a large visual scouting map. The full audit and technical tables are collapsed by default.

The priority engine ranks beds by:

- low-vigour share
- scout-focus share: low + medium
- increase since the previous flight
- largest connected low-vigour zone

Expected optional notes CSV columns are flexible, but `date`, `bed_id`, and `note` are preferred. Use `bed_id=all` for date-level notes.

## Notes

The report is for scouting prioritization only. It is not a disease diagnosis and does not claim yield loss.
