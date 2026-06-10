# Chimera Comparator

Human-annotation-first NDVI reporting tools for strawberry vigour scouting.

The current final package is `strawberry_vigour_polygon_report/`, a customer-facing Field Intelligence Summary for the unified `Strawberry 1` field. Mallard and McIntyre filenames are treated as source labels for the same field.

Current usable output:

- `FINAL_DELIVERABLE.md`
- `deliverable_manifest.csv`
- `build_brush_mark_comparator.py`
- `build_farmer_progression_output.py`
- `build_june_bearing_vigour_report.py`
- `strawberry_vigour_polygon_report/`
- `brush_mark_comparator/`
- `farmer_progression_output/`
- `june_bearing_vigour_report/`

Final rule: human brush marks are truth. Raw NDVI is background only. No NDVI thresholding or automatic stress detection is used.

Primary customer output:

```text
C:\Users\Chimera\Downloads\Strawberry 1 - vigour polygon report\reports\strawberry_1_vigour_polygon_report.html
```

Run the final bed-aware pipeline from `strawberry_vigour_polygon_report/`:

```powershell
python -m src.main --raw-dir "D:\NDVI-dataset\raw-ndvi\Strawberry_1" --annotated-dir "C:\Users\Chimera\Downloads\Strawberry 1 - annotated" --field-name "Strawberry 1" --output-dir "C:\Users\Chimera\Downloads\Strawberry 1 - vigour polygon report" --bed-aware true --bed-numbering bottom_to_top
```
