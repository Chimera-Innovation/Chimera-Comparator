# NDVI Progression Feasibility Audit

- Dataset path: `D:\NDVI-dataset`
- Audit output folder: `D:\NDVI-dataset\MVP - ndvi progression`

## 1. Executive summary

- Total files scanned (excluding audit output folder): **12**
- Total image files: **12**
- Fields detected: **Mallard-Avenue, McIntyre-Road**
- Dates detected: **6** (`2026-05-08`, `2026-05-12`, `2026-05-18`, `2026-05-22`, `2026-05-27`, `2026-05-29`)
- Progression tracking possible today: **yes, at whole-field level**
- Best achievable level today: **LEVEL 2**
- Main blockers: no RGB timeline, no machine-readable annotations, almost no georeferenced repeats, no field context or severity tables.

### Direct answers to the audit questions

1. Can we track disease progression over time? **Yes, but only at a coarse whole-field / manually defined zone level.**
2. At what spatial level can we track it?
   - **whole field**: yes — Two field units have repeated dates and can be compared over time.
   - **plot**: no reliable IDs — No explicit plot IDs or plot boundary files were found.
   - **row**: not reliable — No stable row IDs, row annotations, or repeatable geospatial alignment across dates.
   - **row set**: manual only — A human could define repeatable row-set zones, but none exist in the data today.
   - **grid cell**: experimental — Possible only after image registration on repeated field views, mainly for Mallard-Avenue.
   - **individual plant**: no — Orthomosaic-level NDVI exports are not sufficient for plant-level temporal labels here.
3. How complete is the current timeline? **Partial.** Mallard-Avenue has 4 dates over 17 days; McIntyre-Road has 2 dates over 19 days; no longer-season progression history was found.
4. How much can we do with the current data without collecting more? Whole-field NDVI timelines, manual progression review, and farmer-demo reporting are feasible now.
5. What is missing before we can build a reliable disease progression model? Machine-readable repeated annotations, severity labels, stable row/zone IDs, RGB pairing, and field/treatment/outcome context.

## 2. Data inventory tables

### A. Folder structure

- Top-level folders: `MVP - ndvi progression`, `human-Annotated-ndvi`, `raw-ndvi`
- Date-based folders: **none detected**
- Raw folders: `raw-ndvi`, `raw-ndvi/Strawberry_1`
- Annotated folders: `human-Annotated-ndvi`
- RGB folders: **none detected**
- NDVI/multispectral folders: `raw-ndvi`, `human-Annotated-ndvi`
- Context/metadata folders: **none detected**
- Output/generated folders: `MVP - ndvi progression`

### B. Image inventory

| Metric | Count |
|---|---:|
| Total image files | 12 |
| Readable image files | 12 |
| Unreadable/corrupted image files | 0 |
| RGB images | 0 |
| Raw NDVI images | 6 |
| Annotated NDVI images | 6 |
| Multispectral images (>4 bands) | 0 |
| GeoTIFFs with CRS | 1 |
| JPEG/PNG image exports | 11 |
| Raw drone images | 0 |
| Orthomosaics/orthophotos | 12 |
| Thumbnails/previews | 0 |

### C. Timeline inventory

| Date | Total files | Raw NDVI | RGB | Annotation files | Field context rows | Fields |
|---|---:|---:|---:|---:|---:|---|
| 2026-05-08 | 2 | 1 | 0 | 1 | 0 | McIntyre-Road |
| 2026-05-12 | 2 | 1 | 0 | 1 | 0 | Mallard-Avenue |
| 2026-05-18 | 2 | 1 | 0 | 1 | 0 | Mallard-Avenue |
| 2026-05-22 | 2 | 1 | 0 | 1 | 0 | Mallard-Avenue |
| 2026-05-27 | 2 | 1 | 0 | 1 | 0 | McIntyre-Road |
| 2026-05-29 | 2 | 1 | 0 | 1 | 0 | Mallard-Avenue |

- Date source audit:
  - filename: 6 unique dates (`2026-05-08`, `2026-05-12`, `2026-05-18`, `2026-05-22`, `2026-05-27`, `2026-05-29`)
  - folder: 0 unique dates
  - exif: 0 unique dates
  - metadata: 0 unique dates
- Gaps between all detected dates:
  - 2026-05-08 -> 2026-05-12: 4 days
  - 2026-05-12 -> 2026-05-18: 6 days
  - 2026-05-18 -> 2026-05-22: 4 days
  - 2026-05-22 -> 2026-05-27: 5 days
  - 2026-05-27 -> 2026-05-29: 2 days
- Mallard-Avenue field gaps:
  - 2026-05-12 -> 2026-05-18: 6 days
  - 2026-05-18 -> 2026-05-22: 4 days
  - 2026-05-22 -> 2026-05-29: 7 days
- McIntyre-Road field gaps:
  - 2026-05-08 -> 2026-05-27: 19 days

### D. Annotation inventory

| Format | Count | Class labels | Includes disease/stress labels | Includes severity | Geometry | Row/plot IDs | Dates included |
|---|---:|---|---|---|---|---|---|
| annotated_png_raster_overlay | 6 | none detected | no explicit structured labels | no | baked raster overlay only; no separate polygons/masks/bboxes | no | yes |

- No CVAT, COCO JSON, YOLO TXT, Pascal VOC XML, GeoJSON, shapefiles, or spreadsheet label tables were found.

### E. Field context inventory

- No CSV/XLSX/GeoJSON/TXT/YAML field-context files were found in the dataset scan.

## 3. Matching quality

- RGB-NDVI pairing rate: **not applicable** (no RGB images were found, so this metric is not applicable).
- Image-to-annotation match rate: **100.0%** (6/6 raw NDVI images have a same-field same-date annotated counterpart).
- Image-to-field-context match rate: **0.0%** (no joinable context files found).
- Timeline continuity rate:
  - 0 field units have 1 date
  - 1 field units have 2 dates
  - 0 field units have 3 dates
  - 1 field units have 4 or more dates

## 4. Progression feasibility

| Field / unit | Dates available | Image availability | Annotation availability | Context availability | Feasibility level | Recommended use |
|---|---:|---|---|---|---:|---|
| Mallard-Avenue | 4 | NDVI raw=4, georeferenced=1 | 4 overlay images | none | 2 | Whole-field NDVI progression and manual coarse-zone comparison. |
| McIntyre-Road | 2 | NDVI raw=2, georeferenced=0 | 2 overlay images | none | 2 | Whole-field NDVI progression and manual coarse-zone comparison. |

## 5. What we can build now

- Whole-field NDVI timeline by field/date for Mallard-Avenue and McIntyre-Road.
- Manual worsening/stable/improving review using the human-annotated NDVI overlays.
- Simple field-level progression report for farmer or partner demos.
- Prototype grid-cell or coarse-zone registration experiment on Mallard-Avenue only.

## 6. What we cannot reliably build yet

- Reliable disease-specific progression model training.
- Row-level or plant-level progression tracking.
- RGB+NDVI fusion progression modeling, because no RGB timeline exists in this dataset.
- Severity-calibrated supervised learning, because annotations are raster overlays without structured labels.
- Predictive disease progression or treatment response modeling, because there are no treatment/weather/outcome logs.

## 7. Minimum next annotation work

- Convert at least 15 repeatable zones or row sets across 3 Mallard dates into machine-readable polygons with stable zone IDs.
- Assign a 0-5 disease/stress severity score to those same zones on each date.
- Add the same annotation scheme to at least 10 zones across the 2 McIntyre dates.

## 8. Recommended next technical step

- Recommended next step: **build manual annotation template**
- Why: the biggest bottleneck is not date detection; it is the lack of machine-readable, repeatable zone labels across time.

## Final decision

**With the current data, Chimera can currently do: LEVEL 2**

- Why this level was assigned: the dataset has repeated NDVI dates for 2 field units and matching human-marked overlay images, which is enough for basic progression tracking at whole-field / coarse-zone level, but not enough for a reliable annotated disease model.
- What outputs can be generated immediately: field-level NDVI timeline, manual worsening report, and a farmer-demo progression review deck.
- What is missing for the next level: stable machine-readable zone annotations, severity scale, and consistent zone IDs across dates.
- Whether this is enough for a farmer demo: **yes**, for a visual progression story.
- Whether this is enough for model training: **no**.
- Whether this is enough for investor/partner proof: **yes, as feasibility evidence**, but not as a validated predictive dataset.

## Appendix: files scanned

- `human-Annotated-ndvi/Mallard-Avenue-5-12-2026-orthophoto-NDVI-annoated-png.png` | image=True | format=PNG | size=3460x2547 | georef=False | field=Mallard-Avenue | dates=2026-05-12
- `human-Annotated-ndvi/Mallard-Avenue-5-18-2026-orthophoto-NDVI-annotated.png` | image=True | format=PNG | size=3357x2561 | georef=False | field=Mallard-Avenue | dates=2026-05-18
- `human-Annotated-ndvi/Mallard-Avenue-5-22-2026-orthophoto-NDVI-annotated.png` | image=True | format=PNG | size=3430x2512 | georef=False | field=Mallard-Avenue | dates=2026-05-22
- `human-Annotated-ndvi/Mallard-Avenue-5-29-2026-orthophoto-NDVI-user annotated.png` | image=True | format=PNG | size=2889x2355 | georef=False | field=Mallard-Avenue | dates=2026-05-29
- `human-Annotated-ndvi/McIntyre-Road-5-27-2026-orthophoto-NDVI-annotated.png` | image=True | format=PNG | size=2843x2253 | georef=False | field=McIntyre-Road | dates=2026-05-27
- `human-Annotated-ndvi/McIntyre-Road-5-8-2026-orthophoto-NDVI-annotated.png` | image=True | format=PNG | size=13305x9663 | georef=False | field=McIntyre-Road | dates=2026-05-08
- `raw-ndvi/Strawberry_1/Mallard-Avenue-5-12-2026-orthophoto-NDVI.png` | image=True | format=PNG | size=3460x2547 | georef=False | field=Mallard-Avenue | dates=2026-05-12
- `raw-ndvi/Strawberry_1/Mallard-Avenue-5-18-2026-orthophoto-NDVI.tif` | image=True | format=TIFF | size=3357x2561 | georef=True | field=Mallard-Avenue | dates=2026-05-18
- `raw-ndvi/Strawberry_1/Mallard-Avenue-5-22-2026-orthophoto-NDVI.png` | image=True | format=PNG | size=3430x2512 | georef=False | field=Mallard-Avenue | dates=2026-05-22
- `raw-ndvi/Strawberry_1/Mallard-Avenue-5-29-2026-orthophoto-NDVI.png` | image=True | format=PNG | size=2889x2355 | georef=False | field=Mallard-Avenue | dates=2026-05-29
- `raw-ndvi/Strawberry_1/McIntyre-Road-5-27-2026-orthophoto-NDVI.jpg` | image=True | format=JPEG | size=2843x2253 | georef=False | field=McIntyre-Road | dates=2026-05-27
- `raw-ndvi/Strawberry_1/McIntyre-Road-5-8-2026-orthophoto-NDVI.jpg` | image=True | format=JPEG | size=13305x9663 | georef=False | field=McIntyre-Road | dates=2026-05-08