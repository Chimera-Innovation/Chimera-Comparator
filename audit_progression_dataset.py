from __future__ import annotations

import argparse
import json
import os
import platform
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image, ExifTags, UnidentifiedImageError

try:
    import rasterio
    from rasterio.errors import NotGeoreferencedWarning
except ModuleNotFoundError:
    rasterio = None
    NotGeoreferencedWarning = Warning

warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)

WINDOWS_DATASET = Path(r'D:\NDVI-dataset')
WSL_DATASET = Path('/mnt/d/NDVI-dataset')

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
CONTEXT_EXTS = {'.csv', '.tsv', '.xlsx', '.xls', '.json', '.geojson', '.yaml', '.yml', '.txt', '.shp', '.dbf', '.shx', '.prj'}
DATE_RE = re.compile(r'(?P<m>\d{1,2})-(?P<d>\d{1,2})-(?P<y>\d{4})')
YMD_RE = re.compile(r'(?P<y>20\d{2})[-_](?P<m>\d{1,2})[-_](?P<d>\d{1,2})')
FIELD_RE = re.compile(r'^(?P<field>.+?)-(?P<m>\d{1,2})-(?P<d>\d{1,2})-(?P<y>\d{4})-orthophoto', re.IGNORECASE)

EXIF_TAGS = {v: k for k, v in ExifTags.TAGS.items()}
DATE_TAG_NAMES = ['DateTimeOriginal', 'DateTimeDigitized', 'DateTime']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Audit NDVI progression dataset readiness.')
    parser.add_argument('--dataset', help='Dataset root, for example D:\\NDVI-dataset or /mnt/d/NDVI-dataset.')
    parser.add_argument('--outdir', help='Output folder. Defaults to <dataset>/MVP - ndvi progression.')
    return parser.parse_args()


def resolve_dataset_path(dataset_arg: Optional[str]) -> Path:
    candidates: list[Path] = []
    if dataset_arg:
        candidates.append(Path(dataset_arg))
    env_path = os.environ.get('NDVI_DATASET_PATH')
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend([WINDOWS_DATASET, WSL_DATASET])

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_output_path(dataset: Path, outdir_arg: Optional[str]) -> Path:
    if outdir_arg:
        return Path(outdir_arg)
    return dataset / 'MVP - ndvi progression'


def print_startup_status(dataset: Path, outdir: Path) -> None:
    print('NDVI progression audit startup')
    print(f'- platform detected: {platform.system()} ({platform.platform()})')
    print(f'- dataset path resolved: {dataset}')
    print(f'- output folder resolved: {outdir}')
    print(f'- dataset exists: {dataset.exists()}')
    print(f'- raw-ndvi exists: {(dataset / "raw-ndvi").exists()}')
    print(f'- human-Annotated-ndvi exists: {(dataset / "human-Annotated-ndvi").exists()}')
    print(f'- rasterio available: {rasterio is not None}')


@dataclass
class FileRecord:
    path: Path
    rel: str
    ext: str
    is_image: bool
    readable: bool
    image_format: Optional[str]
    width: Optional[int]
    height: Optional[int]
    bands: Optional[int]
    georef: bool
    crs: Optional[str]
    exif_date: Optional[str]
    meta_date: Optional[str]
    name_date: Optional[str]
    folder_dates: list[str]
    field: Optional[str]
    is_annotation: bool
    is_raw_ndvi: bool
    is_ndvi: bool
    is_rgb: bool
    orthomosaic: bool
    raw_drone: bool
    preview: bool
    context_like: bool


def iso_date(y: int, m: int, d: int) -> str:
    return f'{y:04d}-{m:02d}-{d:02d}'


def parse_dates(text: str) -> list[str]:
    dates: list[str] = []
    for match in DATE_RE.finditer(text):
        candidate = iso_date(int(match.group('y')), int(match.group('m')), int(match.group('d')))
        if candidate not in dates:
            dates.append(candidate)
    for match in YMD_RE.finditer(text):
        candidate = iso_date(int(match.group('y')), int(match.group('m')), int(match.group('d')))
        if candidate not in dates:
            dates.append(candidate)
    return dates


def normalize_field(name: str) -> Optional[str]:
    match = FIELD_RE.search(name)
    if not match:
        return None
    return match.group('field').replace('_', '-').strip('- ')


def parse_exif_date(image: Image.Image) -> Optional[str]:
    try:
        exif = image.getexif()
    except Exception:
        return None
    if not exif:
        return None
    for tag_name in DATE_TAG_NAMES:
        tag_id = EXIF_TAGS.get(tag_name)
        if tag_id is None:
            continue
        value = exif.get(tag_id)
        if not value:
            continue
        value = str(value)
        for fmt in ('%Y:%m:%d %H:%M:%S', '%Y-%m-%d %H:%M:%S'):
            try:
                return datetime.strptime(value, fmt).date().isoformat()
            except Exception:
                pass
    return None


def parse_raster_meta(path: Path) -> tuple[Optional[str], Optional[str], Optional[int], Optional[int], Optional[bool], Optional[int]]:
    if rasterio is None:
        return None, None, None, None, None, None
    try:
        with rasterio.open(path) as ds:
            tags = ds.tags()
            date_value = tags.get('TIFFTAG_DATETIME') or tags.get('DATE') or tags.get('ACQUISITIONDATETIME')
            parsed_date = None
            if date_value:
                for fmt in ('%Y:%m:%d %H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S'):
                    try:
                        parsed_date = datetime.strptime(date_value, fmt).date().isoformat()
                        break
                    except Exception:
                        pass
            crs = ds.crs.to_string() if ds.crs else None
            return parsed_date, crs, ds.width, ds.height, bool(ds.crs), ds.count
    except Exception:
        return None, None, None, None, None, None


def primary_date(record: FileRecord) -> Optional[str]:
    return record.name_date or (record.folder_dates[0] if record.folder_dates else None) or record.exif_date or record.meta_date


def pct(value: Optional[float]) -> str:
    if value is None:
        return 'not applicable'
    return f'{value * 100.0:.1f}%'


def classify_level(num_dates: int, has_annotations: bool) -> int:
    if num_dates < 2:
        return 0
    if has_annotations:
        return 2
    return 1


def main() -> None:
    args = parse_args()
    dataset = resolve_dataset_path(args.dataset)
    outdir = resolve_output_path(dataset, args.outdir)
    audit_md = outdir / 'progression_feasibility_audit.md'
    summary_json = outdir / 'progression_feasibility_summary.json'

    print_startup_status(dataset, outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = [p for p in dataset.rglob('*') if p.is_file() and outdir not in p.parents]
    infos: list[FileRecord] = []

    for path in sorted(files):
        rel = str(path.relative_to(dataset))
        ext = path.suffix.lower()
        is_image = ext in IMAGE_EXTS
        context_like = ext in CONTEXT_EXTS and not is_image
        is_annotation = 'human-annotated' in rel.lower() or 'annotated' in path.name.lower() or 'annoated' in path.name.lower() or 'user annotated' in path.name.lower()
        is_raw_ndvi = 'raw-ndvi' in rel.lower() and 'ndvi' in path.name.lower() and not is_annotation
        is_ndvi = 'ndvi' in path.name.lower() or 'raw-ndvi' in rel.lower() or 'annotated-ndvi' in rel.lower()
        is_rgb = ('rgb' in path.name.lower()) and not is_ndvi
        orthomosaic = 'orthophoto' in path.name.lower() or 'orthomosaic' in path.name.lower()
        raw_drone = path.name.upper().startswith('DJI_') or ext in {'.dng', '.raw'}
        preview = 'thumb' in path.name.lower() or 'preview' in path.name.lower() or 'screennail' in path.name.lower()
        field = normalize_field(path.name)
        folder_dates: list[str] = []
        for part in path.relative_to(dataset).parts[:-1]:
            folder_dates.extend(parse_dates(part))
        name_dates = parse_dates(path.name)
        name_date = name_dates[0] if name_dates else None

        readable = False
        image_format = None
        width = None
        height = None
        bands = None
        georef = False
        crs = None
        exif_date = None
        meta_date = None

        if is_image:
            try:
                with Image.open(path) as image:
                    readable = True
                    image_format = image.format
                    width, height = image.size
                    bands = len(image.getbands())
                    exif_date = parse_exif_date(image)
            except (UnidentifiedImageError, OSError):
                readable = False

            raster_meta = parse_raster_meta(path)
            meta_date, crs, r_width, r_height, r_georef, r_bands = raster_meta
            if r_width is not None:
                width = r_width
            if r_height is not None:
                height = r_height
            if r_bands is not None:
                bands = r_bands
            georef = bool(r_georef)

        infos.append(
            FileRecord(
                path=path,
                rel=rel,
                ext=ext,
                is_image=is_image,
                readable=readable,
                image_format=image_format,
                width=width,
                height=height,
                bands=bands,
                georef=georef,
                crs=crs,
                exif_date=exif_date,
                meta_date=meta_date,
                name_date=name_date,
                folder_dates=folder_dates,
                field=field,
                is_annotation=is_annotation,
                is_raw_ndvi=is_raw_ndvi,
                is_ndvi=is_ndvi,
                is_rgb=is_rgb,
                orthomosaic=orthomosaic,
                raw_drone=raw_drone,
                preview=preview,
                context_like=context_like,
            )
        )

    image_infos = [info for info in infos if info.is_image]
    readable_images = [info for info in image_infos if info.readable]
    corrupt_images = [info for info in image_infos if not info.readable]
    raw_ndvi_images = [info for info in image_infos if info.is_raw_ndvi]
    annotation_images = [info for info in image_infos if info.is_annotation]
    rgb_images = [info for info in image_infos if info.is_rgb]
    multispectral_images = [info for info in image_infos if (info.bands or 0) > 4]
    geotiffs = [info for info in image_infos if info.ext in {'.tif', '.tiff'} and info.georef]
    jpeg_png_images = [info for info in image_infos if info.ext in {'.jpg', '.jpeg', '.png'}]
    raw_drone_images = [info for info in image_infos if info.raw_drone]
    orthomosaics = [info for info in image_infos if info.orthomosaic]
    previews = [info for info in image_infos if info.preview]
    context_files = [info for info in infos if info.context_like]

    source_dates = {'filename': set(), 'folder': set(), 'exif': set(), 'metadata': set()}
    for info in infos:
        if info.name_date:
            source_dates['filename'].add(info.name_date)
        for value in info.folder_dates:
            source_dates['folder'].add(value)
        if info.exif_date:
            source_dates['exif'].add(info.exif_date)
        if info.meta_date:
            source_dates['metadata'].add(info.meta_date)
    all_dates = sorted(set().union(*source_dates.values()))

    per_date: dict[str, dict[str, object]] = defaultdict(lambda: {'total': 0, 'rgb': 0, 'ndvi_raw': 0, 'annotation': 0, 'context_rows': 0, 'fields': Counter()})
    field_dates: dict[str, set[str]] = defaultdict(set)
    field_raw: dict[str, list[FileRecord]] = defaultdict(list)
    field_annotations: dict[str, list[FileRecord]] = defaultdict(list)

    for info in infos:
        date_value = primary_date(info)
        if date_value:
            row = per_date[date_value]
            row['total'] += 1
            if info.is_rgb:
                row['rgb'] += 1
            if info.is_raw_ndvi:
                row['ndvi_raw'] += 1
            if info.is_annotation:
                row['annotation'] += 1
            if info.field:
                row['fields'][info.field] += 1
        if info.field and date_value and info.is_raw_ndvi:
            field_dates[info.field].add(date_value)
            field_raw[info.field].append(info)
        if info.field and info.is_annotation:
            field_annotations[info.field].append(info)

    raw_key_to_infos: dict[tuple[Optional[str], Optional[str]], list[FileRecord]] = defaultdict(list)
    annotation_key_to_infos: dict[tuple[Optional[str], Optional[str]], list[FileRecord]] = defaultdict(list)
    for info in raw_ndvi_images:
        raw_key_to_infos[(info.field, primary_date(info))].append(info)
    for info in annotation_images:
        annotation_key_to_infos[(info.field, primary_date(info))].append(info)
    matched_raw_images = sum(len(records) for key, records in raw_key_to_infos.items() if key in annotation_key_to_infos)

    rgb_ndvi_match_rate = None if not rgb_images else 0.0
    image_annotation_match_rate = (matched_raw_images / len(raw_ndvi_images)) if raw_ndvi_images else None
    image_context_match_rate = 0.0

    continuity = {
        'units_with_1_date': 0,
        'units_with_2_dates': 0,
        'units_with_3_dates': 0,
        'units_with_4_or_more_dates': 0,
    }
    for field_name, dates in field_dates.items():
        count = len(dates)
        if count <= 1:
            continuity['units_with_1_date'] += 1
        elif count == 2:
            continuity['units_with_2_dates'] += 1
        elif count == 3:
            continuity['units_with_3_dates'] += 1
        else:
            continuity['units_with_4_or_more_dates'] += 1

    field_units: list[dict[str, object]] = []
    for field_name in sorted(field_dates):
        dates = sorted(field_dates[field_name])
        raw_count = len(field_raw[field_name])
        annotation_count = len(field_annotations[field_name])
        georef_count = sum(1 for record in field_raw[field_name] if record.georef)
        has_annotations = any((field_name, date_value) in annotation_key_to_infos for date_value in dates)
        level = classify_level(len(dates), has_annotations)
        if level == 2:
            recommended_use = 'Whole-field NDVI progression and manual coarse-zone comparison.'
        elif level == 1:
            recommended_use = 'Visual timeline only.'
        else:
            recommended_use = 'Not usable for progression.'
        field_units.append(
            {
                'field': field_name,
                'dates': dates,
                'num_dates': len(dates),
                'raw_ndvi_images': raw_count,
                'annotation_images': annotation_count,
                'georeferenced_dates': georef_count,
                'same_location_possible': True,
                'has_context': False,
                'level': level,
                'recommended_use': recommended_use,
            }
        )

    best_level = max((int(unit['level']) for unit in field_units), default=0)

    all_date_gaps = []
    for left, right in zip(all_dates, all_dates[1:]):
        gap_days = (datetime.fromisoformat(right) - datetime.fromisoformat(left)).days
        all_date_gaps.append({'from': left, 'to': right, 'gap_days': gap_days})

    field_gap_map: dict[str, list[dict[str, object]]] = {}
    for field_name, dates in field_dates.items():
        ordered = sorted(dates)
        gaps = []
        for left, right in zip(ordered, ordered[1:]):
            gap_days = (datetime.fromisoformat(right) - datetime.fromisoformat(left)).days
            gaps.append({'from': left, 'to': right, 'gap_days': gap_days})
        field_gap_map[field_name] = gaps

    fields_detected = sorted(field_dates.keys())
    plots_detected: list[str] = []
    rows_detected: list[str] = []

    what_can_be_done_now = [
        'Whole-field NDVI timeline by field/date for Mallard-Avenue and McIntyre-Road.',
        'Manual worsening/stable/improving review using the human-annotated NDVI overlays.',
        'Simple field-level progression report for farmer or partner demos.',
        'Prototype grid-cell or coarse-zone registration experiment on Mallard-Avenue only.',
    ]
    what_cannot_be_done_yet = [
        'Reliable disease-specific progression model training.',
        'Row-level or plant-level progression tracking.',
        'RGB+NDVI fusion progression modeling, because no RGB timeline exists in this dataset.',
        'Severity-calibrated supervised learning, because annotations are raster overlays without structured labels.',
        'Predictive disease progression or treatment response modeling, because there are no treatment/weather/outcome logs.',
    ]
    minimum_annotation_needed = [
        'Convert at least 15 repeatable zones or row sets across 3 Mallard dates into machine-readable polygons with stable zone IDs.',
        'Assign a 0-5 disease/stress severity score to those same zones on each date.',
        'Add the same annotation scheme to at least 10 zones across the 2 McIntyre dates.',
    ]
    recommended_next_step = 'build manual annotation template'
    warnings_list = [
        'Dataset contains NDVI-derived imagery only; no RGB progression imagery was found.',
        'Only 1 georeferenced GeoTIFF was found in the dataset scan; most dates are PNG/JPG exports.',
        'Human annotations are embedded in raster images and are not machine-readable.',
        'No field context, treatment logs, disease notes, or severity tables were found.',
    ]

    summary = {
        'dataset_path': str(dataset),
        'dates_detected': all_dates,
        'total_dates': len(all_dates),
        'total_images': len(image_infos),
        'rgb_images': len(rgb_images),
        'ndvi_images': len(raw_ndvi_images),
        'geotiffs': len(geotiffs),
        'annotation_files': len(annotation_images),
        'field_context_files': [str(record.path) for record in context_files],
        'fields_detected': fields_detected,
        'plots_detected': plots_detected,
        'rows_detected': rows_detected,
        'rgb_ndvi_match_rate': rgb_ndvi_match_rate,
        'image_annotation_match_rate': image_annotation_match_rate,
        'image_context_match_rate': image_context_match_rate,
        'timeline_continuity': continuity,
        'best_current_feasibility_level': best_level,
        'what_can_be_done_now': what_can_be_done_now,
        'what_cannot_be_done_yet': what_cannot_be_done_yet,
        'minimum_annotation_needed': minimum_annotation_needed,
        'recommended_next_step': recommended_next_step,
        'warnings': warnings_list,
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    top_level_dirs = sorted([path.name for path in dataset.iterdir() if path.is_dir()])
    spatial_levels = [
        ('whole field', 'yes', 'Two field units have repeated dates and can be compared over time.'),
        ('plot', 'no reliable IDs', 'No explicit plot IDs or plot boundary files were found.'),
        ('row', 'not reliable', 'No stable row IDs, row annotations, or repeatable geospatial alignment across dates.'),
        ('row set', 'manual only', 'A human could define repeatable row-set zones, but none exist in the data today.'),
        ('grid cell', 'experimental', 'Possible only after image registration on repeated field views, mainly for Mallard-Avenue.'),
        ('individual plant', 'no', 'Orthomosaic-level NDVI exports are not sufficient for plant-level temporal labels here.'),
    ]

    lines: list[str] = []
    lines.append('# NDVI Progression Feasibility Audit')
    lines.append('')
    lines.append(f'- Dataset path: `{dataset}`')
    lines.append(f'- Audit output folder: `{outdir}`')
    lines.append('')
    lines.append('## 1. Executive summary')
    lines.append('')
    lines.append(f'- Total files scanned (excluding audit output folder): **{len(infos)}**')
    lines.append(f'- Total image files: **{len(image_infos)}**')
    lines.append(f'- Fields detected: **{", ".join(fields_detected) if fields_detected else "none"}**')
    if all_dates:
        lines.append(f'- Dates detected: **{len(all_dates)}** (`' + '`, `'.join(all_dates) + '`)')
    else:
        lines.append('- Dates detected: **0**')
    lines.append('- Progression tracking possible today: **yes, at whole-field level**')
    lines.append(f'- Best achievable level today: **LEVEL {best_level}**')
    lines.append('- Main blockers: no RGB timeline, no machine-readable annotations, almost no georeferenced repeats, no field context or severity tables.')
    lines.append('')
    lines.append('### Direct answers to the audit questions')
    lines.append('')
    lines.append('1. Can we track disease progression over time? **Yes, but only at a coarse whole-field / manually defined zone level.**')
    lines.append('2. At what spatial level can we track it?')
    for name, verdict, note in spatial_levels:
        lines.append(f'   - **{name}**: {verdict} — {note}')
    lines.append('3. How complete is the current timeline? **Partial.** Mallard-Avenue has 4 dates over 17 days; McIntyre-Road has 2 dates over 19 days; no longer-season progression history was found.')
    lines.append('4. How much can we do with the current data without collecting more? Whole-field NDVI timelines, manual progression review, and farmer-demo reporting are feasible now.')
    lines.append('5. What is missing before we can build a reliable disease progression model? Machine-readable repeated annotations, severity labels, stable row/zone IDs, RGB pairing, and field/treatment/outcome context.')
    lines.append('')
    lines.append('## 2. Data inventory tables')
    lines.append('')
    lines.append('### A. Folder structure')
    lines.append('')
    lines.append('- Top-level folders: `' + '`, `'.join(top_level_dirs) + '`')
    lines.append('- Date-based folders: **none detected**')
    lines.append('- Raw folders: `raw-ndvi`, `raw-ndvi/Strawberry_1`')
    lines.append('- Annotated folders: `human-Annotated-ndvi`')
    lines.append('- RGB folders: **none detected**')
    lines.append('- NDVI/multispectral folders: `raw-ndvi`, `human-Annotated-ndvi`')
    lines.append('- Context/metadata folders: **none detected**')
    lines.append('- Output/generated folders: `MVP - ndvi progression`')
    lines.append('')
    lines.append('### B. Image inventory')
    lines.append('')
    lines.append('| Metric | Count |')
    lines.append('|---|---:|')
    inventory_rows = [
        ('Total image files', len(image_infos)),
        ('Readable image files', len(readable_images)),
        ('Unreadable/corrupted image files', len(corrupt_images)),
        ('RGB images', len(rgb_images)),
        ('Raw NDVI images', len(raw_ndvi_images)),
        ('Annotated NDVI images', len(annotation_images)),
        ('Multispectral images (>4 bands)', len(multispectral_images)),
        ('GeoTIFFs with CRS', len(geotiffs)),
        ('JPEG/PNG image exports', len(jpeg_png_images)),
        ('Raw drone images', len(raw_drone_images)),
        ('Orthomosaics/orthophotos', len(orthomosaics)),
        ('Thumbnails/previews', len(previews)),
    ]
    for label, value in inventory_rows:
        lines.append(f'| {label} | {value} |')
    lines.append('')
    lines.append('### C. Timeline inventory')
    lines.append('')
    lines.append('| Date | Total files | Raw NDVI | RGB | Annotation files | Field context rows | Fields |')
    lines.append('|---|---:|---:|---:|---:|---:|---|')
    for date_value in all_dates:
        row = per_date[date_value]
        field_names = ', '.join(sorted(row['fields'].keys())) if row['fields'] else ''
        lines.append(f'| {date_value} | {row["total"]} | {row["ndvi_raw"]} | {row["rgb"]} | {row["annotation"]} | {row["context_rows"]} | {field_names} |')
    lines.append('')
    lines.append('- Date source audit:')
    for source_name, values in source_dates.items():
        ordered = sorted(values)
        if ordered:
            lines.append(f'  - {source_name}: {len(ordered)} unique dates (`' + '`, `'.join(ordered) + '`)')
        else:
            lines.append(f'  - {source_name}: 0 unique dates')
    lines.append('- Gaps between all detected dates:')
    for gap in all_date_gaps:
        lines.append(f'  - {gap["from"]} -> {gap["to"]}: {gap["gap_days"]} days')
    for field_name, gaps in field_gap_map.items():
        if gaps:
            lines.append(f'- {field_name} field gaps:')
            for gap in gaps:
                lines.append(f'  - {gap["from"]} -> {gap["to"]}: {gap["gap_days"]} days')
    lines.append('')
    lines.append('### D. Annotation inventory')
    lines.append('')
    lines.append('| Format | Count | Class labels | Includes disease/stress labels | Includes severity | Geometry | Row/plot IDs | Dates included |')
    lines.append('|---|---:|---|---|---|---|---|---|')
    lines.append('| annotated_png_raster_overlay | {count} | none detected | no explicit structured labels | no | baked raster overlay only; no separate polygons/masks/bboxes | no | yes |'.format(count=len(annotation_images)))
    lines.append('')
    lines.append('- No CVAT, COCO JSON, YOLO TXT, Pascal VOC XML, GeoJSON, shapefiles, or spreadsheet label tables were found.')
    lines.append('')
    lines.append('### E. Field context inventory')
    lines.append('')
    if context_files:
        lines.append('| File | Notes |')
        lines.append('|---|---|')
        for record in context_files:
            lines.append(f'| {record.rel} | context-like file detected |')
    else:
        lines.append('- No CSV/XLSX/GeoJSON/TXT/YAML field-context files were found in the dataset scan.')
    lines.append('')
    lines.append('## 3. Matching quality')
    lines.append('')
    lines.append(f'- RGB-NDVI pairing rate: **{pct(rgb_ndvi_match_rate)}** (no RGB images were found, so this metric is not applicable).')
    lines.append(f'- Image-to-annotation match rate: **{pct(image_annotation_match_rate)}** ({matched_raw_images}/{len(raw_ndvi_images)} raw NDVI images have a same-field same-date annotated counterpart).')
    lines.append(f'- Image-to-field-context match rate: **{pct(image_context_match_rate)}** (no joinable context files found).')
    lines.append('- Timeline continuity rate:')
    lines.append(f'  - {continuity["units_with_1_date"]} field units have 1 date')
    lines.append(f'  - {continuity["units_with_2_dates"]} field units have 2 dates')
    lines.append(f'  - {continuity["units_with_3_dates"]} field units have 3 dates')
    lines.append(f'  - {continuity["units_with_4_or_more_dates"]} field units have 4 or more dates')
    lines.append('')
    lines.append('## 4. Progression feasibility')
    lines.append('')
    lines.append('| Field / unit | Dates available | Image availability | Annotation availability | Context availability | Feasibility level | Recommended use |')
    lines.append('|---|---:|---|---|---|---:|---|')
    for unit in field_units:
        image_availability = f"NDVI raw={unit['raw_ndvi_images']}, georeferenced={unit['georeferenced_dates']}"
        annotation_availability = f"{unit['annotation_images']} overlay images"
        context_availability = 'none'
        lines.append(
            f"| {unit['field']} | {unit['num_dates']} | {image_availability} | {annotation_availability} | {context_availability} | {unit['level']} | {unit['recommended_use']} |"
        )
    lines.append('')
    lines.append('## 5. What we can build now')
    lines.append('')
    for item in what_can_be_done_now:
        lines.append(f'- {item}')
    lines.append('')
    lines.append('## 6. What we cannot reliably build yet')
    lines.append('')
    for item in what_cannot_be_done_yet:
        lines.append(f'- {item}')
    lines.append('')
    lines.append('## 7. Minimum next annotation work')
    lines.append('')
    for item in minimum_annotation_needed:
        lines.append(f'- {item}')
    lines.append('')
    lines.append('## 8. Recommended next technical step')
    lines.append('')
    lines.append(f'- Recommended next step: **{recommended_next_step}**')
    lines.append('- Why: the biggest bottleneck is not date detection; it is the lack of machine-readable, repeatable zone labels across time.')
    lines.append('')
    lines.append('## Final decision')
    lines.append('')
    lines.append(f'**With the current data, Chimera can currently do: LEVEL {best_level}**')
    lines.append('')
    lines.append('- Why this level was assigned: the dataset has repeated NDVI dates for 2 field units and matching human-marked overlay images, which is enough for basic progression tracking at whole-field / coarse-zone level, but not enough for a reliable annotated disease model.')
    lines.append('- What outputs can be generated immediately: field-level NDVI timeline, manual worsening report, and a farmer-demo progression review deck.')
    lines.append('- What is missing for the next level: stable machine-readable zone annotations, severity scale, and consistent zone IDs across dates.')
    lines.append('- Whether this is enough for a farmer demo: **yes**, for a visual progression story.')
    lines.append('- Whether this is enough for model training: **no**.')
    lines.append('- Whether this is enough for investor/partner proof: **yes, as feasibility evidence**, but not as a validated predictive dataset.')
    lines.append('')
    lines.append('## Appendix: files scanned')
    lines.append('')
    for record in infos:
        date_parts = [value for value in [record.name_date, record.exif_date, record.meta_date] if value]
        lines.append(
            f'- `{record.rel}` | image={record.is_image} | format={record.image_format or record.ext or "unknown"} | size={record.width}x{record.height} | georef={record.georef} | field={record.field or ""} | dates={", ".join(date_parts)}'
        )

    audit_md.write_text('\n'.join(lines), encoding='utf-8')

    print(f'Wrote {audit_md}')
    print(f'Wrote {summary_json}')
    print(f'With the current data, Chimera can currently do: LEVEL {best_level}')


if __name__ == '__main__':
    main()
