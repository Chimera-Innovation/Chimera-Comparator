from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImagePair:
    date: str
    raw_path: Path
    annotated_path: Path
    raw_source_label: str
    annotated_source_label: str


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {folder}")
    return sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def parse_date_from_name(path: Path) -> str | None:
    match = re.search(r"(?P<month>\d{1,2})-(?P<day>\d{1,2})-(?P<year>\d{4})", path.name)
    if not match:
        return None
    return f"{int(match.group('year')):04d}-{int(match.group('month')):02d}-{int(match.group('day')):02d}"


def parse_source_label(path: Path) -> str:
    match = re.match(r"(?P<label>.+?)-\d{1,2}-\d{1,2}-\d{4}-orthophoto-NDVI", path.name)
    if match:
        return match.group("label")
    return path.stem


def index_by_date(paths: list[Path]) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in paths:
        date = parse_date_from_name(path)
        if not date:
            print(f"[WARN] Skipping file without parseable date: {path}")
            continue
        existing = indexed.get(date)
        if existing:
            # Prefer newer-looking annotation-output files when duplicate dates exist.
            if "annotated-output" in path.name.lower() or "annotated" in path.name.lower() and "annotated" not in existing.name.lower():
                indexed[date] = path
            else:
                print(f"[WARN] Duplicate date {date}; keeping {existing.name}, skipping {path.name}")
        else:
            indexed[date] = path
    return indexed


def match_image_pairs(raw_dir: Path, annotated_dir: Path) -> list[ImagePair]:
    pairs, _ = match_image_pairs_with_audit(raw_dir, annotated_dir)
    return pairs


def match_image_pairs_with_audit(raw_dir: Path, annotated_dir: Path) -> tuple[list[ImagePair], list[dict[str, str]]]:
    raw_by_date = index_by_date(list_images(raw_dir))
    annotated_by_date = index_by_date(list_images(annotated_dir))
    pairs: list[ImagePair] = []
    audit_rows: list[dict[str, str]] = []
    for date, annotated_path in sorted(annotated_by_date.items()):
        raw_path = raw_by_date.get(date)
        if not raw_path:
            print(f"[WARN] No raw image match for annotated date {date}: {annotated_path.name}")
            audit_rows.append(
                {
                    "status": "WARNING",
                    "check": "missing raw/annotated pair",
                    "date": date,
                    "detail": "An annotated image had no matching raw image and was excluded from this release.",
                }
            )
            continue
        pair = ImagePair(
            date=date,
            raw_path=raw_path,
            annotated_path=annotated_path,
            raw_source_label=parse_source_label(raw_path),
            annotated_source_label=parse_source_label(annotated_path),
        )
        print(f"[MATCH] {date}: raw={raw_path.name} annotated={annotated_path.name}")
        pairs.append(pair)
    unmatched_raw = sorted(set(raw_by_date) - set(annotated_by_date))
    for date in unmatched_raw:
        print(f"[WARN] Raw image has no annotation for date {date}: {raw_by_date[date].name}")
        audit_rows.append(
            {
                "status": "WARNING",
                "check": "missing raw/annotated pair",
                "date": date,
                "detail": "A raw image had no matching annotation and was excluded from this release.",
            }
        )
    return pairs, audit_rows
