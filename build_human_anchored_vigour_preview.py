from __future__ import annotations

import csv
import html
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "strawberry_vigour_polygon_report"))

from src.annotation_parser import get_annotation_mask, human_annotation_colour_masks, resize_annotated_to_raw
from src.image_matching import ImagePair, index_by_date, list_images, match_image_pairs_with_audit, parse_source_label
from src.main import (
    class_statistics_from_seeds,
    classify_canopy_by_ndvi,
    combined_segmentation_image,
    filled_region_masks,
    labels_to_class_masks,
    ndvi_feature_stack,
    panel_image,
    seed_polygon_masks,
    smooth_classification,
    smooth_ndvi_features,
    transparent_overlay,
    canopy_mask_from_raw,
)


RAW_DIRS = [
    Path(r"C:\Users\Chimera\Downloads\Strawberry 1"),
    Path(r"D:\NDVI-dataset\raw-ndvi\Strawberry_1"),
]
ANNOTATED_DIR = Path(r"C:\Users\Chimera\Downloads\Strawberry 1 - annotated")
OUTPUT_DIR = ROOT / "human_anchored_vigour_zone_preview_product"
FIELD_NAME = "Strawberry 1"
PRODUCT_NAME = "Human-Anchored Vigour Zone Preview"
PRODUCT_WORDING = (
    "Generated from human-marked regions and rendered NDVI colour response. "
    "For scouting prioritization only. Not a disease diagnosis or validated automatic classification."
)


def resize_pair_for_batch(raw_bgr: np.ndarray, annotated_bgr: np.ndarray, max_dimension: int = 3600) -> tuple[np.ndarray, np.ndarray, float]:
    height, width = raw_bgr.shape[:2]
    largest = max(height, width)
    if largest <= max_dimension:
        return raw_bgr, annotated_bgr, 1.0
    scale = max_dimension / largest
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return (
        cv2.resize(raw_bgr, size, interpolation=cv2.INTER_AREA),
        cv2.resize(annotated_bgr, size, interpolation=cv2.INTER_AREA),
        scale,
    )


def save(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    height = min(left.shape[0], right.shape[0])
    def fit(image: np.ndarray) -> np.ndarray:
        scale = height / image.shape[0]
        width = max(1, int(round(image.shape[1] * scale)))
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return np.hstack([fit(left), fit(right)])


def concern_panel(raw_bgr: np.ndarray, preview_labels: np.ndarray, canopy_mask: np.ndarray, stats: dict[str, dict[str, object]]) -> tuple[np.ndarray, float, float]:
    features = smooth_ndvi_features(ndvi_feature_stack(raw_bgr), canopy_mask)
    ordered = ("low_vigour", "medium_vigour", "high_vigour")
    centers = np.array([float(np.asarray(stats[name]["mean"], dtype=np.float32)[1]) for name in ordered], dtype=np.float32)
    proxy = features[:, :, 1]
    dist = np.dstack([np.abs(proxy - center) for center in centers])
    nearest = np.argmin(dist, axis=2).astype(np.uint8) + 1
    nearest[canopy_mask == 0] = 0
    sorted_dist = np.sort(dist, axis=2)
    margin = sorted_dist[:, :, 1] - sorted_dist[:, :, 0]
    weak = (canopy_mask > 0) & (margin < 0.025)
    disagree = (preview_labels > 0) & (nearest > 0) & (preview_labels != nearest)
    panel = (raw_bgr.astype(np.float32) * 0.45).astype(np.uint8)
    panel[weak] = (0, 210, 255)
    panel[disagree] = (30, 30, 255)
    panel[canopy_mask == 0] = (55, 55, 55)
    canopy_count = max(1, int(np.count_nonzero(canopy_mask)))
    return panel, int(np.count_nonzero(weak)) / canopy_count, int(np.count_nonzero(disagree)) / canopy_count


def six_panel_sheet(
    raw_bgr: np.ndarray,
    outlines: np.ndarray,
    seeds: np.ndarray,
    preview: np.ndarray,
    raw_vs_preview: np.ndarray,
    concern: np.ndarray,
) -> np.ndarray:
    panels = [
        panel_image(raw_bgr, "1. Raw NDVI/background", (520, 360)),
        panel_image(outlines, "2. Human annotation outlines", (520, 360)),
        panel_image(seeds, "3. Seed polygons used", (520, 360)),
        panel_image(preview, "4. Human-anchored vigour preview", (520, 360)),
        panel_image(raw_vs_preview, "5. Raw + preview side-by-side", (520, 360)),
        panel_image(concern, "6. Concern / uncertainty", (520, 360)),
    ]
    return np.vstack([np.hstack(panels[:3]), np.hstack(panels[3:])])


def write_timeline(rows: list[dict[str, object]]) -> None:
    cards = []
    for row in rows:
        image = cv2.imread(str(Path(row["six_panel_path"])), cv2.IMREAD_COLOR)
        if image is None:
            continue
        title = f"{row['date']} - {row['status']}"
        cards.append(panel_image(image, title, (520, 360)))
    if not cards:
        return
    per_row = 2
    blanks_needed = (-len(cards)) % per_row
    cards.extend([np.full_like(cards[0], 255) for _ in range(blanks_needed)])
    sheet_rows = [np.hstack(cards[i : i + per_row]) for i in range(0, len(cards), per_row)]
    save(OUTPUT_DIR / "timeline_contact_sheet.png", np.vstack(sheet_rows))


def relative_path(path: Path) -> str:
    return path.relative_to(OUTPUT_DIR).as_posix()


def write_report(rows: list[dict[str, object]]) -> None:
    latest = rows[-1] if rows else None
    timeline_img = "timeline_contact_sheet.png"
    latest_dir = Path(latest["date"]) if latest else Path(".")
    latest_preview = latest_dir / "human_anchored_vigour_preview.png"
    latest_proof = latest_dir / "six_panel_product_proof.png"
    latest_raw_vs = latest_dir / "raw_vs_preview_side_by_side.png"
    latest_concern = latest_dir / "concern_uncertainty_panel.png"
    latest_outlines = latest_dir / "human_annotation_outlines.png"
    rows_html = "\n".join(
        f"<tr><td>{html.escape(str(row['date']))}</td><td>{html.escape(str(row['status']))}</td>"
        f"<td>{row['seed_pixels']}</td><td>{row['preview_pixels']}</td>"
        f"<td>{row['weak_uncertainty_percent']:.2f}%</td><td>{row['disagreement_percent']:.2f}%</td>"
        f"<td>{html.escape(str(row['review_reason']))}</td></tr>"
        for row in rows
    )
    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{FIELD_NAME} &mdash; {PRODUCT_NAME}</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #081018; color: #f5f7fb; }}
    header {{ padding: 24px 28px; background: #0c1722; border-bottom: 1px solid #243447; }}
    main {{ padding: 22px; display: grid; gap: 22px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    h2 {{ margin: 0 0 12px; }}
    .sub {{ color: #aeb9c7; }}
    .section {{ background: #101b27; border: 1px solid #243447; padding: 16px; }}
    img {{ max-width: 100%; display: block; }}
    table {{ width: 100%; border-collapse: collapse; color: #f5f7fb; }}
    th, td {{ border-bottom: 1px solid #243447; padding: 8px; text-align: left; }}
    th {{ color: #c8d3df; }}
    footer {{ padding: 20px 28px 30px; color: #aeb9c7; }}
  </style>
</head>
<body>
  <header>
    <h1>{FIELD_NAME} &mdash; {PRODUCT_NAME}</h1>
    <div class="sub">{PRODUCT_WORDING}</div>
  </header>
  <main>
    <section class="section">
      <h2>1. Latest Human-Anchored Vigour Preview</h2>
      <img src="{relative_path(OUTPUT_DIR / latest_preview) if latest else ''}" alt="Latest Human-Anchored Vigour Zone Preview">
    </section>
    <section class="section">
      <h2>2. Date Timeline</h2>
      <img src="{timeline_img}" alt="Timeline contact sheet">
    </section>
    <section class="section">
      <h2>3. Human Annotation Reference</h2>
      <img src="{relative_path(OUTPUT_DIR / latest_outlines) if latest else ''}" alt="Human annotation reference">
    </section>
    <section class="section">
      <h2>4. Preview vs Raw NDVI</h2>
      <img src="{relative_path(OUTPUT_DIR / latest_raw_vs) if latest else ''}" alt="Preview vs raw NDVI">
    </section>
    <section class="section">
      <h2>5. Concern / Uncertainty</h2>
      <img src="{relative_path(OUTPUT_DIR / latest_concern) if latest else ''}" alt="Concern and uncertainty">
    </section>
    <section class="section">
      <h2>6. Technical Audit Note</h2>
      <p>{PRODUCT_WORDING}</p>
      <p>RGB-derived NDVI proxy separability was previously tested and did not support a validated full-field automatic classification. These previews stay anchored to human-marked regions and are exploratory.</p>
      <table>
        <thead><tr><th>Date</th><th>Status</th><th>Seed pixels</th><th>Preview pixels</th><th>Weak uncertainty</th><th>Disagreement</th><th>Review reason</th></tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </section>
    <section class="section">
      <h2>Six-Panel Latest Proof</h2>
      <img src="{relative_path(OUTPUT_DIR / latest_proof) if latest else ''}" alt="Latest six-panel proof">
    </section>
  </main>
  <footer>{PRODUCT_WORDING}</footer>
</body>
</html>
"""
    (OUTPUT_DIR / "product_report.html").write_text(doc, encoding="utf-8")


def process_pair(pair) -> dict[str, object] | None:
    raw_bgr = cv2.imread(str(pair.raw_path), cv2.IMREAD_COLOR)
    annotated_bgr = cv2.imread(str(pair.annotated_path), cv2.IMREAD_COLOR)
    if raw_bgr is None or annotated_bgr is None:
        return None
    annotated_bgr, resized_to_raw = resize_annotated_to_raw(raw_bgr, annotated_bgr)
    raw_bgr, annotated_bgr, scale = resize_pair_for_batch(raw_bgr, annotated_bgr)
    date_dir = OUTPUT_DIR / pair.date
    date_dir.mkdir(parents=True, exist_ok=True)

    annotation_mask = get_annotation_mask(raw_bgr, annotated_bgr)
    class_masks = {
        name: mask.astype(np.uint8) * 255
        for name, mask in human_annotation_colour_masks(raw_bgr, annotated_bgr, annotation_mask).items()
    }
    _repaired, seed_masks = filled_region_masks(class_masks)
    outlines = combined_segmentation_image(class_masks)
    seeds = combined_segmentation_image(seed_masks)

    features = ndvi_feature_stack(raw_bgr)
    canopy_mask = canopy_mask_from_raw(raw_bgr)
    smoothed_features = smooth_ndvi_features(features, canopy_mask)
    stats, stat_rows = class_statistics_from_seeds(features, seed_masks, canopy_mask)
    labels = classify_canopy_by_ndvi(smoothed_features, canopy_mask, stats)
    preview_labels = smooth_classification(labels, canopy_mask)
    preview_masks = labels_to_class_masks(preview_labels)
    preview_layer = combined_segmentation_image(preview_masks)
    preview_overlay = transparent_overlay(raw_bgr, preview_layer, 0.45)
    raw_vs_preview = side_by_side(raw_bgr, preview_overlay)
    concern, weak_ratio, disagree_ratio = concern_panel(raw_bgr, preview_labels, canopy_mask, stats)
    proof = six_panel_sheet(raw_bgr, outlines, seeds, preview_layer, raw_vs_preview, concern)

    save(date_dir / "raw_ndvi.png", raw_bgr)
    save(date_dir / "human_annotation_outlines.png", outlines)
    save(date_dir / "seed_polygons.png", seeds)
    save(date_dir / "human_anchored_vigour_preview.png", preview_layer)
    save(date_dir / "human_anchored_vigour_preview_overlay.png", preview_overlay)
    save(date_dir / "raw_vs_preview_side_by_side.png", raw_vs_preview)
    save(date_dir / "concern_uncertainty_panel.png", concern)
    save(date_dir / "six_panel_product_proof.png", proof)

    seed_pixels = int(sum(np.count_nonzero(mask) for mask in seed_masks.values()))
    preview_pixels = int(np.count_nonzero(preview_labels))
    review_reasons = []
    if seed_pixels == 0:
        review_reasons.append("no human seed polygons")
    if weak_ratio > 0.35:
        review_reasons.append("high weak-margin uncertainty")
    if disagree_ratio > 0.20:
        review_reasons.append("preview/raw-response disagreement")
    if resized_to_raw:
        review_reasons.append("annotated image resized to raw")
    status = "REVIEW" if review_reasons else "PASS"

    return {
        "date": pair.date,
        "raw_path": str(pair.raw_path),
        "annotated_path": str(pair.annotated_path),
        "processing_scale": scale,
        "seed_pixels": seed_pixels,
        "preview_pixels": preview_pixels,
        "weak_uncertainty_percent": weak_ratio * 100.0,
        "disagreement_percent": disagree_ratio * 100.0,
        "status": status,
        "review_reason": "; ".join(review_reasons),
        "six_panel_path": str(date_dir / "six_panel_product_proof.png"),
        "preview_path": str(date_dir / "human_anchored_vigour_preview.png"),
    }


def match_all_raw_dirs() -> tuple[list[ImagePair], list[dict[str, str]]]:
    raw_by_date = {}
    audit_rows: list[dict[str, str]] = []
    for raw_dir in RAW_DIRS:
        if not raw_dir.exists():
            audit_rows.append(
                {
                    "status": "WARNING",
                    "check": "missing raw folder",
                    "date": "",
                    "detail": f"Raw folder does not exist and was skipped: {raw_dir}",
                }
            )
            continue
        indexed = index_by_date(list_images(raw_dir))
        for date, path in indexed.items():
            existing = raw_by_date.get(date)
            if existing is None:
                raw_by_date[date] = path
            else:
                audit_rows.append(
                    {
                        "status": "INFO",
                        "check": "duplicate raw date",
                        "date": date,
                        "detail": f"Keeping {existing}; skipping duplicate raw file {path}",
                    }
                )

    annotated_by_date = index_by_date(list_images(ANNOTATED_DIR))
    pairs: list[ImagePair] = []
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
        print(f"[MATCH] {date}: raw={raw_path.name} annotated={annotated_path.name}")
        pairs.append(
            ImagePair(
                date=date,
                raw_path=raw_path,
                annotated_path=annotated_path,
                raw_source_label=parse_source_label(raw_path),
                annotated_source_label=parse_source_label(annotated_path),
            )
        )

    for date, raw_path in sorted(raw_by_date.items()):
        if date not in annotated_by_date:
            audit_rows.append(
                {
                    "status": "WARNING",
                    "check": "missing raw/annotated pair",
                    "date": date,
                    "detail": f"A raw image had no matching annotation and was excluded: {raw_path}",
                }
            )
    return pairs, audit_rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs, audit_rows = match_all_raw_dirs()
    rows: list[dict[str, object]] = []
    for pair in sorted(pairs, key=lambda item: item.date):
        row = process_pair(pair)
        if row is not None:
            rows.append(row)
            print(f"[WRITE] {pair.date}: {row['status']} {row['six_panel_path']}")
    skipped_path = OUTPUT_DIR / "skipped_inputs.csv"
    with skipped_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["status", "check", "date", "detail"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_rows)
    summary_path = OUTPUT_DIR / "product_audit_summary.csv"
    fieldnames = [
        "date",
        "raw_path",
        "annotated_path",
        "processing_scale",
        "seed_pixels",
        "preview_pixels",
        "weak_uncertainty_percent",
        "disagreement_percent",
        "status",
        "review_reason",
        "six_panel_path",
        "preview_path",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_timeline(rows)
    write_report(rows)
    print(f"[DONE] Product output: {OUTPUT_DIR}")
    print(f"[DONE] Dates processed: {len(rows)}")


if __name__ == "__main__":
    main()
