from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r"D:\NDVI-dataset\MVP - ndvi progression")
OUT_DIR = ROOT / "raw_with_human_reference_annotations"
BLACK_SCRIPT_DIR = ROOT

sys.path.insert(0, str(BLACK_SCRIPT_DIR))
import build_black_paint_marking_progression as black  # noqa: E402


MAX_PANEL_DIM = 620


def font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


SMALL = font(16)
TITLE = font(28)


def save_mask_image(mask, pair: black.ImagePair, path: Path) -> int:
    image = Image.fromarray((mask.astype("uint8") * 255), mode="L")
    image = image.resize(pair.raw_size, Image.Resampling.NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return int(sum(1 for value in image.getdata() if value))


def make_transferred_image(pair: black.ImagePair, stroke_mask_path: Path, filled_mask_path: Path, output: Path, include_fill: bool) -> None:
    raw = Image.open(pair.raw_path).convert("RGB")
    stroke = Image.open(stroke_mask_path).convert("L")
    if include_fill:
        filled = Image.open(filled_mask_path).convert("L")
        fill_color = Image.new("RGB", raw.size, (255, 0, 255))
        fill_alpha = filled.point(lambda value: 70 if value else 0)
        raw.paste(fill_color, (0, 0), fill_alpha)
    stroke_color = Image.new("RGB", raw.size, (0, 255, 255))
    stroke_alpha = stroke.point(lambda value: 255 if value else 0)
    raw.paste(stroke_color, (0, 0), stroke_alpha)
    draw = ImageDraw.Draw(raw)
    draw.rectangle((0, 0, raw.size[0], 72), fill=(0, 0, 0))
    draw.text((18, 12), f"{pair.field_name} {pair.date}", fill=(255, 255, 255), font=TITLE)
    footer = "Human reference annotation transferred from Paint-marked image"
    draw.rectangle((0, raw.size[1] - 42, raw.size[0], raw.size[1]), fill=(0, 0, 0))
    draw.text((18, raw.size[1] - 32), footer, fill=(255, 255, 255), font=SMALL)
    raw.thumbnail((1400, 1400), Image.Resampling.LANCZOS)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw.save(output)


def make_manual_review_placeholder(pair: black.ImagePair, output: Path) -> None:
    raw = Image.open(pair.raw_path).convert("RGB")
    draw = ImageDraw.Draw(raw)
    draw.rectangle((0, 0, raw.size[0], 92), fill=(0, 0, 0))
    draw.text((18, 12), f"{pair.field_name} {pair.date}", fill=(255, 255, 255), font=TITLE)
    draw.text((18, 50), "Manual review required: human reference marking was not confidently transferred", fill=(255, 220, 120), font=SMALL)
    raw.thumbnail((1400, 1400), Image.Resampling.LANCZOS)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw.save(output)


def panel_image(path: Path, label: str, max_dim: int = MAX_PANEL_DIM) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (image.size[0], image.size[1] + 54), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((8, 10), label, fill=(20, 20, 20), font=SMALL)
    panel.paste(image, (0, 54))
    return panel


def make_verification(pair: black.ImagePair, stroke_mask_path: Path, output_path: Path, transferred_path: Path) -> None:
    panels = [
        panel_image(pair.raw_path, "Raw NDVI"),
        panel_image(pair.annotated_path, "Human annotated reference"),
        panel_image(stroke_mask_path, "Extracted human marking mask"),
        panel_image(transferred_path, "Raw NDVI with transferred annotation"),
    ]
    width = sum(panel.size[0] for panel in panels)
    height = max(panel.size[1] for panel in panels)
    canvas = Image.new("RGB", (width, height), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.size[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def make_timeline(field: str, rows: list[dict[str, str]], output: Path) -> None:
    panels = []
    for row in rows:
        image = Image.open(row["output_annotated_raw_path"]).convert("RGB")
        image.thumbnail((520, 520), Image.Resampling.LANCZOS)
        panel = Image.new("RGB", (image.size[0], image.size[1] + 62), "white")
        draw = ImageDraw.Draw(panel)
        draw.text((8, 8), f"{row['date']} | {row['transfer_status']}", fill=(20, 20, 20), font=SMALL)
        panel.paste(image, (0, 62))
        panels.append(panel)
    width = sum(panel.size[0] for panel in panels)
    height = max(panel.size[1] for panel in panels) + 96
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 18), f"{field} Raw NDVI with Human Reference Annotations", fill=(20, 20, 20), font=TITLE)
    draw.text((18, 54), "Paint-marked human annotations transferred onto raw NDVI background", fill=(120, 30, 20), font=SMALL)
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 96))
        x += panel.size[0]
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def write_pairing_audit(pairs: list[black.ImagePair]) -> None:
    lines = [
        "# Pairing Audit",
        "",
        "| field name | date | raw image path | human annotated reference image path | raw dimensions | annotated dimensions | dimensions match | resize/registration needed |",
        "|---|---|---|---|---:|---:|---|---|",
    ]
    for pair in pairs:
        lines.append(
            f"| {pair.field_name} | {pair.date} | `{pair.raw_path}` | `{pair.annotated_path}` | {pair.raw_size[0]}x{pair.raw_size[1]} | {pair.annotated_size[0]}x{pair.annotated_size[1]} | {pair.raw_size == pair.annotated_size} | {'resize annotated to raw for comparison' if pair.resized_annotated_to_raw else 'none'} |"
        )
    (OUT_DIR / "pairing_audit.md").write_text("\n".join(lines), encoding="utf-8")


def write_final_report(rows: list[dict[str, str]]) -> None:
    success = [row for row in rows if row["transfer_status"].startswith("success")]
    manual = [row for row in rows if row["manual_review_required"] == "yes"]
    lines = [
        "# Annotation Transfer Final Report",
        "",
        f"1. Raw images annotated using human reference markings: **{len(success)}**",
        "2. Dates transferred successfully:",
        "",
    ]
    lines.extend(f"- {row['field_name']} {row['date']} ({row['transfer_status']})" for row in success)
    lines.extend(["", "3. Dates requiring manual review:", ""])
    lines.extend([f"- {row['field_name']} {row['date']}: {row['notes']}" for row in manual] or ["- none"])
    lines.extend(
        [
            "",
            "4. Output follows only human annotations: **yes**",
            "5. NDVI-based automatic detection used: **no**",
            "",
            "Raw NDVI images were annotated using only the human Paint-marked reference images. No NDVI thresholding or automatic stress detection was used.",
        ]
    )
    (OUT_DIR / "annotation_transfer_final_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = black.match_pairs()
    write_pairing_audit(pairs)
    summary_rows: list[dict[str, str]] = []
    manual_rows: list[dict[str, str]] = []
    rows_by_field: dict[str, list[dict[str, str]]] = {}

    for pair in pairs:
        stroke_mask, filled_mask, settings = black.extract_black_marks(pair)
        field_dir = OUT_DIR / pair.field_name
        stroke_path = field_dir / "masks" / f"{pair.field_name}_{pair.date}_human_reference_marking_mask.png"
        filled_path = field_dir / "masks" / f"{pair.field_name}_{pair.date}_human_reference_filled_region_mask.png"
        output_path = field_dir / "annotated_raw" / f"{pair.field_name}_{pair.date}_raw_with_human_reference_annotation.png"
        verification_path = field_dir / "verification" / f"{pair.field_name}_{pair.date}_verification.png"
        stroke_pixels = save_mask_image(stroke_mask, pair, stroke_path)
        filled_pixels = save_mask_image(filled_mask, pair, filled_path)
        weak_extraction = settings.get("stroke_pixels_processing_space", 0) < 200
        marking_extracted = stroke_pixels > 0 and not weak_extraction
        filled_recovered = filled_pixels > 0
        manual_review = not marking_extracted
        if manual_review:
            status = "manual_review_required"
            notes = "Human markings were not confidently extracted; do not claim transfer is complete."
        elif filled_recovered:
            status = "success_outline_and_fill"
            notes = "Black Paint stroke transferred; filled region overlay included. Raw NDVI thresholding not used."
        else:
            status = "success_outline_only"
            notes = "Black Paint stroke transferred; no filled region recovered. Raw NDVI thresholding not used."

        # Even manual-review rows get a verification panel, but not a fake transferred annotation.
        if manual_review:
            make_manual_review_placeholder(pair, output_path)
        else:
            make_transferred_image(pair, stroke_path, filled_path, output_path, include_fill=filled_recovered)
        make_verification(pair, stroke_path, verification_path, output_path)

        row = {
            "field_name": pair.field_name,
            "date": pair.date,
            "raw_ndvi_path": str(pair.raw_path),
            "human_reference_path": str(pair.annotated_path),
            "output_annotated_raw_path": str(output_path),
            "verification_panel_path": str(verification_path),
            "transfer_status": status,
            "marking_extracted": "yes" if marking_extracted else "no",
            "filled_region_recovered": "yes" if filled_recovered else "no",
            "manual_review_required": "yes" if manual_review else "no",
            "notes": notes + f" settings={settings}",
        }
        summary_rows.append(row)
        rows_by_field.setdefault(pair.field_name, []).append(row)
        if manual_review:
            manual_rows.append(
                {
                    "field_name": pair.field_name,
                    "date": pair.date,
                    "raw_ndvi_path": str(pair.raw_path),
                    "human_reference_path": str(pair.annotated_path),
                    "reason": "human markings not confidently extracted",
                    "manual_action_needed": "manually trace Paint-marked reference annotation and transfer to raw background",
                    "notes": notes,
                }
            )

    summary_columns = [
        "field_name",
        "date",
        "raw_ndvi_path",
        "human_reference_path",
        "output_annotated_raw_path",
        "verification_panel_path",
        "transfer_status",
        "marking_extracted",
        "filled_region_recovered",
        "manual_review_required",
        "notes",
    ]
    with (OUT_DIR / "annotation_transfer_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_columns)
        writer.writeheader()
        writer.writerows(summary_rows)

    manual_columns = ["field_name", "date", "raw_ndvi_path", "human_reference_path", "reason", "manual_action_needed", "notes"]
    with (OUT_DIR / "manual_transfer_review.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manual_columns)
        writer.writeheader()
        writer.writerows(manual_rows)

    timeline_count = 0
    for field, rows in rows_by_field.items():
        rows.sort(key=lambda row: row["date"])
        make_timeline(field, rows, OUT_DIR / field / f"{field}_raw_annotation_transfer_timeline.png")
        timeline_count += 1

    write_final_report(summary_rows)
    print(f"- output folder: {OUT_DIR}")
    print(f"- number of image pairs processed: {len(summary_rows)}")
    print(f"- successful transfers: {sum(row['transfer_status'].startswith('success') for row in summary_rows)}")
    print(f"- manual review required: {len(manual_rows)}")
    print(f"- timeline images created: {timeline_count}")
    print(f"- CSV path: {OUT_DIR / 'annotation_transfer_summary.csv'}")
    print(f"- final report path: {OUT_DIR / 'annotation_transfer_final_report.md'}")
    print("- final rule: Human reference markings are the source of truth. Raw NDVI is the background only.")


if __name__ == "__main__":
    main()
