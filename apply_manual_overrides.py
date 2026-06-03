from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"D:\NDVI-dataset\MVP - ndvi progression")
OUT_DIR = ROOT / "black_paint_marking_progression"
TEMPLATE = OUT_DIR / "manual_override_template.csv"
SUMMARY = OUT_DIR / "black_paint_marking_summary.csv"
VALIDATION_REPORT = OUT_DIR / "manual_override_validation_report.md"


def font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


SMALL = font(16)
TITLE = font(28)


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [{key: clean(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def polygon_points(row: dict[str, str]) -> list[tuple[float, float]] | None:
    points: list[tuple[float, float]] = []
    for idx in range(1, 5):
        x = clean(row.get(f"x{idx}", ""))
        y = clean(row.get(f"y{idx}", ""))
        if not x or not y:
            return None
        try:
            points.append((float(x), float(y)))
        except ValueError:
            return None
    return points


def create_mask(row: dict[str, str], points: list[tuple[float, float]]) -> tuple[Path, int, float]:
    raw_path = Path(row["raw_ndvi_path"])
    with Image.open(raw_path) as raw:
        size = raw.size
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(points, fill=255)
    field = row["field_name"]
    date = row["date"]
    mask_path = OUT_DIR / field / "masks" / f"{field}_{date}_manual_filled_region_mask.png"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(mask_path)
    area_pixels = sum(1 for value in mask.getdata() if value)
    area_percent = area_pixels / (size[0] * size[1]) * 100
    return mask_path, area_pixels, area_percent


def create_manual_overlay(row: dict[str, str], mask_path: Path) -> Path:
    raw = Image.open(row["raw_ndvi_path"]).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    fill = Image.new("RGB", raw.size, (255, 220, 0))
    alpha = mask.point(lambda value: 100 if value else 0)
    raw.paste(fill, (0, 0), alpha)
    outline = ImageDraw.Draw(raw)
    points = polygon_points(row) or []
    if points:
        outline.line(points + [points[0]], fill=(0, 0, 0), width=8)
    raw.thumbnail((650, 650), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(raw)
    draw.rectangle((0, 0, raw.size[0], 44), fill=(0, 0, 0))
    draw.text((10, 11), f"{row['field_name']} {row['date']} manual override", fill=(255, 255, 255), font=SMALL)
    overlay_path = OUT_DIR / row["field_name"] / "overlays" / f"{row['field_name']}_{row['date']}_manual_override_overlay.png"
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    raw.save(overlay_path)
    return overlay_path


def regenerate_mcintyre_timeline() -> bool:
    field_dir = OUT_DIR / "McIntyre-Road"
    first_manual = field_dir / "overlays" / "McIntyre-Road_2026-05-08_manual_override_overlay.png"
    second_auto = field_dir / "overlays" / "McIntyre-Road_2026-05-27_black_paint_overlay.png"
    if not first_manual.exists() or not second_auto.exists():
        return False
    panels = []
    for label, path in [("2026-05-08 manual override", first_manual), ("2026-05-27 automatic black Paint extraction", second_auto)]:
        image = Image.open(path).convert("RGB")
        image.thumbnail((520, 520), Image.Resampling.LANCZOS)
        panel = Image.new("RGB", (image.size[0], image.size[1] + 70), "white")
        draw = ImageDraw.Draw(panel)
        draw.text((8, 8), label, fill=(20, 20, 20), font=SMALL)
        panel.paste(image, (0, 70))
        panels.append(panel)
    width = sum(panel.size[0] for panel in panels)
    height = max(panel.size[1] for panel in panels) + 88
    timeline = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(timeline)
    draw.text((18, 18), "McIntyre-Road Black Paint Progression Timeline", fill=(20, 20, 20), font=TITLE)
    draw.text((18, 52), "2026-05-08 uses manual override; raw NDVI remains background only.", fill=(120, 30, 20), font=SMALL)
    x = 0
    for panel in panels:
        timeline.paste(panel, (x, 88))
        x += panel.size[0]
    timeline.save(field_dir / "McIntyre-Road_black_paint_progression_timeline.png")
    return True


def update_summary(area_pixels: int, area_percent: float) -> bool:
    rows = read_csv(SUMMARY)
    updated = False
    for row in rows:
        if row.get("field_name") == "McIntyre-Road" and row.get("date") == "2026-05-08":
            row["extraction_status"] = "manual_override_applied"
            row["filled_region_area_pixels"] = str(area_pixels)
            row["filled_region_area_percent"] = f"{area_percent:.6f}"
            row["num_marked_regions"] = "1"
            row["notes"] = "manually traced from black Paint annotation"
            updated = True
    if rows:
        write_csv(SUMMARY, rows, list(rows[0].keys()))
    return updated


def write_validation(template_exists: bool, included: bool, coords_filled: bool, mask_created: bool, timeline_regenerated: bool, summary_updated: bool) -> None:
    lines = [
        "# Manual Override Validation Report",
        "",
        f"- Manual override template exists: **{'yes' if template_exists else 'no'}**",
        f"- McIntyre-Road 2026-05-08 is included: **{'yes' if included else 'no'}**",
        f"- Polygon coordinates are filled: **{'yes' if coords_filled else 'no'}**",
        f"- Manual override mask was created: **{'yes' if mask_created else 'no'}**",
        f"- Timeline was regenerated with manual override: **{'yes' if timeline_regenerated else 'no'}**",
        f"- Summary CSV was updated with manual override: **{'yes' if summary_updated else 'no'}**",
        "",
    ]
    if coords_filled and mask_created and timeline_regenerated and summary_updated:
        lines.append("Decision: manual override is **complete**.")
    else:
        lines.append("Decision: manual override is **not complete** yet. Fill polygon coordinates in `manual_override_template.csv`, then run `apply_manual_overrides.py`.")
    VALIDATION_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    template_exists = TEMPLATE.exists()
    included = False
    coords_filled = False
    mask_created = False
    timeline_regenerated = False
    summary_updated = False

    if template_exists:
        rows = read_csv(TEMPLATE)
        for row in rows:
            if row.get("field_name") == "McIntyre-Road" and row.get("date") == "2026-05-08":
                included = True
                points = polygon_points(row)
                if points:
                    coords_filled = True
                    mask_path, area_pixels, area_percent = create_mask(row, points)
                    mask_created = mask_path.exists()
                    create_manual_overlay(row, mask_path)
                    timeline_regenerated = regenerate_mcintyre_timeline()
                    summary_updated = update_summary(area_pixels, area_percent)
                break

    write_validation(template_exists, included, coords_filled, mask_created, timeline_regenerated, summary_updated)
    print(f"manual_override_template_exists={template_exists}")
    print(f"mcintyre_2026_05_08_included={included}")
    print(f"polygon_coordinates_filled={coords_filled}")
    print(f"manual_override_mask_created={mask_created}")
    print(f"timeline_regenerated={timeline_regenerated}")
    print(f"summary_updated={summary_updated}")


if __name__ == "__main__":
    main()
