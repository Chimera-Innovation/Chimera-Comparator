from __future__ import annotations

import csv
import math
import textwrap
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


Image.MAX_IMAGE_PIXELS = None

ROOT = Path(r"D:\NDVI-dataset\MVP - ndvi progression")
COMPARATOR_DIR = ROOT / "brush_mark_comparator"
AUDIT_CSV = COMPARATOR_DIR / "brush_mark_extraction_audit.csv"
OUT_DIR = ROOT / "farmer_progression_output"
ANALYSIS_MAX_DIM = 1700
MAX_VIS_DIM = 1200


STATUS_LABELS = {
    "2026-05-08": "Initial Concern",
    "2026-05-12": "Initial Concern",
    "2026-05-18": "Emerging Pattern",
    "2026-05-22": "Expanding Concern",
    "2026-05-27": "Harvest-Time Concern",
    "2026-05-29": "Harvest-Time Concern",
}


@dataclass
class Observation:
    field_name: str
    date: str
    raw_path: Path
    annotated_path: Path
    extraction_status: str
    brush_mask_path: Path
    overlay_path: Path
    raw_size: tuple[int, int]
    analysis_size: tuple[int, int]
    filled_mask: np.ndarray | None = None
    filled_mask_path: Path | None = None
    coverage_percent: float | None = None
    components: list[dict] | None = None
    new_zone_count: int | None = None
    persistent_zone_count: int | None = None
    expanded_zone_count: int | None = None
    change_from_previous: float | None = None
    trend: str = "manual_review_required"
    notes: str = ""


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ("arial.ttf", "DejaVuSans.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


FONT_SMALL = font(18)
FONT_BODY = font(23)
FONT_BODY_BOLD = font(23, bold=True)
FONT_H2 = font(31, bold=True)
FONT_H1 = font(46, bold=True)


def parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def display_date(value: str) -> str:
    parsed = parse_date(value)
    return parsed.strftime("May %-d") if hasattr(parsed, "strftime") else value


def safe_display_date(value: str) -> str:
    parsed = parse_date(value)
    return f"May {parsed.day}"


def read_observations() -> list[Observation]:
    rows: list[Observation] = []
    with AUDIT_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            field = row["field_name"]
            date = row["date"]
            raw_path = Path(row["raw_path"])
            raw = Image.open(raw_path)
            raw_size = raw.size
            scale = min(1.0, ANALYSIS_MAX_DIM / max(raw_size))
            analysis_size = (max(1, round(raw_size[0] * scale)), max(1, round(raw_size[1] * scale)))
            rows.append(
                Observation(
                    field_name=field,
                    date=date,
                    raw_path=raw_path,
                    annotated_path=Path(row["annotated_path"]),
                    extraction_status=row["extraction_status"],
                    brush_mask_path=COMPARATOR_DIR / field / "masks" / f"{field}_{date}_brush_mask.png",
                    overlay_path=COMPARATOR_DIR / field / "overlays" / f"{field}_{date}_raw_with_brush_marks.png",
                    raw_size=raw_size,
                    analysis_size=analysis_size,
                )
            )
            raw.close()
    return sorted(rows, key=lambda obs: (obs.field_name, parse_date(obs.date)))


def mask_size(mask: np.ndarray) -> tuple[int, int]:
    return (mask.shape[1], mask.shape[0])


def dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = mask.astype(bool)
    for _ in range(iterations):
        padded = np.pad(out, 1, mode="constant")
        next_mask = np.zeros_like(out, dtype=bool)
        for dy in range(3):
            for dx in range(3):
                next_mask |= padded[dy : dy + out.shape[0], dx : dx + out.shape[1]]
        out = next_mask
    return out


def erode(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = mask.astype(bool)
    for _ in range(iterations):
        padded = np.pad(out, 1, mode="constant")
        next_mask = np.ones_like(out, dtype=bool)
        for dy in range(3):
            for dx in range(3):
                next_mask &= padded[dy : dy + out.shape[0], dx : dx + out.shape[1]]
        out = next_mask
    return out


def close_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    return erode(dilate(mask, iterations), iterations)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    background = ~mask
    visited = np.zeros_like(mask, dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for x in range(width):
        if background[0, x]:
            queue.append((x, 0))
        if background[height - 1, x]:
            queue.append((x, height - 1))
    for y in range(height):
        if background[y, 0]:
            queue.append((0, y))
        if background[y, width - 1]:
            queue.append((width - 1, y))
    while queue:
        x, y = queue.popleft()
        if x < 0 or y < 0 or x >= width or y >= height:
            continue
        if visited[y, x] or not background[y, x]:
            continue
        visited[y, x] = True
        queue.extend(((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)))
    holes = background & ~visited
    return mask | holes


def components(mask: np.ndarray, min_area: int = 64) -> list[dict]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    found: list[dict] = []
    ys, xs = np.nonzero(mask)
    for start_x, start_y in zip(xs.tolist(), ys.tolist()):
        if visited[start_y, start_x]:
            continue
        queue: deque[tuple[int, int]] = deque([(start_x, start_y)])
        visited[start_y, start_x] = True
        points: list[tuple[int, int]] = []
        min_x = max_x = start_x
        min_y = max_y = start_y
        while queue:
            x, y = queue.popleft()
            points.append((x, y))
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1), (x - 1, y - 1), (x + 1, y + 1), (x - 1, y + 1), (x + 1, y - 1)):
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                if visited[ny, nx] or not mask[ny, nx]:
                    continue
                visited[ny, nx] = True
                queue.append((nx, ny))
        if len(points) >= min_area:
            found.append({"points": points, "area": len(points), "bbox": (min_x, min_y, max_x, max_y)})
    found.sort(key=lambda item: item["area"], reverse=True)
    return found


def recover_filled_region(mask_path: Path, target_size: tuple[int, int]) -> np.ndarray:
    brush = Image.open(mask_path).convert("L")
    if brush.size != target_size:
        brush = brush.resize(target_size, Image.Resampling.NEAREST)
    mask = np.asarray(brush) > 0
    if not mask.any():
        return mask

    scale = max(target_size) / 2200
    close_iterations = max(3, min(14, round(7 * scale)))
    closed = close_mask(mask, close_iterations)
    filled = fill_holes(closed)
    expanded = dilate(filled, max(1, round(2 * scale)))

    # Keep recovered regions anchored to the human brush strokes. This avoids turning
    # a loose open mark into an invented whole-field polygon.
    anchored = np.zeros_like(expanded, dtype=bool)
    anchor_components = components(expanded, min_area=max(96, int(expanded.size * 0.00003)))
    for comp in anchor_components:
        comp_mask = np.zeros_like(expanded, dtype=bool)
        for x, y in comp["points"]:
            comp_mask[y, x] = True
        brush_overlap = int((comp_mask & dilate(mask, close_iterations)).sum())
        if brush_overlap > 0:
            for x, y in comp["points"]:
                anchored[y, x] = True
    return anchored


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    image = image.resize(size, Image.Resampling.NEAREST)
    return np.asarray(image) > 0


def fit_image(image: Image.Image, box: tuple[int, int], fill: str = "white") -> Image.Image:
    out = Image.new("RGB", box, fill)
    work = image.convert("RGB").copy()
    work.thumbnail(box, Image.Resampling.LANCZOS)
    x = (box[0] - work.size[0]) // 2
    y = (box[1] - work.size[1]) // 2
    out.paste(work, (x, y))
    return out


def overlay_region(raw_path: Path, mask: np.ndarray, title: str, subtitle: str, max_dim: int = 1200) -> Image.Image:
    raw = Image.open(raw_path).convert("RGB")
    raw.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    small_mask = resize_mask(mask, raw.size)
    overlay = Image.new("RGBA", raw.size, (255, 0, 120, 0))
    alpha = Image.fromarray((small_mask.astype(np.uint8) * 112), mode="L")
    overlay.putalpha(alpha)
    out = raw.convert("RGBA")
    out.alpha_composite(overlay)
    out = out.convert("RGB")
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, out.size[0], 86), fill=(0, 0, 0))
    draw.text((18, 10), title, fill="white", font=FONT_H2)
    draw.text((18, 50), subtitle, fill=(255, 230, 140), font=FONT_SMALL)
    return out


def location_from_mask(mask: np.ndarray | None) -> str:
    if mask is None or not mask.any():
        return "manual review required"
    comps = components(mask, min_area=max(64, int(mask.size * 0.00002)))
    if not comps:
        return "manual review required"
    x1, y1, x2, y2 = comps[0]["bbox"]
    cx = (x1 + x2) / 2 / mask.shape[1]
    cy = (y1 + y2) / 2 / mask.shape[0]
    vertical = "upper" if cy < 0.33 else "lower" if cy > 0.66 else "central"
    horizontal = "left" if cx < 0.33 else "right" if cx > 0.66 else "center"
    if horizontal == "center":
        return f"{vertical}-field area"
    return f"{vertical}-{horizontal} field area"


def summarize_components(mask: np.ndarray | None, limit: int = 3) -> list[str]:
    if mask is None or not mask.any():
        return ["manual review required"]
    comps = components(mask, min_area=max(64, int(mask.size * 0.00002)))[:limit]
    labels: list[str] = []
    for comp in comps:
        comp_mask = np.zeros_like(mask, dtype=bool)
        for x, y in comp["points"]:
            comp_mask[y, x] = True
        labels.append(location_from_mask(comp_mask))
    return labels or ["manual review required"]


def trend_from_change(change: float | None) -> str:
    if change is None:
        return "stable"
    if abs(change) < 0.05:
        return "stable"
    return "increased" if change > 0 else "decreased"


def analyze_field(observations: list[Observation]) -> None:
    previous: Observation | None = None
    for obs in observations:
        field_dir = OUT_DIR / obs.field_name
        obs.filled_mask_path = field_dir / "filled_masks" / f"{obs.field_name}_{obs.date}_filled_human_marked_region_mask.png"
        if obs.extraction_status not in {"success", "partial"} or not obs.brush_mask_path.exists():
            obs.trend = "manual_review_required"
            obs.notes = "Comparator extraction was not available; no 0% value reported."
            continue
        obs.filled_mask = recover_filled_region(obs.brush_mask_path, obs.analysis_size)
        save_mask(obs.filled_mask, obs.filled_mask_path)
        total_pixels = obs.filled_mask.size
        filled_pixels = int(obs.filled_mask.sum())
        if filled_pixels == 0:
            obs.trend = "manual_review_required"
            obs.notes = "Human brush marks did not recover into a filled region; manual review required."
            continue
        obs.coverage_percent = filled_pixels / total_pixels * 100
        obs.components = components(obs.filled_mask, min_area=max(64, int(total_pixels * 0.00002)))
        if previous and previous.filled_mask is not None and previous.coverage_percent is not None:
            prev_mask = resize_mask(previous.filled_mask, obs.analysis_size)
            obs.change_from_previous = obs.coverage_percent - previous.coverage_percent
            new_count = 0
            persistent_count = 0
            expanded_count = 0
            for comp in obs.components or []:
                comp_mask = np.zeros_like(obs.filled_mask, dtype=bool)
                for x, y in comp["points"]:
                    comp_mask[y, x] = True
                overlap = int((comp_mask & prev_mask).sum())
                outside = int((comp_mask & ~prev_mask).sum())
                overlap_ratio = overlap / max(1, comp["area"])
                if overlap_ratio < 0.05:
                    new_count += 1
                else:
                    persistent_count += 1
                    if outside / max(1, comp["area"]) > 0.15:
                        expanded_count += 1
            obs.new_zone_count = new_count
            obs.persistent_zone_count = persistent_count
            obs.expanded_zone_count = expanded_count
        else:
            obs.change_from_previous = None
            obs.new_zone_count = len(obs.components or [])
            obs.persistent_zone_count = 0
            obs.expanded_zone_count = 0
        obs.trend = trend_from_change(obs.change_from_previous)
        obs.notes = "Human-marked filled region recovered from brush comparator output. Raw NDVI used as background only."
        previous = obs


def write_summary_csv(all_observations: Iterable[Observation]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    columns = [
        "field_name",
        "date",
        "vigour_loss_percent",
        "change_from_previous_date",
        "new_zone_count",
        "persistent_zone_count",
        "expanded_zone_count",
        "overall_trend",
        "notes",
    ]
    with (OUT_DIR / "field_progression_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for obs in all_observations:
            writer.writerow(
                {
                    "field_name": obs.field_name,
                    "date": obs.date,
                    "vigour_loss_percent": "" if obs.coverage_percent is None else f"{obs.coverage_percent:.2f}",
                    "change_from_previous_date": "" if obs.change_from_previous is None else f"{obs.change_from_previous:+.2f}",
                    "new_zone_count": "" if obs.new_zone_count is None else obs.new_zone_count,
                    "persistent_zone_count": "" if obs.persistent_zone_count is None else obs.persistent_zone_count,
                    "expanded_zone_count": "" if obs.expanded_zone_count is None else obs.expanded_zone_count,
                    "overall_trend": obs.trend,
                    "notes": obs.notes,
                }
            )


def draw_wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, width: int, font_obj: ImageFont.ImageFont, fill: str | tuple[int, int, int] = "black", line_gap: int = 8) -> int:
    x, y = xy
    chars = max(14, width // 12)
    for paragraph in text.split("\n"):
        for line in textwrap.wrap(paragraph, width=chars) or [""]:
            draw.text((x, y), line, fill=fill, font=font_obj)
            y += font_obj.size + line_gap if hasattr(font_obj, "size") else 26
    return y


def make_growth_map(field: str, observations: list[Observation]) -> Path:
    usable = [obs for obs in observations if obs.filled_mask is not None]
    output = OUT_DIR / field / "field_concern_growth_map.png"
    if not usable:
        return output
    latest = usable[-1]
    base = Image.open(latest.raw_path).convert("RGB")
    base.thumbnail((MAX_VIS_DIM, MAX_VIS_DIM), Image.Resampling.LANCZOS)
    masks = [resize_mask(obs.filled_mask, base.size) for obs in usable if obs.filled_mask is not None]
    first = masks[0]
    stack = np.stack(masks, axis=0)
    count = stack.sum(axis=0)
    union = count > 0
    persistent = count >= 2
    later_new = union & ~first
    first_only = first & ~persistent

    out = base.convert("RGBA")
    color = np.zeros((base.size[1], base.size[0], 4), dtype=np.uint8)
    color[first_only] = (40, 135, 255, 125)
    color[persistent] = (255, 220, 0, 145)
    color[later_new] = (255, 40, 40, 155)
    out.alpha_composite(Image.fromarray(color, mode="RGBA"))
    out = out.convert("RGB")
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, out.size[0], 118), fill=(0, 0, 0))
    draw.text((18, 12), f"{field} Concern Growth Map", fill="white", font=FONT_H2)
    draw.text((18, 54), "Human-marked regions only. Raw NDVI is background.", fill=(255, 230, 140), font=FONT_SMALL)
    legend = [("Blue", (40, 135, 255), "First observed concern"), ("Yellow", (255, 220, 0), "Persistent concern"), ("Red", (255, 40, 40), "Expanded/new concern")]
    lx = 18
    ly = 88
    for _, rgb, label in legend:
        draw.rectangle((lx, ly, lx + 22, ly + 22), fill=rgb)
        draw.text((lx + 30, ly - 1), label, fill="white", font=FONT_SMALL)
        lx += 260
    output.parent.mkdir(parents=True, exist_ok=True)
    out.save(output)
    return output


def make_trend_chart(field: str, observations: list[Observation]) -> Path:
    output = OUT_DIR / field / "vigour_loss_percent_over_time.png"
    usable = [obs for obs in observations if obs.coverage_percent is not None]
    width, height = 1300, 760
    out = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(out)
    draw.text((48, 32), f"{field} Human-Marked Vigour-Loss Coverage %", fill=(20, 20, 20), font=FONT_H2)
    draw.text((48, 72), "Coverage is based only on recovered human-marked filled regions.", fill=(70, 70, 70), font=FONT_SMALL)
    chart = (110, 150, 1220, 620)
    draw.rectangle(chart, outline=(40, 40, 40), width=2)
    if not usable:
        draw.text((160, 300), "Manual review required", fill=(160, 0, 0), font=FONT_H2)
        output.parent.mkdir(parents=True, exist_ok=True)
        out.save(output)
        return output
    values = [obs.coverage_percent or 0 for obs in usable]
    max_y = max(5.0, math.ceil(max(values) * 1.25))
    for step in range(6):
        y_val = max_y * step / 5
        y = chart[3] - (chart[3] - chart[1]) * step / 5
        draw.line((chart[0], y, chart[2], y), fill=(225, 225, 225), width=1)
        draw.text((34, y - 12), f"{y_val:.1f}%", fill=(80, 80, 80), font=FONT_SMALL)
    points: list[tuple[int, int]] = []
    for index, obs in enumerate(usable):
        x = chart[0] + (chart[2] - chart[0]) * (index + 0.5) / max(1, len(usable))
        y = chart[3] - (obs.coverage_percent or 0) / max_y * (chart[3] - chart[1])
        points.append((round(x), round(y)))
    if len(points) > 1:
        draw.line(points, fill=(196, 0, 95), width=5)
    for point, obs in zip(points, usable):
        draw.ellipse((point[0] - 10, point[1] - 10, point[0] + 10, point[1] + 10), fill=(196, 0, 95))
        draw.text((point[0] - 34, point[1] - 42), f"{obs.coverage_percent:.2f}%", fill=(30, 30, 30), font=FONT_SMALL)
        draw.text((point[0] - 34, chart[3] + 24), safe_display_date(obs.date), fill=(30, 30, 30), font=FONT_SMALL)
    draw.text((580, 690), "Date", fill=(45, 45, 45), font=FONT_BODY_BOLD)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.save(output)
    return output


def make_timeline(field: str, observations: list[Observation]) -> Path:
    output = OUT_DIR / field / "field_farmer_timeline.png"
    tile_w, tile_h = 460, 520
    header_h = 120
    cols = min(4, max(1, len(observations)))
    rows = math.ceil(len(observations) / cols)
    out = Image.new("RGB", (cols * tile_w, header_h + rows * tile_h), "white")
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, out.size[0], header_h), fill=(22, 32, 42))
    draw.text((28, 18), f"{field} Concern Timeline", fill="white", font=FONT_H1)
    draw.text((30, 72), "Raw NDVI background with recovered human-marked filled regions.", fill=(240, 225, 165), font=FONT_BODY)
    for index, obs in enumerate(observations):
        x = index % cols * tile_w
        y = header_h + index // cols * tile_h
        draw.rectangle((x, y, x + tile_w - 1, y + tile_h - 1), outline=(215, 215, 215))
        if obs.filled_mask is not None and obs.coverage_percent is not None:
            panel_img = overlay_region(
                obs.raw_path,
                obs.filled_mask,
                f"{safe_display_date(obs.date)}",
                STATUS_LABELS.get(obs.date, "Human-Observed Concern"),
                max_dim=520,
            )
        else:
            panel_img = Image.new("RGB", (tile_w, 360), (245, 245, 245))
            ImageDraw.Draw(panel_img).text((24, 150), "Manual review required", fill=(160, 0, 0), font=FONT_BODY_BOLD)
        panel_img = fit_image(panel_img, (tile_w - 28, 350))
        out.paste(panel_img, (x + 14, y + 18))
        coverage = "manual review" if obs.coverage_percent is None else f"{obs.coverage_percent:.2f}%"
        draw.text((x + 22, y + 386), f"Coverage: {coverage}", fill=(25, 25, 25), font=FONT_BODY_BOLD)
        draw.text((x + 22, y + 424), f"Status: {STATUS_LABELS.get(obs.date, obs.trend.title())}", fill=(80, 80, 80), font=FONT_BODY)
        trend = obs.trend.replace("_", " ").title()
        draw.text((x + 22, y + 462), f"Trend: {trend}", fill=(80, 80, 80), font=FONT_BODY)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.save(output)
    return output


def make_dashboard(field: str, observations: list[Observation], growth_map: Path, chart: Path, timeline: Path) -> Path:
    output = OUT_DIR / field / "field_farmer_dashboard.png"
    width, height = 1800, 2300
    out = Image.new("RGB", (width, height), (248, 249, 247))
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, width, 170), fill=(22, 32, 42))
    draw.text((54, 34), f"{field} Seasonal Vigour-Loss Progression", fill="white", font=FONT_H1)
    draw.text((56, 96), "Human-observed field concern evolution from May 8 to May 29", fill=(235, 225, 170), font=FONT_BODY)

    def place_image(path: Path, box: tuple[int, int, int, int], label: str) -> None:
        draw.text((box[0], box[1] - 42), label, fill=(25, 25, 25), font=FONT_H2)
        image = fit_image(Image.open(path).convert("RGB"), (box[2] - box[0], box[3] - box[1]))
        out.paste(image, (box[0], box[1]))
        draw.rectangle(box, outline=(210, 210, 210), width=2)

    place_image(timeline, (56, 240, 1744, 820), "1. Concern Timeline")
    place_image(chart, (56, 910, 880, 1390), "2. Vigour-Loss Coverage Trend")
    place_image(growth_map, (936, 910, 1744, 1390), "3. Growth Map")

    latest = next((obs for obs in reversed(observations) if obs.filled_mask is not None), None)
    first = next((obs for obs in observations if obs.filled_mask is not None), None)
    persistent_mask = None
    expansion_mask = None
    usable_masks = [obs.filled_mask for obs in observations if obs.filled_mask is not None]
    if latest and usable_masks:
        resized = [resize_mask(mask, mask_size(latest.filled_mask)) for mask in usable_masks]
        stack = np.stack(resized, axis=0)
        persistent_mask = stack.sum(axis=0) >= 2
        if first and first.filled_mask is not None:
            expansion_mask = latest.filled_mask & ~resize_mask(first.filled_mask, mask_size(latest.filled_mask))
    blocks = [
        ("4. Persistent Areas", summarize_components(persistent_mask), (56, 1490)),
        ("5. Expansion Areas", summarize_components(expansion_mask), (636, 1490)),
        ("6. Recommended Areas To Scout Next Season", summarize_components(latest.filled_mask if latest else None), (1136, 1490)),
    ]
    for title, items, (x, y) in blocks:
        draw.rectangle((x, y, x + 470, y + 430), fill="white", outline=(215, 215, 215), width=2)
        yy = draw_wrapped(draw, (x + 24, y + 24), title, 410, FONT_BODY_BOLD, fill=(25, 25, 25), line_gap=6)
        yy += 20
        for item in items:
            yy = draw_wrapped(draw, (x + 28, yy), f"- {item}", 410, FONT_BODY, fill=(55, 55, 55), line_gap=9)
            yy += 12
    draw.rectangle((0, height - 110, width, height), fill=(22, 32, 42))
    draw.text((56, height - 74), "For scouting prioritization only. Not a disease diagnosis.", fill="white", font=FONT_BODY_BOLD)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.save(output)
    return output


def write_field_summary(field: str, observations: list[Observation]) -> Path:
    output = OUT_DIR / field / "field_farmer_summary.md"
    usable = [obs for obs in observations if obs.coverage_percent is not None]
    first = usable[0] if usable else None
    latest = usable[-1] if usable else None
    latest_mask = latest.filled_mask if latest else None
    first_mask = first.filled_mask if first else None
    persistent_mask = None
    expansion_mask = None
    if latest and first and latest_mask is not None and first_mask is not None:
        masks = [resize_mask(obs.filled_mask, mask_size(latest_mask)) for obs in usable if obs.filled_mask is not None]
        persistent_mask = np.stack(masks, axis=0).sum(axis=0) >= 2 if masks else None
        expansion_mask = latest_mask & ~resize_mask(first_mask, mask_size(latest_mask))
    if first and latest:
        overall = f"{first.coverage_percent:.2f}% -> {latest.coverage_percent:.2f}%"
        trend = "Increasing" if latest.coverage_percent > first.coverage_percent else "Decreasing" if latest.coverage_percent < first.coverage_percent else "Stable"
    else:
        overall = "manual review required"
        trend = "manual review required"
    coverage_lines = []
    for obs in observations:
        value = "manual review required" if obs.coverage_percent is None else f"{obs.coverage_percent:.2f}%"
        coverage_lines.append(f"{safe_display_date(obs.date)}: {value}")
    lines = [
        "# FIELD SUMMARY",
        "",
        f"Field: {field}",
        "",
        f"First concern observed: {first.date if first else 'manual review required'}",
        "",
        f"Latest concern observed: {latest.date if latest else 'manual review required'}",
        "",
        "Human-marked vigour-loss coverage:",
        "",
        *coverage_lines,
        "",
        f"Overall change: {overall}",
        "",
        f"Primary concern area: {location_from_mask(latest_mask)}",
        "",
        f"Most persistent area: {location_from_mask(persistent_mask)}",
        "",
        f"Largest expansion area: {location_from_mask(expansion_mask)}",
        "",
        f"Expansion trend: {trend}",
        "",
        "## Operational Interpretation",
        "",
        "Human-marked concern zones expanded or persisted through the season in the highlighted areas. These locations should be prioritized for future investigation and scouting.",
        "",
        "Potential factors to review:",
        "",
        "- irrigation uniformity",
        "- drainage",
        "- fertility",
        "- disease pressure",
        "- crop history",
        "- field operations",
        "",
        "This report does not diagnose disease. It identifies human-observed concern, persistent low-vigour areas, and vigour-loss zones that require investigation.",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def write_season_story(field: str, observations: list[Observation]) -> Path:
    output = OUT_DIR / field / "field_season_story.md"
    usable = [obs for obs in observations if obs.coverage_percent is not None]
    latest = usable[-1] if usable else None
    first = usable[0] if usable else None
    persistent_mask = None
    expansion_mask = None
    if latest and first and latest.filled_mask is not None and first.filled_mask is not None:
        masks = [resize_mask(obs.filled_mask, mask_size(latest.filled_mask)) for obs in usable if obs.filled_mask is not None]
        persistent_mask = np.stack(masks, axis=0).sum(axis=0) >= 2 if masks else None
        expansion_mask = latest.filled_mask & ~resize_mask(first.filled_mask, mask_size(latest.filled_mask))
    lines = [
        f"# {field} Season Story",
        "",
        "This story uses human-observed Paint markings only. Raw NDVI is included as context and background, not as a detector.",
        "",
    ]
    for index, obs in enumerate(observations):
        date_label = safe_display_date(obs.date)
        if obs.coverage_percent is None:
            sentence = "Manual review is required before reporting a coverage value."
        elif index == 0:
            sentence = f"Early concern first observed at {obs.coverage_percent:.2f}% human-marked vigour-loss coverage."
        elif obs.trend == "increased":
            sentence = f"Additional concern zones emerged or expanded, reaching {obs.coverage_percent:.2f}% coverage."
        elif obs.trend == "decreased":
            sentence = f"Marked coverage decreased to {obs.coverage_percent:.2f}%, while remaining human-observed areas still require review."
        else:
            sentence = f"Marked coverage remained stable at {obs.coverage_percent:.2f}%."
        lines.extend([f"## {date_label}", "", sentence, ""])
    lines.extend(
        [
            "## Summary",
            "",
            f"Concern started in the {location_from_mask(first.filled_mask if first else None)}.",
            "",
            f"Concern persisted most clearly in the {location_from_mask(persistent_mask)}.",
            "",
            f"Concern expanded most clearly in the {location_from_mask(expansion_mask)}.",
            "",
            "Next season, scouting should prioritize the repeated human-observed concern zones before diagnosing cause. Review irrigation uniformity, drainage, fertility, disease pressure, crop history, and field operations.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def write_final_report(by_field: dict[str, list[Observation]]) -> Path:
    output = OUT_DIR / "farmer_ready_report.md"
    lines = [
        "# Farmer-Ready Human-Marked Vigour-Loss Progression Report",
        "",
        "Human Paint annotations are ground truth. Raw NDVI is background/context only. No NDVI thresholding, disease diagnosis, or invented stress zones were used.",
        "",
    ]
    for field, observations in by_field.items():
        usable = [obs for obs in observations if obs.coverage_percent is not None]
        first = usable[0] if usable else None
        latest = usable[-1] if usable else None
        if latest and first and latest.filled_mask is not None and first.filled_mask is not None:
            masks = [resize_mask(obs.filled_mask, mask_size(latest.filled_mask)) for obs in usable if obs.filled_mask is not None]
            persistent_mask = np.stack(masks, axis=0).sum(axis=0) >= 2 if masks else None
            expansion_mask = latest.filled_mask & ~resize_mask(first.filled_mask, mask_size(latest.filled_mask))
            if latest.coverage_percent > first.coverage_percent:
                trend_word = "increasing"
                change_sentence = f"Human-marked vigour-loss coverage increased from {first.coverage_percent:.2f}% to {latest.coverage_percent:.2f}% across the observation period."
            elif latest.coverage_percent < first.coverage_percent:
                trend_word = "decreasing"
                change_sentence = f"Human-marked vigour-loss coverage changed from {first.coverage_percent:.2f}% to {latest.coverage_percent:.2f}% across the observation period."
            else:
                trend_word = "stable"
                change_sentence = f"Human-marked vigour-loss coverage remained stable at {latest.coverage_percent:.2f}% across the observation period."
        else:
            persistent_mask = None
            expansion_mask = None
            trend_word = "manual_review_required"
            change_sentence = "Manual review is required before reporting human-marked vigour-loss coverage."
        coverage = ", ".join(
            f"{safe_display_date(obs.date)}: {'manual review required' if obs.coverage_percent is None else f'{obs.coverage_percent:.2f}%'}"
            for obs in observations
        )
        lines.extend(
            [
                f"## {field}",
                "",
                f"1. Where did concern first appear? {location_from_mask(first.filled_mask if first else None)}.",
                "",
                f"2. Which zones persisted the longest? {location_from_mask(persistent_mask)}.",
                "",
                f"3. Which zones expanded the most? {location_from_mask(expansion_mask)}.",
                "",
                f"4. What was the vigour-loss coverage progression? {coverage}.",
                "",
                f"5. Which areas should be investigated next season? {location_from_mask(latest.filled_mask if latest else None)} and other repeated highlighted areas.",
                "",
                f"6. Is concern increasing, stable, or decreasing? {trend_word}.",
                "",
                f"7. What is the highest-priority scouting area? {location_from_mask(latest.filled_mask if latest else None)}.",
                "",
                change_sentence,
                "",
                "The highlighted areas represent repeated human-observed concern zones and should be prioritized for future investigation.",
                "",
            ]
        )
    lines.extend(
        [
            "## Farmer Action",
            "",
            "Use these outputs to decide what changed, where it changed, how much changed, and where to scout next season. This is for scouting prioritization only and is not a disease diagnosis.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    observations = read_observations()
    by_field: dict[str, list[Observation]] = {}
    for obs in observations:
        by_field.setdefault(obs.field_name, []).append(obs)
    for field_observations in by_field.values():
        analyze_field(field_observations)
    write_summary_csv(observations)
    for field, field_observations in by_field.items():
        growth = make_growth_map(field, field_observations)
        chart = make_trend_chart(field, field_observations)
        timeline = make_timeline(field, field_observations)
        make_dashboard(field, field_observations, growth, chart, timeline)
        write_field_summary(field, field_observations)
        write_season_story(field, field_observations)
    final_report = write_final_report(by_field)
    print(f"Farmer progression output: {OUT_DIR}")
    print(f"Summary CSV: {OUT_DIR / 'field_progression_summary.csv'}")
    print(f"Final report: {final_report}")
    print("Rule honored: human-marked filled regions only; no NDVI thresholding.")


if __name__ == "__main__":
    main()
