from __future__ import annotations

import csv
import json
import math
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont


ROOT = Path(r"D:\NDVI-dataset\MVP - ndvi progression")
OUT_DIR = ROOT / "brush_mark_comparator"

sys.path.insert(0, str(ROOT))
import build_black_paint_marking_progression as pairlib  # noqa: E402


MAX_PROCESS_DIM = 2200
MAX_PANEL_DIM = 520
MIN_COMPONENT_AREA = 18
MAX_COMPONENT_FILL_RATIO = 0.75
MIN_CHANGED_PERCENT = 0.00002


def font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


SMALL = font(16)
TITLE = font(28)


def open_rgb(path: Path, max_dim: int | None = None) -> Image.Image:
    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(path).convert("RGB")
    if max_dim:
        image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    return image.copy()


def resized_pair(pair: pairlib.ImagePair) -> tuple[Image.Image, Image.Image, float, float]:
    raw = open_rgb(pair.raw_path)
    annotated = open_rgb(pair.annotated_path)
    if annotated.size != raw.size:
        annotated = annotated.resize(raw.size, Image.Resampling.BILINEAR)
    original_w, original_h = raw.size
    scale = min(1.0, MAX_PROCESS_DIM / max(original_w, original_h))
    target = (max(1, round(original_w * scale)), max(1, round(original_h * scale)))
    return (
        raw.resize(target, Image.Resampling.BILINEAR),
        annotated.resize(target, Image.Resampling.BILINEAR),
        original_w / target[0],
        original_h / target[1],
    )


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


def neighbor_count(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
    total = np.zeros(mask.shape, dtype=np.uint8)
    for dy in range(3):
        for dx in range(3):
            total += padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return total


def components(mask: np.ndarray) -> list[dict[str, Any]]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    active = set(zip(xs.tolist(), ys.tolist()))
    found: list[dict[str, Any]] = []
    while active:
        start = active.pop()
        queue: deque[tuple[int, int]] = deque([start])
        visited[start[1], start[0]] = True
        points: list[tuple[int, int]] = []
        min_x = max_x = start[0]
        min_y = max_y = start[1]
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
                active.discard((nx, ny))
                queue.append((nx, ny))
        found.append({"points": points, "area": len(points), "bbox": [min_x, min_y, max_x, max_y]})
    return found


def filter_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    height, width = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    kept = 0
    for comp in components(mask):
        x1, y1, x2, y2 = comp["bbox"]
        comp_w = x2 - x1 + 1
        comp_h = y2 - y1 + 1
        bbox_area = comp_w * comp_h
        fill_ratio = comp["area"] / max(1, bbox_area)
        touches_border = x1 <= 2 or y1 <= 2 or x2 >= width - 3 or y2 >= height - 3
        giant_border = touches_border and (comp_w > width * 0.55 or comp_h > height * 0.55)
        header_bar = y1 < 45 and comp_w > width * 0.35
        tiny = comp["area"] < MIN_COMPONENT_AREA
        blob_not_stroke = fill_ratio > MAX_COMPONENT_FILL_RATIO and comp["area"] > 600
        if tiny or giant_border or header_bar or blob_not_stroke:
            continue
        kept += 1
        for x, y in comp["points"]:
            out[y, x] = True
    return out, kept


def make_difference_heatmap(diff: np.ndarray) -> Image.Image:
    if diff.max() <= 0:
        scaled = diff.astype(np.uint8)
    else:
        scaled = np.clip(diff / max(1, np.percentile(diff, 99)) * 255, 0, 255).astype(np.uint8)
    rgb = np.zeros((diff.shape[0], diff.shape[1], 3), dtype=np.uint8)
    rgb[:, :, 0] = scaled
    rgb[:, :, 1] = np.clip(scaled // 2, 0, 255)
    return Image.fromarray(rgb, mode="RGB")


def extract_brush_mask(pair: pairlib.ImagePair) -> tuple[np.ndarray, Image.Image, dict[str, Any]]:
    raw_small, ann_small, sx, sy = resized_pair(pair)
    raw = np.asarray(raw_small).astype(np.int16)
    ann = np.asarray(ann_small).astype(np.int16)
    channel_diff = np.abs(ann - raw)
    diff = channel_diff.max(axis=2)
    gray_raw = raw.mean(axis=2)
    gray_ann = ann.mean(axis=2)
    gray_delta = np.abs(gray_ann - gray_raw)

    nonzero = diff[diff > 0]
    if nonzero.size:
        adaptive = max(10, min(45, float(np.percentile(nonzero, 94))))
    else:
        adaptive = 255

    strong_diff = diff >= adaptive
    dark_support = (gray_ann < 115) & (gray_raw - gray_ann > 6) & (diff > 7)
    edge_support = (gray_delta > max(6, adaptive * 0.42)) & (diff > 6)
    raw_candidate = strong_diff | dark_support | edge_support
    raw_candidate &= neighbor_count(raw_candidate) >= 2
    raw_candidate = erode(dilate(raw_candidate, 1), 1)
    raw_candidate = dilate(raw_candidate, 1)
    filtered, num_components = filter_components(raw_candidate)

    changed_pixel_percent = float((diff >= adaptive).sum()) / diff.size * 100
    brush_pixel_percent = float(filtered.sum()) / filtered.size * 100
    settings = {
        "adaptive_threshold": round(adaptive, 3),
        "changed_pixel_percent": round(changed_pixel_percent, 6),
        "brush_pixel_percent_processing_space": round(brush_pixel_percent, 6),
        "num_components": num_components,
        "processing_size": raw_small.size,
        "scale_x": sx,
        "scale_y": sy,
        "raw_ndvi_thresholding_used": False,
    }
    return filtered, make_difference_heatmap(diff), settings


def save_mask(mask: np.ndarray, pair: pairlib.ImagePair, path: Path) -> int:
    image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    image = image.resize(pair.raw_size, Image.Resampling.NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return int(np.asarray(image).astype(bool).sum())


def transfer_overlay(pair: pairlib.ImagePair, mask_path: Path, output: Path, status: str) -> None:
    raw = Image.open(pair.raw_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    cyan = Image.new("RGB", raw.size, (0, 255, 255))
    alpha = mask.point(lambda value: 255 if value else 0)
    raw.paste(cyan, (0, 0), alpha)
    draw = ImageDraw.Draw(raw)
    draw.rectangle((0, 0, raw.size[0], 78), fill=(0, 0, 0))
    draw.text((18, 12), f"{pair.field_name} {pair.date}", fill=(255, 255, 255), font=TITLE)
    draw.text((18, 48), f"Brush mark comparator transfer: {status}", fill=(255, 220, 120), font=SMALL)
    raw.thumbnail((1400, 1400), Image.Resampling.LANCZOS)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw.save(output)


def panel(path_or_image: Path | Image.Image, label: str) -> Image.Image:
    if isinstance(path_or_image, Image.Image):
        image = path_or_image.convert("RGB")
    else:
        image = Image.open(path_or_image).convert("RGB")
    image.thumbnail((MAX_PANEL_DIM, MAX_PANEL_DIM), Image.Resampling.LANCZOS)
    out = Image.new("RGB", (image.size[0], image.size[1] + 54), "white")
    draw = ImageDraw.Draw(out)
    draw.text((8, 10), label, fill=(20, 20, 20), font=SMALL)
    out.paste(image, (0, 54))
    return out


def debug_panel(pair: pairlib.ImagePair, heatmap: Image.Image, mask_path: Path, overlay_path: Path, output: Path) -> None:
    mask_img = Image.open(mask_path).convert("RGB")
    panels = [
        panel(pair.raw_path, "Raw NDVI"),
        panel(pair.annotated_path, "Human annotated reference"),
        panel(ImageEnhance.Contrast(heatmap).enhance(1.4), "Difference heatmap"),
        panel(mask_img, "Extracted brush mask"),
        panel(overlay_path, "Raw NDVI with transferred brush marks"),
    ]
    width = sum(p.size[0] for p in panels)
    height = max(p.size[1] for p in panels)
    canvas = Image.new("RGB", (width, height), "white")
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.size[0]
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def status_from(settings: dict[str, Any], brush_pixels_full: int, total_pixels: int) -> tuple[str, str]:
    brush_percent_full = brush_pixels_full / total_pixels * 100
    if settings["changed_pixel_percent"] < MIN_CHANGED_PERCENT:
        return "manual_review_required", "Raw and annotated images are nearly identical; no reliable brush difference detected."
    if brush_pixels_full == 0 or brush_percent_full < 0.0005:
        return "manual_review_required", "Visible brush marks may exist, but extracted brush mask is near zero; manual review required."
    if settings["num_components"] <= 1 or brush_percent_full < 0.01:
        return "partial", "Thin brush marks transferred, but extraction is weak; inspect debug panel."
    return "success", "Brush marks transferred from raw-vs-annotated difference. No NDVI thresholding used."


def write_audit(rows: list[dict[str, str]]) -> None:
    columns = [
        "field_name",
        "date",
        "raw_path",
        "annotated_path",
        "changed_pixel_percent",
        "brush_pixel_percent",
        "num_components",
        "extraction_status",
        "notes",
    ]
    with (OUT_DIR / "brush_mark_extraction_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{column: row[column] for column in columns} for row in rows])


def write_report(rows: list[dict[str, str]]) -> None:
    succeeded = [row for row in rows if row["extraction_status"] == "success"]
    partial = [row for row in rows if row["extraction_status"] == "partial"]
    manual = [row for row in rows if row["extraction_status"] == "manual_review_required"]
    failed = [row for row in rows if row["extraction_status"] == "failed"]
    lines = [
        "# Brush Mark Comparator Final Report",
        "",
        "- Human brush marks are truth. Raw NDVI is background only.",
        "- No NDVI thresholding or automatic stress detection was used.",
        "",
        f"- Successful dates: {len(succeeded)}",
        f"- Partial dates: {len(partial)}",
        f"- Manual review required: {len(manual)}",
        f"- Failed dates: {len(failed)}",
        "",
        "## Success",
        "",
    ]
    lines.extend([f"- {row['field_name']} {row['date']}" for row in succeeded] or ["- none"])
    lines.extend(["", "## Partial", ""])
    lines.extend([f"- {row['field_name']} {row['date']}: {row['notes']}" for row in partial] or ["- none"])
    lines.extend(["", "## Manual Review Required", ""])
    lines.extend([f"- {row['field_name']} {row['date']}: {row['notes']}" for row in manual] or ["- none"])
    lines.extend(
        [
            "",
            "Final rule: Human brush marks are truth. Raw NDVI is background only.",
        ]
    )
    (OUT_DIR / "brush_mark_comparator_final_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = pairlib.match_pairs()
    rows: list[dict[str, str]] = []
    for pair in pairs:
        mask, heatmap, settings = extract_brush_mask(pair)
        mask_path = OUT_DIR / pair.field_name / "masks" / f"{pair.field_name}_{pair.date}_brush_mask.png"
        overlay_path = OUT_DIR / pair.field_name / "overlays" / f"{pair.field_name}_{pair.date}_raw_with_brush_marks.png"
        debug_path = OUT_DIR / pair.field_name / "debug" / f"{pair.field_name}_{pair.date}_brush_debug.png"
        brush_pixels = save_mask(mask, pair, mask_path)
        status, note = status_from(settings, brush_pixels, pair.raw_size[0] * pair.raw_size[1])
        transfer_overlay(pair, mask_path, overlay_path, status)
        debug_panel(pair, heatmap, mask_path, overlay_path, debug_path)
        row = {
            "field_name": pair.field_name,
            "date": pair.date,
            "raw_path": str(pair.raw_path),
            "annotated_path": str(pair.annotated_path),
            "changed_pixel_percent": f"{settings['changed_pixel_percent']:.6f}",
            "brush_pixel_percent": "" if status == "manual_review_required" else f"{brush_pixels / (pair.raw_size[0] * pair.raw_size[1]) * 100:.6f}",
            "num_components": str(settings["num_components"]),
            "extraction_status": status,
            "notes": note + " " + json.dumps(settings),
            "overlay_path": str(overlay_path),
            "debug_path": str(debug_path),
        }
        rows.append(row)

    write_audit(rows)
    write_report(rows)
    print(f"- output folder: {OUT_DIR}")
    print(f"- image pairs processed: {len(rows)}")
    print(f"- success: {sum(row['extraction_status'] == 'success' for row in rows)}")
    print(f"- partial: {sum(row['extraction_status'] == 'partial' for row in rows)}")
    print(f"- manual review required: {sum(row['extraction_status'] == 'manual_review_required' for row in rows)}")
    print(f"- audit CSV: {OUT_DIR / 'brush_mark_extraction_audit.csv'}")
    print(f"- final report: {OUT_DIR / 'brush_mark_comparator_final_report.md'}")
    print("- final rule: Human brush marks are truth. Raw NDVI is background only.")


if __name__ == "__main__":
    main()
