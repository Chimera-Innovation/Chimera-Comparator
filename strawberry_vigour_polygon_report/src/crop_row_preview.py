from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .row_detection import RowRegion, detect_rows, write_row_lines_geojson


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview OpenCV strawberry crop-row inference.")
    parser.add_argument("--image", required=True, type=Path, help="Raw NDVI/OSAVI/orthomosaic image.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--date", default="")
    parser.add_argument("--row-count", default=None, type=int)
    return parser.parse_args()


def draw_preview(raw_bgr: np.ndarray, rows: list[RowRegion], confidence: float, output_path: Path) -> None:
    preview = raw_bgr.copy()
    h, w = preview.shape[:2]
    scale = max(h, w) / 1600
    thickness = max(1, int(round(1.5 * scale)))
    line_color = (255, 255, 255) if confidence >= 0.9 else (255, 220, 80)
    for row in rows:
        points = np.array(row.points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(preview, [points], isClosed=False, color=line_color, thickness=thickness, lineType=cv2.LINE_AA)
        if row.row_number == 1 or row.row_number % 10 == 0:
            label = f"{row.row_number}"
            x = max(0, min(w - 80, row.x_min + int(12 * scale)))
            y = max(20, min(h - 20, int(row.center_y)))
            cv2.putText(preview, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, max(0.45, 0.45 * scale), line_color, max(1, thickness), cv2.LINE_AA)

    banner_h = max(70, int(70 * scale))
    overlay = preview.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (8, 15, 22), cv2.FILLED)
    preview = cv2.addWeighted(overlay, 0.76, preview, 0.24, 0)
    status = "verified rows" if confidence >= 0.9 else "preview only - not customer row numbering"
    text = f"OpenCV crop-row inference: {len(rows)} row guides | confidence={confidence:.2f} | {status}"
    cv2.putText(preview, text, (int(24 * scale), int(45 * scale)), cv2.FONT_HERSHEY_SIMPLEX, max(0.58, 0.58 * scale), (255, 255, 255), max(1, int(2 * scale)), cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), preview)
    print(f"[WRITE] Crop-row preview: {output_path}")


def write_report(rows: list[RowRegion], confidence: float, image_path: Path, output_path: Path) -> None:
    source = rows[0].source if rows else "none"
    display = "yes" if confidence >= 0.9 else "no"
    lines = [
        "# Crop Row Inference Preview",
        "",
        f"Image: {image_path}",
        f"Row guides: {len(rows)}",
        f"Confidence: {confidence:.2f}",
        f"Source: {source}",
        f"Customer row labels allowed: {display}",
        "",
        "Decision:",
    ]
    if confidence >= 0.9:
        lines.append("- Row labels may be shown in customer-facing maps.")
    else:
        lines.append("- Keep row labels hidden from customer-facing maps.")
        lines.append("- Use location blocks such as upper-right production block or south-central production block.")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[WRITE] Crop-row report: {output_path}")


def main() -> None:
    args = parse_args()
    raw_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if raw_bgr is None:
        raise SystemExit(f"Could not read image: {args.image}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, confidence = detect_rows(raw_bgr, "top_to_bottom", args.row_count)
    geojson_path = args.output_dir / "crop_row_lines.geojson"
    preview_path = args.output_dir / "crop_row_detection_preview.png"
    report_path = args.output_dir / "crop_row_detection_report.md"
    write_row_lines_geojson(rows, geojson_path, args.date, str(args.image))
    draw_preview(raw_bgr, rows, confidence, preview_path)
    write_report(rows, confidence, args.image, report_path)


if __name__ == "__main__":
    main()
