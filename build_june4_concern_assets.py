from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from strawberry_vigour_polygon_report.src.annotation_parser import get_annotation_mask, resize_annotated_to_raw
from strawberry_vigour_polygon_report.src.growth_stage import growth_stage_for_date


ROOT = Path(__file__).resolve().parent
PRODUCT_DIR = ROOT / "human_anchored_vigour_zone_preview_product"
DATE = "2026-06-04"
RAW_PATH = Path(r"C:\Users\Chimera\Downloads\Strawberry 1\Mallard-Avenue-6-4-2026-orthophoto-NDVI.png")
POLYGON_PATH = Path(r"C:\Users\Chimera\Downloads\strawberry1-annotation-polygon\Mallard-Avenue-6-4-2026-orthophoto-NDVI-polygon.png")


def load(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def save(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def extract_polygon_masks(raw: np.ndarray, polygon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diff_mask = get_annotation_mask(raw, polygon)
    gray = cv2.cvtColor(polygon, cv2.COLOR_BGR2GRAY)
    stroke = ((gray < 80) & (diff_mask > 0)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(stroke, cv2.MORPH_CLOSE, kernel)
    contours, _hierarchy = cv2.findContours((closed > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(stroke)
    min_area = max(150.0, raw.shape[0] * raw.shape[1] * 0.00001)
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(filled, [contour], -1, 255, cv2.FILLED)
    return stroke, filled


def transparent_overlay(raw: np.ndarray, layer: np.ndarray, alpha: float) -> np.ndarray:
    active = np.any(layer > 0, axis=2)
    mixed = cv2.addWeighted(raw, 1.0 - alpha, layer, alpha, 0)
    out = raw.copy()
    out[active] = mixed[active]
    return out


def panel(image: np.ndarray, title: str, size: tuple[int, int] = (520, 360)) -> np.ndarray:
    w, h = size
    title_h = 40
    canvas = np.full((h + title_h, w, 3), 255, dtype=np.uint8)
    scale = min(w / image.shape[1], h / image.shape[0])
    resized = cv2.resize(image, (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    x = (w - resized.shape[1]) // 2
    y = title_h + (h - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    cv2.putText(canvas, title, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)
    return canvas


def main() -> None:
    raw = load(RAW_PATH)
    polygon = load(POLYGON_PATH)
    polygon, _resized = resize_annotated_to_raw(raw, polygon)

    stroke, filled = extract_polygon_masks(raw, polygon)
    seed = np.zeros_like(raw)
    seed[filled > 0] = (32, 32, 32)
    seed[stroke > 0] = (0, 0, 0)

    human_ref = raw.copy()
    human_ref[stroke > 0] = (0, 0, 0)
    preview = np.zeros_like(raw)
    uncertainty = np.full_like(raw, 55)
    uncertainty[filled > 0] = (0, 170, 255)
    overlay = transparent_overlay(raw, seed, 0.45)

    date_dir = PRODUCT_DIR / DATE
    save(date_dir / "raw_ndvi.png", raw)
    save(date_dir / "human_annotation_outlines.png", human_ref)
    save(date_dir / "seed_polygons.png", seed)
    save(date_dir / "human_anchored_vigour_preview.png", preview)
    save(date_dir / "human_anchored_vigour_preview_overlay.png", raw)
    save(date_dir / "raw_vs_preview_side_by_side.png", np.hstack([raw, overlay]))
    save(date_dir / "concern_uncertainty_panel.png", uncertainty)

    proof = np.vstack(
        [
            np.hstack([panel(raw, "1. Raw NDVI/background"), panel(human_ref, "2. Concern polygons"), panel(seed, "3. Concern areas")]),
            np.hstack([panel(preview, "4. Vigour preview unavailable"), panel(overlay, "5. Raw + concern overlay"), panel(uncertainty, "6. Concern review")]),
        ]
    )
    save(date_dir / "six_panel_product_proof.png", proof)

    print(f"[WRITE] {date_dir}")
    print(f"[INFO] {DATE} {growth_stage_for_date(DATE)} uses concern polygons only; no vigour annotation source was available.")


if __name__ == "__main__":
    main()
