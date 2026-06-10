from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np

from .bed_detection import field_bounds, make_field_mask


@dataclass(frozen=True)
class RowRegion:
    row_number: int
    row_id: str
    bed_id: str
    y_min: int
    y_max: int
    x_min: int
    x_max: int
    area_pixels: int
    confidence: float = 0.4
    source: str = "approximate_equal_spacing"
    points: tuple[tuple[int, int], ...] = ()

    @property
    def center_y(self) -> float:
        if self.points:
            return float(np.mean([point[1] for point in self.points]))
        return (self.y_min + self.y_max) / 2.0


def detect_rows(
    raw_bgr: np.ndarray,
    numbering: str = "top_to_bottom",
    requested_row_count: int | None = None,
) -> tuple[list[RowRegion], float]:
    """Infer crop-row guides and return a customer-display confidence score."""
    if numbering != "top_to_bottom":
        print(f"[ROWS] Requested {numbering}; using top_to_bottom because Row 1 is the top-most visible strawberry row.")
        numbering = "top_to_bottom"

    field_mask = make_field_mask(raw_bgr)
    y_min, y_max, x_min, x_max = field_bounds(field_mask)
    if y_max <= y_min or x_max <= x_min:
        height, width = raw_bgr.shape[:2]
        y_min, y_max, x_min, x_max = 0, height - 1, 0, width - 1

    texture_rows, texture_confidence = infer_rows_from_texture(
        raw_bgr,
        field_mask,
        y_min,
        y_max,
        x_min,
        x_max,
        requested_row_count,
    )
    if texture_rows and texture_confidence >= 0.55:
        display_note = "eligible for customer map" if texture_confidence >= 0.9 else "hidden from customer map"
        print(f"[ROWS] Inferred {len(texture_rows)} crop-row guides from OpenCV texture; confidence={texture_confidence:.2f}; {display_note}.")
        return texture_rows, texture_confidence

    confidence = 0.4
    source = "approximate_equal_spacing"
    row_count = requested_row_count if requested_row_count and requested_row_count > 0 else estimate_row_count(raw_bgr, y_min, y_max)
    row_count = max(1, row_count)
    edges = np.linspace(y_min, y_max, row_count + 1).round().astype(int)
    rows: list[RowRegion] = []
    for index in range(row_count):
        top = int(edges[index])
        bottom = int(edges[index + 1])
        if bottom <= top:
            bottom = top + 1
        band_mask = field_mask[top : bottom + 1, x_min : x_max + 1]
        area = int(np.count_nonzero(band_mask))
        if area <= 0:
            area = (bottom - top + 1) * (x_max - x_min + 1)
        row_number = index + 1
        row_id = f"row_{row_number:03d}"
        center_y = int(round((top + bottom) / 2))
        line_points = continuous_row_points(x_min, x_max, center_y, raw_bgr.shape[1])
        rows.append(
            RowRegion(
                row_number=row_number,
                row_id=row_id,
                bed_id=row_id,
                y_min=top,
                y_max=bottom,
                x_min=x_min,
                x_max=x_max,
                area_pixels=area,
                confidence=confidence,
                source=source,
                points=line_points,
            )
        )
    print(f"[ROWS] Created {len(rows)} approximate row guides; confidence={confidence:.2f}; hidden from customer map.")
    return rows, confidence


def infer_rows_from_texture(
    raw_bgr: np.ndarray,
    field_mask: np.ndarray,
    y_min: int,
    y_max: int,
    x_min: int,
    x_max: int,
    requested_row_count: int | None,
) -> tuple[list[RowRegion], float]:
    signal = crop_row_signal(raw_bgr, field_mask)
    if signal.size == 0:
        return [], 0.0
    spacing = estimate_texture_spacing(signal, y_min, y_max, requested_row_count)
    peaks = find_row_peaks(signal, y_min, y_max, spacing, requested_row_count)
    if len(peaks) < 12:
        return [], 0.0
    source = "opencv_row_texture"
    if requested_row_count and requested_row_count > 0:
        snapped_peaks, snap_score = snap_peaks_to_requested_count(
            signal,
            peaks,
            y_min,
            y_max,
            spacing,
            requested_row_count,
        )
        if len(snapped_peaks) == requested_row_count:
            peaks = snapped_peaks
            source = "opencv_row_texture_human_confirmed_count"
            spacing = float(np.median(np.diff(peaks))) if len(peaks) > 1 else spacing
    confidence = score_texture_rows(signal, peaks, spacing, y_min, y_max)
    if source == "opencv_row_texture_human_confirmed_count":
        confidence = max(confidence, min(0.94, 0.72 + 0.22 * snap_score))
    rows = row_regions_from_peaks(peaks, spacing, raw_bgr.shape[1], field_mask, x_min, x_max, confidence, source)
    return rows, confidence


def crop_row_signal(raw_bgr: np.ndarray, field_mask: np.ndarray) -> np.ndarray:
    raw = raw_bgr.astype(np.float32)
    blue = raw[:, :, 0]
    green = raw[:, :, 1]
    red = raw[:, :, 2]
    green_index = (2.0 * green - red - blue)
    green_index = cv2.GaussianBlur(green_index, (0, 0), sigmaX=1.2, sigmaY=1.2)
    valid = field_mask.astype(np.float32)
    counts = valid.sum(axis=1)
    weighted = (green_index * valid).sum(axis=1)
    signal = np.divide(weighted, np.maximum(1.0, counts), out=np.zeros_like(weighted), where=counts > 0)
    signal = normalize_signal(signal)
    smooth = cv2.GaussianBlur(signal.reshape(-1, 1).astype(np.float32), (1, 0), sigmaX=0, sigmaY=2.5).reshape(-1)
    trend = cv2.GaussianBlur(smooth.reshape(-1, 1), (1, 0), sigmaX=0, sigmaY=22).reshape(-1)
    detrended = smooth - trend
    return normalize_signal(detrended)


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    finite = np.nan_to_num(signal.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    low, high = np.percentile(finite, [2, 98])
    if high <= low:
        return np.zeros_like(finite, dtype=np.float32)
    return np.clip((finite - low) / (high - low), 0.0, 1.0).astype(np.float32)


def estimate_texture_spacing(signal: np.ndarray, y_min: int, y_max: int, requested_row_count: int | None) -> float:
    visible_height = max(1, y_max - y_min + 1)
    if requested_row_count and requested_row_count > 0:
        return max(8.0, visible_height / requested_row_count)
    scoped = signal[y_min : y_max + 1]
    scoped = scoped - float(np.mean(scoped))
    if scoped.size < 32 or np.allclose(scoped, 0):
        return max(12.0, visible_height / estimate_row_count_from_height(visible_height))
    corr = np.correlate(scoped, scoped, mode="full")[scoped.size - 1 :]
    if corr[0] > 0:
        corr = corr / corr[0]
    min_lag = max(8, int(visible_height / 140))
    max_lag = min(70, max(min_lag + 1, int(visible_height / 35)))
    if max_lag <= min_lag or corr.size <= max_lag:
        return max(12.0, visible_height / estimate_row_count_from_height(visible_height))
    lag = int(np.argmax(corr[min_lag:max_lag]) + min_lag)
    return float(max(8, lag))


def find_row_peaks(
    signal: np.ndarray,
    y_min: int,
    y_max: int,
    spacing: float,
    requested_row_count: int | None,
) -> list[int]:
    min_distance = max(6, int(round(spacing * 0.55)))
    scoped = signal[y_min : y_max + 1]
    threshold = float(np.percentile(scoped, 61))
    candidates: list[tuple[float, int]] = []
    for y in range(max(y_min + 1, 1), min(y_max, signal.size - 2)):
        if signal[y] >= threshold and signal[y] >= signal[y - 1] and signal[y] >= signal[y + 1]:
            candidates.append((float(signal[y]), y))
    candidates.sort(reverse=True)
    chosen: list[int] = []
    max_count = requested_row_count if requested_row_count and requested_row_count > 0 else estimate_row_count_from_height(y_max - y_min + 1) + 12
    for _score, y in candidates:
        if all(abs(y - existing) >= min_distance for existing in chosen):
            chosen.append(y)
        if len(chosen) >= max_count:
            break
    return sorted(chosen)


def snap_peaks_to_requested_count(
    signal: np.ndarray,
    peaks: list[int],
    y_min: int,
    y_max: int,
    spacing: float,
    requested_row_count: int,
) -> tuple[list[int], float]:
    """Use a grower-confirmed row count to snap the texture rhythm to an exact row model."""
    if requested_row_count <= 0:
        return peaks, 0.0
    if len(peaks) < max(12, int(requested_row_count * 0.55)):
        return peaks, 0.0

    visible_height = max(1, y_max - y_min + 1)
    diffs = np.diff(peaks).astype(np.float32) if len(peaks) > 1 else np.array([spacing], dtype=np.float32)
    median_spacing = float(np.median(diffs)) if diffs.size else float(spacing)
    expected_spacing = visible_height / max(1, requested_row_count)
    base_spacings = [median_spacing, spacing, expected_spacing]
    spacing_candidates: list[float] = []
    for base in base_spacings:
        if base <= 0:
            continue
        for factor in np.linspace(0.94, 1.06, 13):
            candidate = float(base * factor)
            if 6.0 <= candidate <= 90.0 and all(abs(candidate - existing) > 0.2 for existing in spacing_candidates):
                spacing_candidates.append(candidate)

    start_candidates: list[tuple[float, float]] = []
    max_edge_gap = min(6, max(1, requested_row_count - len(peaks) + 4))
    for candidate_spacing in spacing_candidates:
        start_candidates.append((candidate_spacing, y_min + 0.5 * candidate_spacing))
        start_candidates.append((candidate_spacing, y_max - (requested_row_count - 0.5) * candidate_spacing))
        for first_index in range(max_edge_gap + 1):
            start_candidates.append((candidate_spacing, float(peaks[0] - first_index * candidate_spacing)))
        for missing_after_last in range(max_edge_gap + 1):
            last_index = requested_row_count - 1 - missing_after_last
            if last_index >= 0:
                start_candidates.append((candidate_spacing, float(peaks[-1] - last_index * candidate_spacing)))

    best_rows: list[int] = []
    best_score = -1.0
    for candidate_spacing, start in start_candidates:
        grid = start + np.arange(requested_row_count, dtype=np.float32) * candidate_spacing
        refined = refine_requested_grid_to_signal(signal, grid, y_min, y_max, candidate_spacing)
        score = score_requested_grid(signal, refined, peaks, y_min, y_max, candidate_spacing)
        if score > best_score:
            best_score = score
            best_rows = refined

    return best_rows, float(max(0.0, min(1.0, best_score)))


def refine_requested_grid_to_signal(
    signal: np.ndarray,
    grid: np.ndarray,
    y_min: int,
    y_max: int,
    spacing: float,
) -> list[int]:
    window = max(3, int(round(spacing * 0.32)))
    refined: list[int] = []
    previous = y_min - 1
    min_gap = max(3, int(round(spacing * 0.45)))
    for center in grid:
        estimated = int(round(center))
        low = max(y_min, estimated - window)
        high = min(y_max, estimated + window)
        if high <= low:
            chosen = int(np.clip(estimated, y_min, y_max))
        else:
            scoped = signal[low : high + 1]
            chosen = int(low + int(np.argmax(scoped)))
        if refined and chosen - previous < min_gap:
            chosen = min(y_max, previous + min_gap)
        refined.append(int(np.clip(chosen, y_min, y_max)))
        previous = refined[-1]
    return refined


def score_requested_grid(
    signal: np.ndarray,
    rows: list[int],
    peaks: list[int],
    y_min: int,
    y_max: int,
    spacing: float,
) -> float:
    if not rows:
        return 0.0
    rows_array = np.array(rows, dtype=np.float32)
    peaks_array = np.array(peaks, dtype=np.float32)
    distances = np.min(np.abs(peaks_array[:, None] - rows_array[None, :]), axis=1) if peaks_array.size else np.array([spacing])
    alignment = 1.0 - float(np.median(distances) / max(1.0, spacing))
    alignment = float(np.clip(alignment, 0.0, 1.0))
    values = np.array([signal[int(np.clip(row, 0, signal.size - 1))] for row in rows], dtype=np.float32)
    scoped = signal[y_min : y_max + 1]
    contrast = float(np.mean(values) - np.median(scoped))
    contrast_score = float(np.clip(contrast / 0.28, 0.0, 1.0))
    diffs = np.diff(rows_array)
    regularity = 1.0
    if diffs.size:
        regularity = 1.0 - float(np.std(diffs) / max(1.0, float(np.median(diffs))))
        regularity = float(np.clip(regularity, 0.0, 1.0))
    inside_score = float(np.mean((rows_array >= y_min) & (rows_array <= y_max)))
    return 0.42 * alignment + 0.28 * contrast_score + 0.2 * regularity + 0.1 * inside_score


def score_texture_rows(signal: np.ndarray, peaks: list[int], spacing: float, y_min: int, y_max: int) -> float:
    if len(peaks) < 2:
        return 0.0
    diffs = np.diff(peaks).astype(np.float32)
    median_spacing = float(np.median(diffs))
    spacing_cv = float(np.std(diffs) / max(1.0, median_spacing))
    peak_values = np.array([signal[y] for y in peaks], dtype=np.float32)
    scoped = signal[y_min : y_max + 1]
    contrast = float(np.mean(peak_values) - np.median(scoped))
    count_expected = max(1.0, (y_max - y_min + 1) / max(1.0, spacing))
    count_score = min(1.0, len(peaks) / count_expected)
    regularity_score = max(0.0, 1.0 - spacing_cv)
    contrast_score = min(1.0, max(0.0, contrast / 0.28))
    confidence = 0.2 + 0.32 * regularity_score + 0.28 * contrast_score + 0.2 * count_score
    return float(np.clip(confidence, 0.0, 0.94))


def row_regions_from_peaks(
    peaks: list[int],
    spacing: float,
    image_width: int,
    field_mask: np.ndarray,
    x_min: int,
    x_max: int,
    confidence: float,
    source: str = "opencv_row_texture",
) -> list[RowRegion]:
    rows: list[RowRegion] = []
    half_band = max(3, int(round(spacing * 0.35)))
    for index, center_y in enumerate(peaks, start=1):
        top = max(0, center_y - half_band)
        bottom = min(field_mask.shape[0] - 1, center_y + half_band)
        xs = np.where(field_mask[center_y, :])[0] if 0 <= center_y < field_mask.shape[0] else np.array([])
        if xs.size:
            row_x_min = int(xs.min())
            row_x_max = int(xs.max())
        else:
            row_x_min, row_x_max = x_min, x_max
        band_mask = field_mask[top : bottom + 1, row_x_min : row_x_max + 1]
        area = int(np.count_nonzero(band_mask))
        if area <= 0:
            area = (bottom - top + 1) * (row_x_max - row_x_min + 1)
        row_id = f"row_{index:03d}"
        rows.append(
            RowRegion(
                row_number=index,
                row_id=row_id,
                bed_id=row_id,
                y_min=top,
                y_max=bottom,
                x_min=row_x_min,
                x_max=row_x_max,
                area_pixels=area,
                confidence=confidence,
                source=source,
                points=continuous_row_points(row_x_min, row_x_max, center_y, image_width),
            )
        )
    return rows


def estimate_row_count(raw_bgr: np.ndarray, y_min: int, y_max: int) -> int:
    height = raw_bgr.shape[0]
    visible_height = max(1, y_max - y_min + 1)
    if height > 5000:
        return 80
    estimated = estimate_row_count_from_height(visible_height)
    return max(60, min(90, estimated))


def estimate_row_count_from_height(visible_height: int) -> int:
    return round(max(1, visible_height) / 30)


def continuous_row_points(x_min: int, x_max: int, center_y: int, image_width: int) -> tuple[tuple[int, int], ...]:
    """Return a single continuous guide line spanning the visible planted width."""
    sample_count = 32
    xs = np.linspace(x_min, x_max, sample_count)
    # Keep the fallback guide nearly straight. Real curvature should come from a
    # future true row detector before confidence can exceed the display gate.
    points = [(int(round(np.clip(x, 0, image_width - 1))), int(center_y)) for x in xs]
    return tuple(points)


def row_span_for_bounds(y_min: int, y_max: int, rows: list[RowRegion]) -> tuple[int, int]:
    if not rows:
        return 1, 1
    top = min(y_min, y_max)
    bottom = max(y_min, y_max)
    overlapping = [row.row_number for row in rows if row.y_max >= top and row.y_min <= bottom]
    if overlapping:
        return min(overlapping), max(overlapping)
    nearest_top = min(rows, key=lambda row: abs(row.center_y - top)).row_number
    nearest_bottom = min(rows, key=lambda row: abs(row.center_y - bottom)).row_number
    return min(nearest_top, nearest_bottom), max(nearest_top, nearest_bottom)


def row_region_for_centroid(centroid_y: float, rows: list[RowRegion]) -> RowRegion | None:
    if not rows:
        return None
    for row in rows:
        if row.y_min <= centroid_y <= row.y_max:
            return row
    return min(rows, key=lambda row: abs(row.center_y - centroid_y))


def rows_for_span(row_start: int, row_end: int, rows: list[RowRegion]) -> list[RowRegion]:
    return [row for row in rows if row_start <= row.row_number <= row_end]


def write_row_lines_geojson(rows: list[RowRegion], output_path: Path, date: str = "", source_image: str = "") -> None:
    features = []
    for row in rows:
        coords = [[int(x), int(y)] for x, y in (row.points or ((row.x_min, int(row.center_y)), (row.x_max, int(row.center_y))))]
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "row_id": row.row_id,
                    "row_number": row.row_number,
                    "confidence": row.confidence,
                    "source": row.source,
                    "date": date,
                    "source_image": source_image,
                    "customer_display": row.confidence >= 0.9,
                },
                "geometry": {"type": "LineString", "coordinates": coords},
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2), encoding="utf-8")
    print(f"[WRITE] Row line GeoJSON: {output_path}")
