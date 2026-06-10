from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class BedRegion:
    bed_id: str
    y_min: int
    y_max: int
    x_min: int
    x_max: int
    area_pixels: int


def detect_beds(raw_bgr: np.ndarray, numbering: str = "bottom_to_top", requested_bed_count: int | None = None) -> list[BedRegion]:
    field_mask = make_field_mask(raw_bgr)
    y_min, y_max, x_min, x_max = field_bounds(field_mask)
    if y_max <= y_min or x_max <= x_min:
        height, width = raw_bgr.shape[:2]
        return [BedRegion("bed_01", 0, height - 1, 0, width - 1, height * width)]

    separators = find_bed_separators(field_mask, y_min, y_max)
    ranges = ranges_from_separators(y_min, y_max, separators)
    ranges = [item for item in ranges if item[1] - item[0] >= max(20, raw_bgr.shape[0] // 140)]

    if requested_bed_count and requested_bed_count > 0:
        ranges = uniform_ranges(y_min, y_max, requested_bed_count)
    elif len(ranges) < 2:
        ranges = uniform_ranges(y_min, y_max, estimate_bed_count(raw_bgr.shape[0]))

    ordered = sorted(ranges, key=lambda item: item[0], reverse=(numbering == "bottom_to_top"))
    beds: list[BedRegion] = []
    for index, (start_y, end_y) in enumerate(ordered, start=1):
        top = min(start_y, end_y)
        bottom = max(start_y, end_y)
        band_mask = field_mask[top : bottom + 1, x_min : x_max + 1]
        area = int(np.count_nonzero(band_mask))
        if area <= 0:
            area = (bottom - top + 1) * (x_max - x_min + 1)
        beds.append(BedRegion(f"bed_{index:02d}", top, bottom, x_min, x_max, area))
    print(f"[BEDS] Detected {len(beds)} beds ({numbering}).")
    return beds


def make_field_mask(raw_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    # Exclude black canvas/background and washed-out white margins. This is a
    # navigation mask, not a vigour classifier.
    mask = (value > 35) & (saturation > 18)
    kernel = np.ones((9, 9), np.uint8)
    mask_u8 = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
    return mask_u8 > 0


def field_bounds(field_mask: np.ndarray) -> tuple[int, int, int, int]:
    rows = np.where(field_mask.mean(axis=1) > 0.04)[0]
    cols = np.where(field_mask.mean(axis=0) > 0.04)[0]
    if rows.size == 0 or cols.size == 0:
        return 0, -1, 0, -1
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    window = max(3, window | 1)
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values.astype(np.float32), kernel, mode="same")


def find_bed_separators(field_mask: np.ndarray, y_min: int, y_max: int) -> list[int]:
    density = field_mask.mean(axis=1)
    band = smooth(density, max(11, field_mask.shape[0] // 180))
    scoped = band[y_min : y_max + 1]
    if scoped.size == 0:
        return []
    median = float(np.median(scoped))
    threshold = min(0.28, median * 0.62)
    gap_rows = np.where(scoped < threshold)[0] + y_min
    if gap_rows.size == 0:
        return []

    separators: list[int] = []
    start = int(gap_rows[0])
    last = int(gap_rows[0])
    min_gap_height = max(5, field_mask.shape[0] // 360)
    min_spacing = max(35, field_mask.shape[0] // 45)
    for row in gap_rows[1:]:
        row = int(row)
        if row == last + 1:
            last = row
            continue
        if last - start + 1 >= min_gap_height:
            center = (start + last) // 2
            if not separators or abs(center - separators[-1]) >= min_spacing:
                separators.append(center)
        start = last = row
    if last - start + 1 >= min_gap_height:
        center = (start + last) // 2
        if not separators or abs(center - separators[-1]) >= min_spacing:
            separators.append(center)
    return [sep for sep in separators if y_min + min_spacing < sep < y_max - min_spacing]


def ranges_from_separators(y_min: int, y_max: int, separators: list[int]) -> list[tuple[int, int]]:
    points = [y_min] + sorted(separators) + [y_max]
    ranges: list[tuple[int, int]] = []
    for start, end in zip(points, points[1:]):
        if end <= start:
            continue
        ranges.append((start, end))
    return ranges


def uniform_ranges(y_min: int, y_max: int, count: int) -> list[tuple[int, int]]:
    count = max(1, count)
    edges = np.linspace(y_min, y_max, count + 1).round().astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(count)]


def estimate_bed_count(image_height: int) -> int:
    # Default navigation fallback. It avoids pretending we know exact agronomic
    # bed boundaries when separator detection is weak, while still giving a
    # consistent bed reference layer.
    return max(6, min(14, round(image_height / 260)))


def assign_bed_id(centroid_y: float, beds: list[BedRegion]) -> BedRegion | None:
    if not beds:
        return None
    for bed in beds:
        if bed.y_min <= centroid_y <= bed.y_max:
            return bed
    return min(beds, key=lambda bed: min(abs(centroid_y - bed.y_min), abs(centroid_y - bed.y_max)))
