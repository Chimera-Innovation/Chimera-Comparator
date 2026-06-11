from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from strawberry_vigour_polygon_report.src.growth_stage import growth_stage_for_date


ROOT = Path(__file__).resolve().parent
PRODUCT_DIR = ROOT / "human_anchored_vigour_zone_preview_product"
DAILY_DIR = ROOT / "daily_outputs"
WEB_ASSETS_DIR = ROOT / "web_viewer" / "assets"
DATES = ["2026-05-08", "2026-05-12", "2026-05-18", "2026-05-22", "2026-05-27", "2026-05-29"]


def load(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def field_mask(raw: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    mask = (gray > 8) & (gray < 248)
    return mask


def concern_mask(date: str) -> np.ndarray:
    seed = load(PRODUCT_DIR / date / "seed_polygons.png")
    return np.any(seed > 0, axis=2)


def resized_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0


def percent(value: int, denominator: int) -> float:
    return round(100.0 * value / max(1, denominator), 3)


def compute_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    previous: np.ndarray | None = None
    for date in DATES:
        raw = load(PRODUCT_DIR / date / "raw_ndvi.png")
        current = concern_mask(date)
        field = field_mask(raw)
        if current.shape != field.shape:
            current = resized_mask(current, field.shape)

        field_pixels = int(np.count_nonzero(field))
        concern_pixels = int(np.count_nonzero(current & field))

        if previous is None:
            prev_aligned = np.zeros_like(current, dtype=bool)
        else:
            prev_aligned = resized_mask(previous, current.shape)
            prev_aligned &= field

        current_field = current & field
        new_mask = current_field & ~prev_aligned
        persistent_mask = current_field & prev_aligned
        recovered_mask = prev_aligned & ~current_field

        rows.append(
            {
                "date": date,
                "growth_stage": growth_stage_for_date(date),
                "field_pixels": field_pixels,
                "total_concern_area_pixels": concern_pixels,
                "percent_field_affected": percent(concern_pixels, field_pixels),
                "new_concern_area_pixels": int(np.count_nonzero(new_mask)),
                "new_concern_percent_field": percent(int(np.count_nonzero(new_mask)), field_pixels),
                "persistent_concern_area_pixels": int(np.count_nonzero(persistent_mask)),
                "persistent_concern_percent_field": percent(int(np.count_nonzero(persistent_mask)), field_pixels),
                "recovered_area_pixels": int(np.count_nonzero(recovered_mask)),
                "recovered_percent_field": percent(int(np.count_nonzero(recovered_mask)), field_pixels),
                "comparison_note": "First flight baseline" if previous is None else "Compared with previous matched flight",
            }
        )
        previous = current_field.copy()
    return rows


def write_csv(rows: list[dict[str, object]]) -> Path:
    out = ROOT / "temporal_analytics.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out


def draw_timeline_chart(rows: list[dict[str, object]]) -> Path:
    width, height = 1500, 760
    margin_l, margin_r, margin_t, margin_b = 150, 60, 90, 120
    chart_w = width - margin_l - margin_r
    chart_h = height - margin_t - margin_b
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, "Temporal Concern Area History", (48, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (30, 36, 44), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Human-marked concern polygons only. Compared flight-to-flight.", (50, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (88, 98, 110), 1, cv2.LINE_AA)

    max_val = max(
        1.0,
        max(float(row["percent_field_affected"]) for row in rows),
        max(float(row["new_concern_percent_field"]) for row in rows),
        max(float(row["persistent_concern_percent_field"]) for row in rows),
        max(float(row["recovered_percent_field"]) for row in rows),
    )
    y_max = max(10.0, np.ceil(max_val / 5.0) * 5.0)
    for tick in range(0, int(y_max) + 1, 5):
        y = int(margin_t + chart_h - chart_h * tick / y_max)
        cv2.line(canvas, (margin_l, y), (width - margin_r, y), (225, 230, 236), 1)
        cv2.putText(canvas, f"{tick}%", (70, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (83, 93, 106), 1, cv2.LINE_AA)

    n = len(rows)
    group_w = chart_w / n
    bar_w = int(group_w * 0.16)
    series = [
        ("percent_field_affected", (75, 112, 190), "Total"),
        ("new_concern_percent_field", (35, 35, 235), "New"),
        ("persistent_concern_percent_field", (60, 170, 90), "Persistent"),
        ("recovered_percent_field", (215, 150, 40), "Recovered"),
    ]
    for i, row in enumerate(rows):
        center = int(margin_l + group_w * (i + 0.5))
        offsets = [-1.8, -0.6, 0.6, 1.8]
        for (key, color, _label), offset in zip(series, offsets):
            value = float(row[key])
            x1 = int(center + offset * bar_w)
            x2 = x1 + bar_w
            y1 = int(margin_t + chart_h - chart_h * value / y_max)
            cv2.rectangle(canvas, (x1, y1), (x2, margin_t + chart_h), color, cv2.FILLED)
        cv2.putText(canvas, str(row["date"])[5:], (center - 34, margin_t + chart_h + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (42, 48, 56), 1, cv2.LINE_AA)
        stage = str(row["growth_stage"]).split("/")[0].strip()
        cv2.putText(canvas, stage[:22], (center - 70, margin_t + chart_h + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 100, 112), 1, cv2.LINE_AA)

    legend_x = margin_l
    legend_y = height - 32
    for _key, color, label in series:
        cv2.rectangle(canvas, (legend_x, legend_y - 16), (legend_x + 26, legend_y + 5), color, cv2.FILLED)
        cv2.putText(canvas, label, (legend_x + 36, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (40, 46, 54), 1, cv2.LINE_AA)
        legend_x += 210

    cv2.rectangle(canvas, (margin_l, margin_t), (width - margin_r, margin_t + chart_h), (125, 135, 148), 1)
    out = DAILY_DIR / "temporal_timeline_chart.png"
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    web_out = WEB_ASSETS_DIR / "temporal_timeline_chart.png"
    web_out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(web_out), canvas)
    return out


def main() -> None:
    rows = compute_rows()
    csv_path = write_csv(rows)
    chart_path = draw_timeline_chart(rows)
    print(f"[WRITE] {csv_path}")
    print(f"[WRITE] {chart_path}")


if __name__ == "__main__":
    main()
