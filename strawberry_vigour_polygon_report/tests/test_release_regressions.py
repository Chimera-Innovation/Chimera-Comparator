from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.annotation_parser import CLASS_SCHEMA, PolygonRecord  # noqa: E402
from src.human_guided_polygons import HumanGuidedConfig, build_human_guided_polygons  # noqa: E402
from src.report_generator import block_label_for_bounds, class_percent  # noqa: E402
from src.row_detection import RowRegion  # noqa: E402


def low_mask(shape: tuple[int, int], bounds: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    x_min, y_min, x_max, y_max = bounds
    cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, cv2.FILLED)
    return mask


def synthetic_raw(shape: tuple[int, int]) -> np.ndarray:
    raw = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    raw[:, :] = (35, 80, 190)
    return raw


class ReleaseRegressionTests(unittest.TestCase):
    def test_polygons_stay_inside_human_buffer_influence(self) -> None:
        shape = (120, 180)
        class_masks = {
            "low_vigour": low_mask(shape, (30, 45, 55, 70)),
            "medium_vigour": np.zeros(shape, dtype=np.uint8),
            "high_vigour": np.zeros(shape, dtype=np.uint8),
        }
        records = build_human_guided_polygons(
            class_masks,
            synthetic_raw(shape),
            "Test Field",
            "2026-05-29",
            "Harvest-time ripening",
            [],
            HumanGuidedConfig(
                buffer_distance_pixels=12,
                min_area_pixels=20,
                max_percent_field_area=25.0,
                min_human_overlap_ratio=0.01,
                min_buffer_overlap_ratio=0.95,
                write_debug_outputs=False,
            ),
        )
        self.assertGreater(len(records), 0)
        for record in records:
            xs = [point[0] for point in record.coordinates]
            ys = [point[1] for point in record.coordinates]
            self.assertGreaterEqual(min(xs), 18)
            self.assertLessEqual(max(xs), 67)
            self.assertGreaterEqual(min(ys), 33)
            self.assertLessEqual(max(ys), 82)

    def test_field_wide_human_mark_is_rejected_by_area_guardrail(self) -> None:
        shape = (120, 180)
        class_masks = {
            "low_vigour": np.full(shape, 255, dtype=np.uint8),
            "medium_vigour": np.zeros(shape, dtype=np.uint8),
            "high_vigour": np.zeros(shape, dtype=np.uint8),
        }
        records = build_human_guided_polygons(
            class_masks,
            synthetic_raw(shape),
            "Test Field",
            "2026-05-29",
            "Harvest-time ripening",
            [],
            HumanGuidedConfig(
                buffer_distance_pixels=12,
                min_area_pixels=20,
                max_percent_field_area=5.0,
                write_debug_outputs=False,
            ),
        )
        self.assertEqual(records, [])

    def test_customer_percentages_are_capped_at_100(self) -> None:
        record = PolygonRecord(
            field_name="Test Field",
            date="2026-05-29",
            growth_stage="Harvest-time ripening",
            class_name="low_vigour",
            class_id=CLASS_SCHEMA["low_vigour"],
            polygon_id="P001",
            bed_id="rows_001_001",
            centroid_x=10,
            centroid_y=10,
            area_pixels=250,
            area_percent=250,
            bed_area_pixels=100,
            area_percent_within_bed=100,
            scouting_priority="highest priority",
            confidence_source="test",
            note="test",
            mean_green_index=0,
            mean_visual_ndvi_proxy=0,
            coordinates=[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
        )
        self.assertEqual(class_percent([record], "low_vigour", 100), 100.0)

    def test_no_canny_usage_in_human_guided_extractor(self) -> None:
        source = (ROOT / "src" / "human_guided_polygons.py").read_text(encoding="utf-8")
        self.assertNotIn("Canny", source)

    def test_row_labels_hidden_below_confidence_gate(self) -> None:
        rows = [
            RowRegion(
                row_number=index + 1,
                row_id=f"row_{index + 1:03d}",
                bed_id=f"row_{index + 1:03d}",
                y_min=index * 10,
                y_max=index * 10 + 8,
                x_min=0,
                x_max=100,
                area_pixels=900,
                confidence=0.4,
            )
            for index in range(10)
        ]
        label, row_text = block_label_for_bounds((0, 20, 100, 38), 100, 100, rows, 0.4)
        self.assertNotIn("Rows", label)
        self.assertNotIn("Rows", row_text)

        high_label, high_row_text = block_label_for_bounds((0, 20, 100, 38), 100, 100, rows, 0.95)
        self.assertIn("Approx. Rows", high_label)
        self.assertIn("Rows", high_row_text)


if __name__ == "__main__":
    unittest.main()
