from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from strawberry_vigour_polygon_report.src.annotation_parser import (
    get_annotation_mask,
    human_annotation_colour_masks,
    resize_annotated_to_raw,
)
from strawberry_vigour_polygon_report.src.growth_stage import growth_stage_for_date


ROOT = Path(__file__).resolve().parent
PRODUCT_DIR = ROOT / "human_anchored_vigour_zone_preview_product"
PRINT_DIR = ROOT / "daily_outputs"
VIEWER_DIR = ROOT / "web_viewer"
ASSETS_DIR = VIEWER_DIR / "assets"
DATES = ["2026-05-08", "2026-05-12", "2026-05-18", "2026-05-22", "2026-05-27", "2026-05-29"]

COLORS_BGR = {
    "low_vigour": (35, 35, 235),
    "medium_vigour": (215, 90, 190),
    "high_vigour": (80, 235, 120),
}


def read_summary() -> dict[str, dict[str, str]]:
    path = PRODUCT_DIR / "product_audit_summary.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["date"]: row for row in csv.DictReader(handle)}


def read_temporal_analytics() -> dict[str, dict[str, str]]:
    path = ROOT / "temporal_analytics.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["date"]: row for row in csv.DictReader(handle)}


def load(path: Path | str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def resize_to_width(image: np.ndarray, width: int = 1800) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= width:
        return image
    scale = width / w
    return cv2.resize(image, (width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)


def rgba_from_layer(layer_bgr: np.ndarray, active: np.ndarray, alpha: int = 210) -> np.ndarray:
    rgba = cv2.cvtColor(layer_bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = np.where(active, alpha, 0).astype(np.uint8)
    return rgba


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def human_annotation_overlay(raw_bgr: np.ndarray, annotated_path: str) -> np.ndarray:
    annotated = load(annotated_path)
    annotated, _ = resize_annotated_to_raw(raw_bgr, annotated)
    annotation_mask = get_annotation_mask(raw_bgr, annotated)
    masks = human_annotation_colour_masks(raw_bgr, annotated, annotation_mask)
    layer = np.zeros_like(raw_bgr)
    active = np.zeros(raw_bgr.shape[:2], dtype=bool)
    for name, mask in masks.items():
        m = mask > 0
        layer[m] = COLORS_BGR[name]
        active |= m
    layer = cv2.dilate(layer, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    active = np.any(layer > 0, axis=2)
    return rgba_from_layer(layer, active, 235)


def concern_density_overlay(seed_polygons: np.ndarray) -> np.ndarray:
    mask = np.any(seed_polygons > 0, axis=2).astype(np.float32)
    blur = cv2.GaussianBlur(mask, (0, 0), sigmaX=34, sigmaY=34)
    if blur.max() > 0:
        blur = blur / blur.max()
    layer = np.zeros((*seed_polygons.shape[:2], 3), dtype=np.uint8)
    layer[:, :, 1] = np.clip(185 * blur, 0, 185).astype(np.uint8)
    layer[:, :, 2] = np.clip(255 * blur, 0, 255).astype(np.uint8)
    alpha = np.clip(190 * blur, 0, 190).astype(np.uint8)
    rgba = cv2.cvtColor(layer, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    return rgba


def preview_overlay(preview: np.ndarray) -> np.ndarray:
    active = np.any(preview > 0, axis=2)
    return rgba_from_layer(preview, active, 210)


def uncertainty_overlay(uncertainty: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(uncertainty, cv2.COLOR_BGR2HSV)
    active = (hsv[:, :, 1] > 45) & (hsv[:, :, 2] > 60)
    return rgba_from_layer(uncertainty, active, 185)


def build_assets() -> list[dict[str, str]]:
    summary = read_summary()
    temporal = read_temporal_analytics()
    rows: list[dict[str, str]] = []
    for date in DATES:
        row = summary[date]
        temporal_row = temporal.get(date, {})
        date_dir = PRODUCT_DIR / date
        out_dir = ASSETS_DIR / date
        raw = resize_to_width(load(date_dir / "raw_ndvi.png"))
        seed = cv2.resize(load(date_dir / "seed_polygons.png"), (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_AREA)
        preview = cv2.resize(load(date_dir / "human_anchored_vigour_preview.png"), (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_NEAREST)
        uncertainty = cv2.resize(load(date_dir / "concern_uncertainty_panel.png"), (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_AREA)

        write_png(out_dir / "raw.png", raw)
        write_png(out_dir / "human_annotation_overlay.png", human_annotation_overlay(raw, row["annotated_path"]))
        write_png(out_dir / "concern_density.png", concern_density_overlay(seed))
        write_png(out_dir / "vigour_zone_preview.png", preview_overlay(preview))
        write_png(out_dir / "uncertainty.png", uncertainty_overlay(uncertainty))
        shutil.copyfile(PRINT_DIR / f"{date}_print.png", out_dir / "print.png")
        rows.append(
            {
                "date": date,
                "growthStage": growth_stage_for_date(date),
                "source": Path(row["raw_path"]).name,
                "raw": f"assets/{date}/raw.png",
                "human": f"assets/{date}/human_annotation_overlay.png",
                "density": f"assets/{date}/concern_density.png",
                "preview": f"assets/{date}/vigour_zone_preview.png",
                "uncertainty": f"assets/{date}/uncertainty.png",
                "print": f"assets/{date}/print.png",
                "percentFieldAffected": temporal_row.get("percent_field_affected", ""),
                "newConcern": temporal_row.get("new_concern_percent_field", ""),
                "persistentConcern": temporal_row.get("persistent_concern_percent_field", ""),
                "recovered": temporal_row.get("recovered_percent_field", ""),
            }
        )
    return rows


def write_viewer(rows: list[dict[str, str]]) -> None:
    VIEWER_DIR.mkdir(parents=True, exist_ok=True)
    (VIEWER_DIR / "index.html").write_text(
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Strawberry 1 | Human-Anchored Vigour Review</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <header class="topbar">
    <div class="brand">Strawberry 1</div>
    <label>Date <select id="dateSelect"></select></label>
    <div id="growthStage" class="stage"></div>
  </header>
  <aside class="left-panel">
    <h2>Layers</h2>
    <div id="layerControls"></div>
  </aside>
  <main class="viewport">
    <div id="mapFrame" class="map-frame">
      <img id="rawLayer" class="map-layer" alt="Raw NDVI">
      <img id="humanLayer" class="map-layer overlay" alt="Human Annotation">
      <img id="densityLayer" class="map-layer overlay" alt="Concern Density">
      <img id="previewLayer" class="map-layer overlay" alt="Vigour Zone Preview">
      <img id="uncertaintyLayer" class="map-layer overlay" alt="Uncertainty">
    </div>
  </main>
  <aside class="right-panel">
    <h2>Quick scouting note</h2>
    <p>Inspect marked concern areas.</p>
    <p>Compare with previous date.</p>
    <p>Ground check required.</p>
    <div id="temporalMetrics" class="metrics"></div>
    <figure class="chart">
      <img src="assets/temporal_timeline_chart.png" alt="Temporal concern area chart">
      <figcaption>Concern area history</figcaption>
    </figure>
    <div class="notice">Vigour Zone Preview is exploratory. Not validated automatic classification.</div>
    <button id="prevBtn">Previous</button>
    <button id="nextBtn">Next date</button>
  </aside>
  <footer class="timeline" id="timeline"></footer>
  <script src="app.js"></script>
</body>
</html>
""",
        encoding="utf-8",
    )
    (VIEWER_DIR / "app.js").write_text(f"const DATES = {json.dumps(rows, indent=2)};\n" + APP_JS, encoding="utf-8")
    (VIEWER_DIR / "styles.css").write_text(STYLES_CSS, encoding="utf-8")
    write_product_audit(rows)


def write_product_audit(rows: list[dict[str, str]]) -> None:
    checks = []
    for row in rows:
        date = row["date"]
        required = [
            ROOT / "daily_outputs" / f"{date}_print.png",
            ROOT / "temporal_analytics.csv",
            ASSETS_DIR / "temporal_timeline_chart.png",
            ASSETS_DIR / date / "raw.png",
            ASSETS_DIR / date / "human_annotation_overlay.png",
            ASSETS_DIR / date / "concern_density.png",
            ASSETS_DIR / date / "vigour_zone_preview.png",
            ASSETS_DIR / date / "uncertainty.png",
            ASSETS_DIR / date / "print.png",
        ]
        checks.append(
            {
                "date": date,
                "growth_stage": row["growthStage"],
                "print_output_exists": str((ROOT / "daily_outputs" / f"{date}_print.png").exists()),
                "viewer_assets_exist": str(all(path.exists() for path in required)),
                "temporal_analytics_exists": str((ROOT / "temporal_analytics.csv").exists()),
                "timeline_chart_exists": str((ASSETS_DIR / "temporal_timeline_chart.png").exists()),
                "default_layers": "Raw NDVI + Human Annotation",
                "status": "PASS" if all(path.exists() for path in required) else "REVIEW",
                "note": "Human-Anchored Vigour Review; scouting prioritization only.",
            }
        )
    with (ROOT / "product_audit_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "date",
                "growth_stage",
                "print_output_exists",
                "viewer_assets_exist",
                "temporal_analytics_exists",
                "timeline_chart_exists",
                "default_layers",
                "status",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(checks)


APP_JS = r"""
const layers = [
  { id: "raw", label: "Raw NDVI", element: "rawLayer", defaultOn: true, defaultOpacity: 1 },
  { id: "human", label: "Human Annotation", element: "humanLayer", defaultOn: true, defaultOpacity: 0.9 },
  { id: "density", label: "Concern Density", element: "densityLayer", defaultOn: false, defaultOpacity: 0.65 },
  { id: "preview", label: "Vigour Zone Preview", element: "previewLayer", defaultOn: false, defaultOpacity: 0.8 },
  { id: "uncertainty", label: "Uncertainty", element: "uncertaintyLayer", defaultOn: false, defaultOpacity: 0.75 },
];

let currentIndex = 0;
const state = Object.fromEntries(layers.map(layer => [layer.id, { on: layer.defaultOn, opacity: layer.defaultOpacity }]));

function el(id) {
  return document.getElementById(id);
}

function buildControls() {
  const dateSelect = el("dateSelect");
  DATES.forEach((entry, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = entry.date;
    dateSelect.appendChild(option);
  });
  dateSelect.addEventListener("change", event => {
    currentIndex = Number(event.target.value);
    render();
  });

  const controls = el("layerControls");
  layers.forEach(layer => {
    const row = document.createElement("div");
    row.className = "layer-control";
    row.innerHTML = `
      <label class="check"><input type="checkbox" data-layer="${layer.id}" ${layer.defaultOn ? "checked" : ""}> ${layer.label}</label>
      <input type="range" min="0" max="1" step="0.05" value="${layer.defaultOpacity}" data-opacity="${layer.id}">
    `;
    controls.appendChild(row);
  });
  controls.addEventListener("input", event => {
    const opacityId = event.target.dataset.opacity;
    if (opacityId) {
      state[opacityId].opacity = Number(event.target.value);
      applyLayerState();
    }
  });
  controls.addEventListener("change", event => {
    const layerId = event.target.dataset.layer;
    if (layerId) {
      state[layerId].on = event.target.checked;
      applyLayerState();
    }
  });

  el("prevBtn").addEventListener("click", () => {
    currentIndex = Math.max(0, currentIndex - 1);
    render();
  });
  el("nextBtn").addEventListener("click", () => {
    currentIndex = Math.min(DATES.length - 1, currentIndex + 1);
    render();
  });
}

function buildTimeline() {
  const timeline = el("timeline");
  timeline.innerHTML = "";
  DATES.forEach((entry, index) => {
    const button = document.createElement("button");
    button.className = "thumb";
    button.innerHTML = `<img src="${entry.print}" alt="${entry.date} print"><span>${entry.date}</span>`;
    button.addEventListener("click", () => {
      currentIndex = index;
      render();
    });
    timeline.appendChild(button);
  });
}

function render() {
  const entry = DATES[currentIndex];
  el("dateSelect").value = String(currentIndex);
  el("growthStage").textContent = entry.growthStage;
  el("rawLayer").src = entry.raw;
  el("humanLayer").src = entry.human;
  el("densityLayer").src = entry.density;
  el("previewLayer").src = entry.preview;
  el("uncertaintyLayer").src = entry.uncertainty;
  el("temporalMetrics").innerHTML = `
    <div><strong>${entry.percentFieldAffected || "0"}%</strong><span>field affected</span></div>
    <div><strong>${entry.newConcern || "0"}%</strong><span>new</span></div>
    <div><strong>${entry.persistentConcern || "0"}%</strong><span>persistent</span></div>
    <div><strong>${entry.recovered || "0"}%</strong><span>recovered</span></div>
  `;
  document.querySelectorAll(".thumb").forEach((thumb, index) => thumb.classList.toggle("active", index === currentIndex));
  applyLayerState();
}

function applyLayerState() {
  layers.forEach(layer => {
    const image = el(layer.element);
    image.style.display = state[layer.id].on ? "block" : "none";
    image.style.opacity = String(state[layer.id].opacity);
  });
}

buildControls();
buildTimeline();
render();
"""


STYLES_CSS = r"""
* { box-sizing: border-box; }
body {
  margin: 0;
  min-height: 100vh;
  display: grid;
  grid-template-columns: 260px minmax(0, 1fr) 280px;
  grid-template-rows: 58px minmax(0, 1fr) 126px;
  grid-template-areas:
    "top top top"
    "left center right"
    "timeline timeline timeline";
  font-family: Arial, sans-serif;
  background: #0b1118;
  color: #edf2f7;
}
.topbar {
  grid-area: top;
  display: flex;
  align-items: center;
  gap: 22px;
  padding: 0 18px;
  background: #121b26;
  border-bottom: 1px solid #2a3848;
}
.brand { font-size: 20px; font-weight: 700; }
select { background: #0d141d; color: #edf2f7; border: 1px solid #415064; padding: 7px 9px; }
.stage { color: #b8c4d2; }
.left-panel, .right-panel {
  padding: 18px;
  background: #101821;
  border-color: #2a3848;
}
.left-panel { grid-area: left; border-right: 1px solid #2a3848; }
.right-panel { grid-area: right; border-left: 1px solid #2a3848; }
h2 { font-size: 16px; margin: 0 0 14px; }
.layer-control { padding: 12px 0; border-bottom: 1px solid #253242; }
.check { display: block; margin-bottom: 8px; }
input[type="range"] { width: 100%; }
.viewport {
  grid-area: center;
  display: grid;
  place-items: center;
  padding: 18px;
  min-width: 0;
}
.map-frame {
  position: relative;
  width: 100%;
  height: 100%;
  background: #05070a;
  overflow: hidden;
  border: 1px solid #2a3848;
}
.map-layer {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
}
.overlay { pointer-events: none; }
.right-panel p { margin: 0 0 10px; color: #d5dde7; }
.metrics {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin: 16px 0;
}
.metrics div {
  padding: 9px;
  background: #0d141d;
  border: 1px solid #2d3b4d;
}
.metrics strong {
  display: block;
  font-size: 20px;
}
.metrics span {
  color: #aeb9c7;
  font-size: 12px;
}
.chart {
  margin: 14px 0;
}
.chart img {
  width: 100%;
  display: block;
  background: white;
}
.chart figcaption {
  margin-top: 5px;
  color: #aeb9c7;
  font-size: 12px;
}
.notice {
  margin: 18px 0;
  padding: 12px;
  background: #1b2633;
  border: 1px solid #38495c;
  color: #c5cfda;
  line-height: 1.35;
}
button {
  background: #223246;
  color: #edf2f7;
  border: 1px solid #40536a;
  padding: 9px 12px;
  cursor: pointer;
}
button:hover { background: #2b3d54; }
.timeline {
  grid-area: timeline;
  display: flex;
  gap: 10px;
  padding: 12px;
  overflow-x: auto;
  background: #111923;
  border-top: 1px solid #2a3848;
}
.thumb {
  width: 154px;
  flex: 0 0 auto;
  padding: 5px;
  background: #0d141d;
}
.thumb.active { border-color: #f0b429; }
.thumb img {
  width: 100%;
  height: 76px;
  object-fit: cover;
  display: block;
}
.thumb span {
  display: block;
  margin-top: 4px;
  font-size: 12px;
}
"""


def main() -> None:
    rows = build_assets()
    write_viewer(rows)
    print(f"[WRITE] {VIEWER_DIR}")


if __name__ == "__main__":
    main()
