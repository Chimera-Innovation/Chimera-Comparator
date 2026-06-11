const DATES = [
  {
    "date": "2026-05-08",
    "growthStage": "Bloom / early fruit set",
    "source": "McIntyre-Road-5-8-2026-orthophoto-NDVI.jpg",
    "raw": "assets/2026-05-08/raw.png",
    "human": "assets/2026-05-08/human_annotation_overlay.png",
    "density": "assets/2026-05-08/concern_density.png",
    "preview": "assets/2026-05-08/vigour_zone_preview.png",
    "uncertainty": "assets/2026-05-08/uncertainty.png",
    "print": "assets/2026-05-08/print.png"
  },
  {
    "date": "2026-05-12",
    "growthStage": "Bloom to fruit set",
    "source": "Mallard-Avenue-5-12-2026-orthophoto-NDVI.png",
    "raw": "assets/2026-05-12/raw.png",
    "human": "assets/2026-05-12/human_annotation_overlay.png",
    "density": "assets/2026-05-12/concern_density.png",
    "preview": "assets/2026-05-12/vigour_zone_preview.png",
    "uncertainty": "assets/2026-05-12/uncertainty.png",
    "print": "assets/2026-05-12/print.png"
  },
  {
    "date": "2026-05-18",
    "growthStage": "Green fruit development",
    "source": "Mallard-Avenue-5-18-2026-orthophoto-NDVI.tif",
    "raw": "assets/2026-05-18/raw.png",
    "human": "assets/2026-05-18/human_annotation_overlay.png",
    "density": "assets/2026-05-18/concern_density.png",
    "preview": "assets/2026-05-18/vigour_zone_preview.png",
    "uncertainty": "assets/2026-05-18/uncertainty.png",
    "print": "assets/2026-05-18/print.png"
  },
  {
    "date": "2026-05-22",
    "growthStage": "Fruit sizing / early ripening",
    "source": "Mallard-Avenue-5-22-2026-orthophoto-NDVI.png",
    "raw": "assets/2026-05-22/raw.png",
    "human": "assets/2026-05-22/human_annotation_overlay.png",
    "density": "assets/2026-05-22/concern_density.png",
    "preview": "assets/2026-05-22/vigour_zone_preview.png",
    "uncertainty": "assets/2026-05-22/uncertainty.png",
    "print": "assets/2026-05-22/print.png"
  },
  {
    "date": "2026-05-27",
    "growthStage": "Ripening / early harvest window",
    "source": "McIntyre-Road-5-27-2026-orthophoto-NDVI.jpg",
    "raw": "assets/2026-05-27/raw.png",
    "human": "assets/2026-05-27/human_annotation_overlay.png",
    "density": "assets/2026-05-27/concern_density.png",
    "preview": "assets/2026-05-27/vigour_zone_preview.png",
    "uncertainty": "assets/2026-05-27/uncertainty.png",
    "print": "assets/2026-05-27/print.png"
  },
  {
    "date": "2026-05-29",
    "growthStage": "Harvest-time ripening",
    "source": "Mallard-Avenue-5-29-2026-orthophoto-NDVI.png",
    "raw": "assets/2026-05-29/raw.png",
    "human": "assets/2026-05-29/human_annotation_overlay.png",
    "density": "assets/2026-05-29/concern_density.png",
    "preview": "assets/2026-05-29/vigour_zone_preview.png",
    "uncertainty": "assets/2026-05-29/uncertainty.png",
    "print": "assets/2026-05-29/print.png"
  }
];

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
