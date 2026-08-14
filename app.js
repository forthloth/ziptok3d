const TOKENS = [1, 2, 4, 8, 16, 32, 64, 128];
const PASSES = [1, 2, 3, 4, 5, 6];

const metricConfig = {
  query_iou: { label: "Query IoU", higher: true, residual: value => 100 - value, format: value => value.toFixed(2) },
  mesh_fscore_0p02: { label: "Mesh F1@0.02", higher: true, residual: value => 100 - value, format: value => value.toFixed(2) },
  mesh_cd: { label: "Mesh CD", higher: false, residual: value => value, format: value => value.toFixed(4) },
};

const explorerState = {
  dataset: "ShapeNet",
  metric: "query_iou",
  token: 1,
  passes: 5,
};

const metricData = { ShapeNet: {}, TRELLIS: {} };

const results = {
  ShapeNet: [
    { method: "3DILG", tokens: 512, iou: 95.9, cd: 0.013, f1: 98.0 },
    { method: "VecSet", tokens: 512, iou: 96.3, cd: 0.013, f1: 98.0 },
    { method: "COD-VAE", tokens: 32, iou: 97.1, cd: 0.012, f1: 97.8 },
    { method: "COD-VAE", tokens: 64, iou: 97.5, cd: 0.012, f1: 98.0 },
    { method: "ZipTok3D", tokens: 1, iou: 96.8, cd: 0.012, f1: 97.8, ours: true },
    { method: "ZipTok3D", tokens: 2, iou: 96.9, cd: 0.012, f1: 97.8, ours: true },
    { method: "ZipTok3D", tokens: 4, iou: 96.9, cd: 0.012, f1: 97.9, ours: true },
  ],
  TRELLIS: [
    { method: "VecSet", tokens: 512, iou: 71.47, cd: 0.0249, f1: 88.81 },
    { method: "COD-VAE", tokens: 2, iou: 45.85, cd: 0.0674, f1: 53.95 },
    { method: "COD-VAE", tokens: 32, iou: 75.25, cd: 0.0172, f1: 95.67 },
    { method: "COD-VAE", tokens: 64, iou: 75.75, cd: 0.0168, f1: 95.98 },
    { method: "ZipTok3D", tokens: 1, iou: 75.18, cd: 0.0168, f1: 95.81, ours: true },
    { method: "ZipTok3D", tokens: 2, iou: 75.22, cd: 0.0167, f1: 95.86, ours: true },
    { method: "ZipTok3D", tokens: 4, iou: 75.31, cd: 0.0166, f1: 95.92, ours: true },
  ],
};

const refinementMetrics = {
  1: { iou: 16.85, cd: 0.0328, f1: 69.72, file: "assets/refine-l1.mp4" },
  3: { iou: 52.49, cd: 0.0127, f1: 98.98, file: "assets/refine-l3.mp4" },
  5: { iou: 52.74, cd: 0.0127, f1: 98.94, file: "assets/refine-l5.mp4" },
};

const gallerySources = {
  shapenet: {
    gt: "assets/shapenet-gt.mp4",
    codvae: "assets/shapenet-codvae.mp4",
    ziptok: "assets/shapenet-ziptok.mp4",
  },
  trellis: {
    gt: "assets/trellis-gt.mp4",
    codvae: "assets/trellis-codvae.mp4",
    ziptok: "assets/trellis-ziptok.mp4",
  },
};

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines[0].split(",");
  return lines.slice(1).map(line => {
    const values = line.split(",");
    return Object.fromEntries(headers.map((header, index) => [header, values[index]]));
  });
}

async function loadCsv(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Unable to load ${path}`);
  return parseCsv(await response.text());
}

function storeRows(dataset, queryRows, meshRows) {
  queryRows.filter(row => row.model === "flex").forEach(row => {
    const key = `${Number(row.token_count)}-${Number(row.loop_count)}`;
    metricData[dataset][key] = { query_iou: Number(row.query_iou) };
  });
  meshRows.filter(row => row.model === "flex").forEach(row => {
    const key = `${Number(row.token_count)}-${Number(row.loop_count)}`;
    metricData[dataset][key] = {
      ...metricData[dataset][key],
      mesh_cd: Number(row.mesh_cd),
      mesh_fscore_0p02: Number(row.mesh_fscore_0p02),
    };
  });
}

function mixColor(low, high, amount) {
  const channel = index => Math.round(low[index] + (high[index] - low[index]) * amount);
  return `rgb(${channel(0)}, ${channel(1)}, ${channel(2)})`;
}

function qualityScores(dataset, metric) {
  const config = metricConfig[metric];
  const residuals = Object.values(metricData[dataset]).map(row => config.residual(row[metric]));
  const minimum = Math.min(...residuals);
  const maximum = Math.max(...residuals);
  const denominator = Math.log(maximum / minimum);
  return value => {
    const residual = config.residual(value);
    if (denominator === 0) return 1;
    return Math.max(0, Math.min(1, Math.log(maximum / residual) / denominator));
  };
}

function selectedRow() {
  return metricData[explorerState.dataset][`${explorerState.token}-${explorerState.passes}`];
}

function updateExplorerControls() {
  document.querySelectorAll("[data-dataset]").forEach(button => {
    button.setAttribute("aria-pressed", String(button.dataset.dataset === explorerState.dataset));
  });
  document.querySelectorAll("[data-metric]").forEach(button => {
    button.setAttribute("aria-selected", String(button.dataset.metric === explorerState.metric));
  });
  const tokenIndex = TOKENS.indexOf(explorerState.token);
  document.getElementById("explorer-k").value = String(tokenIndex);
  document.getElementById("explorer-l").value = String(explorerState.passes);
  document.getElementById("explorer-k-value").textContent = String(explorerState.token);
  document.getElementById("explorer-l-value").textContent = String(explorerState.passes);
}

function renderHeatmap() {
  const heatmap = document.getElementById("heatmap");
  const config = metricConfig[explorerState.metric];
  const quality = qualityScores(explorerState.dataset, explorerState.metric);
  heatmap.replaceChildren();

  const corner = document.createElement("div");
  corner.className = "heatmap-corner";
  corner.textContent = "L / K";
  heatmap.appendChild(corner);

  TOKENS.forEach(token => {
    const label = document.createElement("div");
    label.className = "heatmap-axis";
    label.textContent = String(token);
    heatmap.appendChild(label);
  });

  PASSES.forEach(passes => {
    const label = document.createElement("div");
    label.className = "heatmap-axis";
    label.textContent = String(passes);
    heatmap.appendChild(label);

    TOKENS.forEach(token => {
      const row = metricData[explorerState.dataset][`${token}-${passes}`];
      const value = row[explorerState.metric];
      const score = quality(value);
      const button = document.createElement("button");
      button.type = "button";
      button.className = "heatmap-cell";
      button.setAttribute("role", "gridcell");
      button.setAttribute("aria-label", `${explorerState.dataset}, K ${token}, L ${passes}, ${config.label} ${config.format(value)}`);
      button.textContent = config.format(value);
      button.style.backgroundColor = mixColor([225, 238, 232], [8, 115, 99], score);
      button.style.color = score > 0.58 ? "#ffffff" : "#162020";
      if (token === explorerState.token && passes === explorerState.passes) button.classList.add("selected");
      if ((explorerState.dataset === "ShapeNet" && token === 1 && passes === 5) ||
          (explorerState.dataset === "TRELLIS" && token === 4 && passes === 5)) {
        button.classList.add("reported");
      }
      button.addEventListener("click", () => {
        explorerState.token = token;
        explorerState.passes = passes;
        renderExplorer();
      });
      heatmap.appendChild(button);
    });
  });
}

function renderSelection() {
  const row = selectedRow();
  document.getElementById("selection-dataset").textContent = `${explorerState.dataset} operating point`;
  document.getElementById("selection-k").textContent = `K = ${explorerState.token}`;
  document.getElementById("selection-l").textContent = `L = ${explorerState.passes}`;
  document.getElementById("selected-iou").textContent = row.query_iou.toFixed(2);
  document.getElementById("selected-cd").textContent = row.mesh_cd.toFixed(4);
  document.getElementById("selected-f1").textContent = row.mesh_fscore_0p02.toFixed(2);

  const base = metricData[explorerState.dataset][`${explorerState.token}-1`][explorerState.metric];
  const current = row[explorerState.metric];
  const config = metricConfig[explorerState.metric];
  let insight;
  if (explorerState.passes === 1) {
    insight = `Single-pass decoding makes quality depend most strongly on prefix length.`;
  } else {
    const signedChange = current - base;
    const favorable = config.higher ? signedChange : -signedChange;
    const unitChange = explorerState.metric === "mesh_cd" ? Math.abs(signedChange).toFixed(4) : Math.abs(signedChange).toFixed(2);
    const direction = favorable >= 0 ? "improves" : "changes";
    insight = `At K = ${explorerState.token}, ${explorerState.passes} passes ${direction} ${config.label} by ${unitChange} relative to L = 1.`;
  }
  document.getElementById("selection-insight").textContent = insight;
}

function renderExplorer() {
  updateExplorerControls();
  renderHeatmap();
  renderSelection();
}

function renderResults(dataset) {
  document.querySelectorAll("[data-result-dataset]").forEach(button => {
    button.setAttribute("aria-pressed", String(button.dataset.resultDataset === dataset));
  });
  const rows = results[dataset];
  const bestIou = Math.max(...rows.map(row => row.iou));
  const bestCd = Math.min(...rows.map(row => row.cd));
  const bestF1 = Math.max(...rows.map(row => row.f1));
  const body = document.getElementById("results-body");
  body.replaceChildren();
  rows.forEach(row => {
    const tr = document.createElement("tr");
    if (row.ours) tr.className = "ours";
    const cells = [
      { value: row.method },
      { value: row.tokens },
      { value: dataset === "ShapeNet" ? row.iou.toFixed(1) : row.iou.toFixed(2), best: row.iou === bestIou },
      { value: dataset === "ShapeNet" ? row.cd.toFixed(3) : row.cd.toFixed(4), best: row.cd === bestCd },
      { value: dataset === "ShapeNet" ? row.f1.toFixed(1) : row.f1.toFixed(2), best: row.f1 === bestF1 },
    ];
    cells.forEach(cell => {
      const td = document.createElement("td");
      td.textContent = String(cell.value);
      if (cell.best) td.className = "best";
      tr.appendChild(td);
    });
    body.appendChild(tr);
  });
}

function setRefinementDepth(depth) {
  const data = refinementMetrics[depth];
  document.querySelectorAll("[data-depth]").forEach(button => {
    button.setAttribute("aria-pressed", String(Number(button.dataset.depth) === depth));
  });
  document.getElementById("refine-iou").textContent = data.iou.toFixed(2);
  document.getElementById("refine-cd").textContent = data.cd.toFixed(4);
  document.getElementById("refine-f1").textContent = data.f1.toFixed(2);
  document.getElementById("refine-video-label").textContent = `K = 2, L = ${depth}`;
  const video = document.getElementById("refinement-video");
  const previousTime = video.currentTime || 0;
  video.src = data.file;
  video.setAttribute("aria-label", `Bridge reconstruction at ${depth} refinement ${depth === 1 ? "pass" : "passes"}`);
  video.addEventListener("loadedmetadata", () => {
    video.currentTime = Math.min(previousTime, Math.max(0, video.duration - 0.1));
    video.play().catch(() => {});
  }, { once: true });
}

function setGallery(dataset) {
  document.querySelectorAll("[data-gallery]").forEach(button => {
    button.setAttribute("aria-pressed", String(button.dataset.gallery === dataset));
  });
  document.querySelectorAll("#video-grid video").forEach(video => {
    video.src = gallerySources[dataset][video.dataset.role];
    video.load();
    video.play().catch(() => {});
  });
}

function bindControls() {
  document.querySelectorAll("[data-dataset]").forEach(button => {
    button.addEventListener("click", () => {
      explorerState.dataset = button.dataset.dataset;
      explorerState.token = explorerState.dataset === "ShapeNet" ? 1 : 4;
      explorerState.passes = 5;
      renderExplorer();
    });
  });
  document.querySelectorAll("[data-metric]").forEach(button => {
    button.addEventListener("click", () => {
      explorerState.metric = button.dataset.metric;
      renderExplorer();
    });
  });
  document.getElementById("explorer-k").addEventListener("input", event => {
    explorerState.token = TOKENS[Number(event.target.value)];
    renderExplorer();
  });
  document.getElementById("explorer-l").addEventListener("input", event => {
    explorerState.passes = Number(event.target.value);
    renderExplorer();
  });
  document.querySelectorAll("[data-result-dataset]").forEach(button => {
    button.addEventListener("click", () => renderResults(button.dataset.resultDataset));
  });
  document.querySelectorAll("[data-depth]").forEach(button => {
    button.addEventListener("click", () => setRefinementDepth(Number(button.dataset.depth)));
  });
  document.querySelectorAll("[data-gallery]").forEach(button => {
    button.addEventListener("click", () => setGallery(button.dataset.gallery));
  });
}

async function initialize() {
  bindControls();
  renderResults("ShapeNet");
  setGallery("shapenet");
  if (window.lucide) window.lucide.createIcons();

  try {
    const [shapeQuery, shapeMesh, trellisQuery, trellisMesh] = await Promise.all([
      loadCsv("data/shapenet-query.csv"),
      loadCsv("data/shapenet-mesh.csv"),
      loadCsv("data/trellis-query.csv"),
      loadCsv("data/trellis-mesh.csv"),
    ]);
    storeRows("ShapeNet", shapeQuery, shapeMesh);
    storeRows("TRELLIS", trellisQuery, trellisMesh);
    renderExplorer();
  } catch (error) {
    const heatmap = document.getElementById("heatmap");
    heatmap.textContent = "Metric data could not be loaded.";
    console.error(error);
  }
}

document.addEventListener("DOMContentLoaded", initialize);
