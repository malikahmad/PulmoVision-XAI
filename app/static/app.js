const $ = (id) => document.getElementById(id);
const dropzone = $("dropzone");
const input = $("file-input");
const analyze = $("analyze-btn");
let selectedFile = null;
let previewUrl = null;
let responseData = null;
let busy = false;

const views = {
  overlay: { label: "OVERLAY VIEW", alt: "Original X-ray with Grad-CAM activation overlay" },
  heatmap: { label: "HEATMAP VIEW", alt: "Grad-CAM activation heatmap" },
  original: { label: "ORIGINAL IMAGE", alt: "Original resized chest X-ray" }
};

function toast(message) {
  const el = $("toast");
  el.textContent = message;
  el.classList.add("show");
  window.clearTimeout(toast.timer);
  toast.timer = window.setTimeout(() => el.classList.remove("show"), 3000);
}

function setBusy(value) {
  busy = value;
  analyze.disabled = value || !selectedFile;
  $("btn-spinner").classList.toggle("hidden", !value);
  $("analyze-btn").querySelector(".btn-label").textContent = value ? "Analyzing image…" : "Analyze image";
  analyze.querySelector(".btn-arrow").classList.toggle("hidden", value);
}

function clearResult() {
  responseData = null;
  $("results").classList.add("hidden");
  $("empty-state").classList.remove("hidden");
  $("loading-state").classList.add("hidden");
  $("error-state").classList.add("hidden");
  $("confidence-meter").style.width = "0";
}

function setFile(file) {
  if (!file) return;
  const allowed = ["image/jpeg", "image/png", "image/webp"];
  if (!allowed.includes(file.type)) {
    toast("Please choose a JPG, PNG, or WEBP image.");
    return;
  }
  if (file.size > 12 * 1024 * 1024) {
    toast("This image is larger than 12 MB. Choose a smaller file.");
    return;
  }
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  selectedFile = file;
  previewUrl = URL.createObjectURL(file);
  $("preview").src = previewUrl;
  $("file-name").textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
  $("preview-wrap").classList.remove("hidden");
  dropzone.classList.add("hidden");
  clearResult();
  analyze.disabled = false;
  $("analyze-btn").querySelector(".btn-label").textContent = "Analyze image";
}

function clearFile() {
  if (busy) return;
  selectedFile = null;
  input.value = "";
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  $("preview").removeAttribute("src");
  $("preview-wrap").classList.add("hidden");
  dropzone.classList.remove("hidden");
  analyze.disabled = true;
  clearResult();
}

function showError(message) {
  $("loading-state").classList.add("hidden");
  $("results").classList.add("hidden");
  $("empty-state").classList.add("hidden");
  $("error-message").textContent = message;
  $("error-state").classList.remove("hidden");
}

$("browse-btn").addEventListener("click", (event) => {
  event.stopPropagation();
  input.click();
});
dropzone.addEventListener("click", () => input.click());
dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    input.click();
  }
});
input.addEventListener("change", (event) => {
  setFile(event.target.files?.[0]);
  input.value = "";
});
$("remove-btn").addEventListener("click", clearFile);
$("reset-btn").addEventListener("click", clearFile);

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragover");
  });
});
["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
  });
});
dropzone.addEventListener("drop", (event) => {
  const file = event.dataTransfer?.files?.[0];
  if (file) setFile(file);
});

async function analyzeImage() {
  if (!selectedFile || busy) return;
  setBusy(true);
  $("empty-state").classList.add("hidden");
  $("error-state").classList.add("hidden");
  $("results").classList.add("hidden");
  $("loading-state").classList.remove("hidden");

  try {
    const body = new FormData();
    body.append("file", selectedFile, selectedFile.name);
    const response = await fetch("/api/predict", { method: "POST", body });
    let data;
    try {
      data = await response.json();
    } catch {
      throw new Error("The server returned an unreadable response. Please try again.");
    }
    if (!response.ok) {
      throw new Error(data.detail || `Analysis failed (HTTP ${response.status}).`);
    }
    if (!data.images || !data.images.overlay || !data.images.heatmap || !data.images.original) {
      throw new Error("The API response is missing visualization data.");
    }
    responseData = data;
    renderResults(data);
    toast("Analysis completed.");
  } catch (error) {
    showError(error.message || "Something went wrong. Please try again.");
  } finally {
    setBusy(false);
  }
}

function renderResults(data) {
  const isPneumonia = data.class_index === 1 || data.label?.toLowerCase() === "pneumonia";
  $("prediction-label").textContent = data.label || "Unknown";
  $("prediction-symbol").textContent = isPneumonia ? "●" : "○";
  $("prediction-symbol").style.color = isPneumonia ? "#c17b42" : "#168a78";
  $("prediction-description").textContent = isPneumonia
    ? "The model classified this image as Pneumonia."
    : "The model classified this image as Normal.";
  $("confidence-value").textContent = `${Number(data.confidence).toFixed(2)}%`;
  $("confidence-meter").style.width = `${Math.max(0, Math.min(100, Number(data.confidence) || 0))}%`;
  $("class-index").textContent = data.class_index ?? "—";
  $("layer-name").textContent = data.gradcam_layer ? `LAYER · ${data.gradcam_layer}` : "Grad-CAM";
  $("score-note").textContent = data.model_output_note || "Not a calibrated clinical probability";
  $("empty-state").classList.add("hidden");
  $("loading-state").classList.add("hidden");
  $("error-state").classList.add("hidden");
  $("results").classList.remove("hidden");
  setView("overlay");
}

function setView(view) {
  if (!responseData || !views[view]) return;
  $("result-image").src = responseData.images[view];
  $("result-image").alt = views[view].alt;
  $("image-tag").textContent = views[view].label;
  document.querySelectorAll(".tab").forEach((tab) => {
    const active = tab.dataset.view === view;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-pressed", String(active));
  });
}
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => setView(tab.dataset.view));
});
analyze.addEventListener("click", analyzeImage);

async function checkHealth() {
  try {
    const response = await fetch("/api/health", { cache: "no-store" });
    if (!response.ok) throw new Error("API unavailable");
    const data = await response.json();
    const online = Boolean(data.model_loaded);
    $("model-status").textContent = online ? "AI model ready" : "Model unavailable";
    $("status-dot").classList.toggle("online", online);
    $("status-dot").title = data.message || (online ? "Model loaded" : "Model not loaded");
    $("analyze-btn").title = online ? "Analyze selected image" : "The model is not currently available";
  } catch {
    $("model-status").textContent = "API unavailable";
    $("status-dot").classList.remove("online");
    $("status-dot").title = "Could not reach the API";
  }
}
checkHealth();
