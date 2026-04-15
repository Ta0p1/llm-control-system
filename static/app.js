const healthPill = document.getElementById("health-pill");
const modelPill = document.getElementById("model-pill");
const knowledgeDir = document.getElementById("knowledge-dir");
const ingestOutput = document.getElementById("ingest-output");
const answerOutput = document.getElementById("answer-output");
const stepsOutput = document.getElementById("steps-output");
const citationsOutput = document.getElementById("citations-output");
const questionInput = document.getElementById("question");
const imageInput = document.getElementById("image-input");
const imagePreview = document.getElementById("image-preview");

let attachedImages = [];

async function loadHealth() {
  try {
    const response = await fetch("/health");
    const payload = await response.json();
    const servicesReady = payload.ollama_reachable && payload.qdrant_reachable;
    healthPill.textContent = servicesReady ? "Ollama + Qdrant ready" : "Service check incomplete";
    healthPill.className = servicesReady ? "status-pill ok" : "status-pill warn";
    modelPill.textContent = `Recommended text model: ${payload.recommended_model}`;
    const collections = (payload.indexed_collections || []).join(", ") || "none yet";
    knowledgeDir.textContent = `Knowledge directory: ${payload.knowledge_dir} | units: ${payload.total_units} | collections: ${collections}`;
  } catch (error) {
    healthPill.textContent = "Health check failed";
    healthPill.className = "status-pill warn";
    modelPill.textContent = String(error);
  }
}

async function ingestKnowledge() {
  ingestOutput.textContent = "Rebuilding the local index...";
  try {
    const response = await fetch("/ingest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        force_full_rebuild: true,
        rebuild_scope: "core",
        include_silver: true,
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Indexing failed");
    }
    ingestOutput.textContent = JSON.stringify(payload, null, 2);
  } catch (error) {
    ingestOutput.textContent = `Indexing failed: ${error}`;
  }
}

function renderList(node, items, renderer) {
  node.innerHTML = "";
  items.forEach((item) => {
    const child = renderer(item);
    node.appendChild(child);
  });
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = String(reader.result || "");
      const base64 = result.includes(",") ? result.split(",")[1] : result;
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

async function syncAttachedImages() {
  const files = Array.from(imageInput.files || []);
  attachedImages = await Promise.all(
    files.map(async (file) => ({
      name: file.name,
      content: await fileToBase64(file),
      previewUrl: URL.createObjectURL(file),
    }))
  );
  imagePreview.innerHTML = "";
  attachedImages.forEach((image) => {
    const wrapper = document.createElement("div");
    wrapper.className = "thumb";
    const img = document.createElement("img");
    img.src = image.previewUrl;
    img.alt = image.name;
    const label = document.createElement("span");
    label.textContent = image.name;
    wrapper.appendChild(img);
    wrapper.appendChild(label);
    imagePreview.appendChild(wrapper);
  });
}

async function askQuestion() {
  const message = questionInput.value.trim();
  if (!message) return;
  answerOutput.textContent = "Qwen is reading your material and preparing an answer...";
  stepsOutput.innerHTML = "";
  citationsOutput.innerHTML = "";

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        session_id: "web",
        preferred_language: "english",
        mode: "learning",
        images: attachedImages.map((image) => image.content),
        image_names: attachedImages.map((image) => image.name),
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Chat request failed");
    }

    answerOutput.textContent = payload.answer;
    renderList(stepsOutput, payload.steps, (step) => {
      const li = document.createElement("li");
      li.textContent = step;
      return li;
    });
    renderList(citationsOutput, payload.citations, (citation) => {
      const li = document.createElement("li");
      const location = citation.page_or_slide ? ` page/slide ${citation.page_or_slide}` : "";
      const problemTag = citation.problem_id ? ` | problem ${citation.problem_id}` : "";
      li.textContent = `${citation.source_family}${problemTag} | ${citation.source_path}${location} | score=${citation.score}`;
      return li;
    });
    const verificationText = payload.verification_used ? "verification used" : "no verification layer";
    modelPill.textContent = `Model used: ${payload.model_name} | confidence: ${payload.confidence} | ${verificationText}`;
  } catch (error) {
    answerOutput.textContent = `Chat request failed: ${error}`;
  }
}

document.getElementById("refresh-health").addEventListener("click", loadHealth);
document.getElementById("run-ingest").addEventListener("click", ingestKnowledge);
document.getElementById("ask-button").addEventListener("click", askQuestion);
imageInput.addEventListener("change", syncAttachedImages);

loadHealth();
