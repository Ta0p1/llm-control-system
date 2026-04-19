const STORAGE_KEY = "control-assistant-session-id";

const healthPill = document.getElementById("health-pill");
const modelPill = document.getElementById("model-pill");
const knowledgeDir = document.getElementById("knowledge-dir");
const ingestOutput = document.getElementById("ingest-output");
const stepsOutput = document.getElementById("steps-output");
const citationsOutput = document.getElementById("citations-output");
const timingOutput = document.getElementById("timing-output");
const transcript = document.getElementById("transcript");
const sessionLabel = document.getElementById("session-label");
const devPanel = document.getElementById("dev-panel");
const devOverlay = document.getElementById("dev-overlay");

let currentSessionId = loadOrCreateSessionId();
let currentMessages = [];
let activePromptText = "";
let attachedImages = [];
let pendingAssistantId = null;
let shouldFocusPrompt = true;

function loadOrCreateSessionId() {
  const existing = localStorage.getItem(STORAGE_KEY);
  if (existing) return existing;
  const created = `web-${Date.now()}`;
  localStorage.setItem(STORAGE_KEY, created);
  return created;
}

function updateSessionLabel() {
  sessionLabel.textContent = currentSessionId;
}

function resetComposerState() {
  activePromptText = "";
  attachedImages = [];
  pendingAssistantId = null;
  shouldFocusPrompt = true;
}

function switchToNewSession() {
  currentSessionId = `web-${Date.now()}`;
  localStorage.setItem(STORAGE_KEY, currentSessionId);
  currentMessages = [];
  resetComposerState();
  renderTranscript();
  updateSessionLabel();
  timingOutput.textContent = "Timing data will appear after you ask a question.";
  stepsOutput.innerHTML = "";
  citationsOutput.innerHTML = "";
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function renderPlainInline(text) {
  let html = escapeHtml(text);
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\[([^\]]+)\]/g, "<span class=\"source-marker\">[$1]</span>");
  return html;
}

function renderKatexExpression(expression, displayMode = false) {
  const raw = String(expression || "").trim();
  if (!raw) return "";
  try {
    if (window.katex && typeof window.katex.renderToString === "function") {
      return window.katex.renderToString(raw, {
        throwOnError: false,
        displayMode,
        output: "html",
        strict: "ignore",
      });
    }
  } catch (error) {
    console.warn("KaTeX render failed", error);
  }
  const fallbackClass = displayMode ? "math-block-fallback" : "math-inline-fallback";
  return `<code class="${fallbackClass}">${escapeHtml(raw)}</code>`;
}

function renderInlineMarkdown(text) {
  const source = String(text || "");
  const mathPattern = /\$([^$\n]+)\$/g;
  let html = "";
  let lastIndex = 0;
  let match;
  while ((match = mathPattern.exec(source)) !== null) {
    html += renderPlainInline(source.slice(lastIndex, match.index));
    html += renderKatexExpression(match[1], false);
    lastIndex = match.index + match[0].length;
  }
  html += renderPlainInline(source.slice(lastIndex));
  return html;
}

function renderMarkdown(markdown) {
  const lines = String(markdown || "").replace(/\r\n/g, "\n").split("\n");
  const html = [];
  let inUl = false;
  let inOl = false;
  let inCode = false;
  let inMathBlock = false;
  let mathBuffer = [];
  let paragraph = [];

  function flushParagraph() {
    if (!paragraph.length) return;
    html.push(`<p>${renderInlineMarkdown(paragraph.join(" "))}</p>`);
    paragraph = [];
  }

  function closeLists() {
    if (inUl) {
      html.push("</ul>");
      inUl = false;
    }
    if (inOl) {
      html.push("</ol>");
      inOl = false;
    }
  }

  function flushMathBlock() {
    if (!mathBuffer.length) return;
    html.push(`<div class="math-block">${renderKatexExpression(mathBuffer.join("\n").trim(), true)}</div>`);
    mathBuffer = [];
  }

  for (const rawLine of lines) {
    const line = rawLine.trimEnd();
    const trimmed = line.trim();

    if (trimmed === "$$") {
      flushParagraph();
      closeLists();
      if (!inMathBlock) {
        inMathBlock = true;
        mathBuffer = [];
      } else {
        flushMathBlock();
        inMathBlock = false;
      }
      continue;
    }

    if (inMathBlock) {
      mathBuffer.push(rawLine);
      continue;
    }

    if (/^\$\$.+\$\$$/.test(trimmed)) {
      flushParagraph();
      closeLists();
      html.push(`<div class="math-block">${renderKatexExpression(trimmed.slice(2, -2).trim(), true)}</div>`);
      continue;
    }

    if (trimmed.startsWith("```")) {
      flushParagraph();
      closeLists();
      if (!inCode) {
        html.push("<pre><code>");
        inCode = true;
      } else {
        html.push("</code></pre>");
        inCode = false;
      }
      continue;
    }

    if (inCode) {
      html.push(`${escapeHtml(rawLine)}\n`);
      continue;
    }

    if (!trimmed) {
      flushParagraph();
      closeLists();
      continue;
    }

    const headingMatch = trimmed.match(/^(#{1,3})\s+(.+)$/);
    if (headingMatch) {
      flushParagraph();
      closeLists();
      const level = Math.min(headingMatch[1].length, 4);
      html.push(`<h${level}>${renderInlineMarkdown(headingMatch[2])}</h${level}>`);
      continue;
    }

    const ulMatch = trimmed.match(/^[-*]\s+(.+)$/);
    if (ulMatch) {
      flushParagraph();
      if (inOl) {
        html.push("</ol>");
        inOl = false;
      }
      if (!inUl) {
        html.push("<ul>");
        inUl = true;
      }
      html.push(`<li>${renderInlineMarkdown(ulMatch[1])}</li>`);
      continue;
    }

    const olMatch = trimmed.match(/^\d+\.\s+(.+)$/);
    if (olMatch) {
      flushParagraph();
      if (inUl) {
        html.push("</ul>");
        inUl = false;
      }
      if (!inOl) {
        html.push("<ol>");
        inOl = true;
      }
      html.push(`<li>${renderInlineMarkdown(olMatch[1])}</li>`);
      continue;
    }

    closeLists();
    paragraph.push(trimmed);
  }

  flushParagraph();
  closeLists();
  flushMathBlock();
  if (inCode) {
    html.push("</code></pre>");
  }
  return html.join("\n");
}

function autoResizePrompt(textarea) {
  const maxHeight = 180;
  textarea.style.height = "0px";
  textarea.style.height = `${Math.min(textarea.scrollHeight, maxHeight)}px`;
  textarea.style.overflowY = textarea.scrollHeight > maxHeight ? "auto" : "hidden";
}

function renderAttachmentSummary(container) {
  container.innerHTML = "";
  attachedImages.forEach((image) => {
    const label = document.createElement("span");
    label.className = "attachment-chip";
    label.textContent = image.name;
    container.appendChild(label);
  });
}

function buildActivePromptNode() {
  const promptRow = document.createElement("article");
  promptRow.className = "message-row prompt-row";

  const prefix = document.createElement("div");
  prefix.className = "message-prefix active-prefix";
  prefix.textContent = ">>";

  const body = document.createElement("div");
  body.className = "message-body prompt-body";

  const editor = document.createElement("div");
  editor.className = "prompt-editor";

  const mainRow = document.createElement("div");
  mainRow.className = "prompt-main";

  const textarea = document.createElement("textarea");
  textarea.className = "prompt-textarea";
  textarea.rows = 1;
  textarea.placeholder = "Describe the control problem, diagram, or formula you want help with.";
  textarea.value = activePromptText;
  textarea.addEventListener("input", (event) => {
    activePromptText = event.target.value;
    autoResizePrompt(textarea);
  });
  textarea.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      askQuestion();
    }
  });

  const controls = document.createElement("div");
  controls.className = "prompt-controls";

  const fileLabel = document.createElement("label");
  fileLabel.className = "file-trigger";
  fileLabel.textContent = "Attach images";

  const fileInput = document.createElement("input");
  fileInput.type = "file";
  fileInput.accept = "image/*";
  fileInput.multiple = true;
  fileInput.hidden = true;
  fileInput.addEventListener("change", async (event) => {
    const files = Array.from(event.target.files || []);
    attachedImages = await Promise.all(
      files.map(async (file) => ({
        name: file.name,
        content: await fileToBase64(file),
        previewUrl: URL.createObjectURL(file),
      }))
    );
    renderTranscript();
  });
  fileLabel.appendChild(fileInput);

  const hint = document.createElement("span");
  hint.className = "tool-hint";
  hint.textContent = "Enter to run";

  const sendButton = document.createElement("button");
  sendButton.type = "button";
  sendButton.className = "send-button";
  sendButton.textContent = "Run";
  sendButton.addEventListener("click", askQuestion);

  controls.appendChild(fileLabel);
  controls.appendChild(hint);
  controls.appendChild(sendButton);

  const preview = document.createElement("div");
  preview.className = "attachment-strip";
  renderAttachmentSummary(preview);

  mainRow.appendChild(textarea);
  mainRow.appendChild(controls);
  editor.appendChild(mainRow);
  editor.appendChild(preview);
  body.appendChild(editor);
  promptRow.appendChild(prefix);
  promptRow.appendChild(body);

  if (shouldFocusPrompt) {
    queueMicrotask(() => {
      autoResizePrompt(textarea);
      textarea.focus();
      textarea.setSelectionRange(textarea.value.length, textarea.value.length);
      shouldFocusPrompt = false;
    });
  } else {
    queueMicrotask(() => autoResizePrompt(textarea));
  }

  return promptRow;
}

function renderTranscript() {
  transcript.innerHTML = "";

  if (!currentMessages.length) {
    const emptyState = document.createElement("div");
    emptyState.className = "transcript-empty";
    emptyState.innerHTML = `<p class="prompt-line">&gt;&gt; Ask your first question.</p>`;
    transcript.appendChild(emptyState);
  } else {
    currentMessages.forEach((message) => {
      const row = document.createElement("article");
      row.className = `message-row ${message.role}`;
      if (message.pending) {
        row.classList.add("pending");
      }

      const prefix = document.createElement("div");
      prefix.className = "message-prefix";
      prefix.textContent = message.role === "user" ? ">>" : "";

      const body = document.createElement("div");
      body.className = `message-body ${message.role === "assistant" ? "markdown-like" : "plain-user"}`;
      if (message.role === "assistant") {
        body.innerHTML = renderMarkdown(message.content || "");
      } else {
        body.textContent = message.content;
      }

      row.appendChild(prefix);
      row.appendChild(body);
      transcript.appendChild(row);
    });
  }

  if (!pendingAssistantId) {
    transcript.appendChild(buildActivePromptNode());
  }
  transcript.scrollTop = transcript.scrollHeight;
}

function renderList(node, items, renderer) {
  node.innerHTML = "";
  items.forEach((item) => {
    const child = renderer(item);
    node.appendChild(child);
  });
}

async function loadHealth() {
  try {
    const response = await fetch("/health");
    const payload = await response.json();
    const servicesReady = payload.ollama_reachable && payload.qdrant_reachable;
    healthPill.textContent = servicesReady ? "Ollama + Qdrant ready" : "Service check incomplete";
    healthPill.className = servicesReady ? "status-pill ok" : "status-pill warn";
    modelPill.textContent = `Active runtime model: ${payload.recommended_model}`;
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
    await loadHealth();
  } catch (error) {
    ingestOutput.textContent = `Indexing failed: ${error}`;
  }
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

function formatTiming(timing) {
  const total = timing.total_duration_ms ?? 0;
  const stages = timing.stage_timings || {};
  const modelCalls = timing.model_calls || {};
  const metadata = timing.metadata || {};
  const stageEntries = Object.entries(stages);
  const slowest = stageEntries.sort((a, b) => b[1] - a[1])[0];
  const stageLines = Object.entries(stages).map(([name, ms]) => `${name}: ${ms} ms`);
  const modelLines = Object.entries(modelCalls).map(([name, ms]) => `${name}: ${ms} ms`);
  const metaLines = [
    `path: ${metadata.path_type ?? "unknown"}`,
    `images: ${metadata.image_count ?? 0}`,
    `primary hits: ${metadata.primary_hit_count ?? 0}`,
    `verification hits: ${metadata.verification_hit_count ?? 0}`,
    `tool results: ${metadata.tool_count ?? 0}`,
    `review used: ${metadata.review_used ?? false}`,
    `compressed context chars: ${metadata.compressed_context_chars ?? 0}`,
    `compose answer empty: ${metadata.compose_answer_empty ?? false}`,
  ];
  return [
    `total: ${total} ms`,
    slowest ? `slowest stage: ${slowest[0]} (${slowest[1]} ms)` : "slowest stage: n/a",
    "",
    "stages:",
    ...(stageLines.length ? stageLines : ["none"]),
    "",
    "model calls:",
    ...(modelLines.length ? modelLines : ["none"]),
    "",
    "metadata:",
    ...metaLines,
  ].join("\n");
}

function updateDeveloperPanel(payload) {
  renderList(stepsOutput, payload.steps || [], (step) => {
    const li = document.createElement("li");
    li.textContent = step;
    return li;
  });
  renderList(citationsOutput, payload.citations || [], (citation) => {
    const li = document.createElement("li");
    const location = citation.page_or_slide ? ` page/slide ${citation.page_or_slide}` : "";
    const problemTag = citation.problem_id ? ` | problem ${citation.problem_id}` : "";
    li.textContent = `${citation.source_family}${problemTag} | ${citation.source_path}${location} | score=${citation.score}`;
    return li;
  });
  const verificationText = payload.verification_used ? "verification used" : "no verification layer";
  modelPill.textContent = `Model used: ${payload.model_name} | confidence: ${payload.confidence} | ${verificationText}`;
  timingOutput.textContent = formatTiming(payload.timing || {});
}

function upsertPendingAssistant(content) {
  if (!pendingAssistantId) return;
  currentMessages = currentMessages.map((message) =>
    message.id === pendingAssistantId ? { ...message, content, pending: false } : message
  );
  pendingAssistantId = null;
  shouldFocusPrompt = true;
  renderTranscript();
}

async function askQuestion() {
  const message = activePromptText.trim();
  if (!message) return;

  const imagesToSend = [...attachedImages];
  const imageMarker = imagesToSend.length
    ? `\n[Attached images: ${imagesToSend.map((image) => image.name).join(", ")}]`
    : "";

  const userMessage = {
    id: `user-${Date.now()}`,
    role: "user",
    content: `${message}${imageMarker}`,
  };
  pendingAssistantId = `assistant-${Date.now()}`;
  const assistantPlaceholder = {
    id: pendingAssistantId,
    role: "assistant",
    content: "_Computing result..._",
    pending: true,
  };

  currentMessages = [...currentMessages, userMessage, assistantPlaceholder];
  activePromptText = "";
  attachedImages = [];
  renderTranscript();

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        session_id: currentSessionId,
        preferred_language: "english",
        mode: "learning",
        images: imagesToSend.map((image) => image.content),
        image_names: imagesToSend.map((image) => image.name),
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Chat request failed");
    }
    upsertPendingAssistant(payload.answer);
    updateDeveloperPanel(payload);
    await loadSessionHistory();
  } catch (error) {
    upsertPendingAssistant(`## Request Failed\n\n${String(error)}`);
    timingOutput.textContent = "No timing data available because the request failed.";
  }
}

async function loadSessionHistory() {
  try {
    const response = await fetch(`/session/${encodeURIComponent(currentSessionId)}/history`);
    const payload = await response.json();
    currentMessages = (payload.messages || []).map((message, index) => ({
      id: `${message.role}-${index}-${message.created_at}`,
      role: message.role,
      content: message.content,
    }));
    shouldFocusPrompt = true;
    renderTranscript();
  } catch (error) {
    console.error("Failed to load session history", error);
  }
}

function toggleDevPanel(forceOpen) {
  const shouldOpen = typeof forceOpen === "boolean" ? forceOpen : !devPanel.classList.contains("open");
  devPanel.classList.toggle("open", shouldOpen);
  devOverlay.classList.toggle("hidden", !shouldOpen);
  document.getElementById("toggle-dev-panel").setAttribute("aria-expanded", String(shouldOpen));
  devPanel.setAttribute("aria-hidden", String(!shouldOpen));
}

document.getElementById("refresh-health").addEventListener("click", loadHealth);
document.getElementById("run-ingest").addEventListener("click", ingestKnowledge);
document.getElementById("new-session").addEventListener("click", switchToNewSession);
document.getElementById("toggle-dev-panel").addEventListener("click", () => toggleDevPanel());
document.getElementById("close-dev-panel").addEventListener("click", () => toggleDevPanel(false));
devOverlay.addEventListener("click", () => toggleDevPanel(false));

document.addEventListener("keydown", (event) => {
  if (event.ctrlKey && event.shiftKey && event.key.toLowerCase() === "d") {
    event.preventDefault();
    toggleDevPanel();
  }
});

updateSessionLabel();
loadHealth();
loadSessionHistory();
