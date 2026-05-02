const API_BASE = "http://127.0.0.1:8000";
const HISTORY_KEY = "dut_chat_history_v1";

const chatMessages = document.getElementById("chatMessages");
const historyList = document.getElementById("historyList");
const statusBar = document.getElementById("statusBar");
const modelSelect = document.getElementById("modelSelect");
const topKSelect = document.getElementById("topKSelect");
const queryInput = document.getElementById("queryInput");
const chatForm = document.getElementById("chatForm");
const planBtn = document.getElementById("planBtn");
const clearHistoryBtn = document.getElementById("clearHistoryBtn");
const messageTemplate = document.getElementById("messageTemplate");

let sessions = loadHistory();
let activeSessionId = sessions[0]?.id || createSession();

function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
  } catch {
    return [];
  }
}

function saveHistory() {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(sessions));
}

function createSession() {
  const id = `session_${Date.now()}`;
  sessions.unshift({
    id,
    title: "Cuộc trò chuyện mới",
    createdAt: new Date().toISOString(),
    messages: [],
  });
  saveHistory();
  return id;
}

function getActiveSession() {
  return sessions.find((item) => item.id === activeSessionId);
}

function formatTime(value) {
  return new Date(value).toLocaleString("vi-VN", {
    hour: "2-digit",
    minute: "2-digit",
    day: "2-digit",
    month: "2-digit",
  });
}

function setStatus(text, isError = false) {
  statusBar.textContent = text;
  statusBar.classList.toggle("error", isError);
}

function renderHistory() {
  historyList.innerHTML = "";
  if (!sessions.length) {
    historyList.innerHTML = '<div class="empty-state">Hiện chưa có cuộc hội thoại nào.</div>';
    return;
  }

  for (const session of sessions) {
    const item = document.createElement("button");
    item.type = "button";
    item.className = "history-item";
    item.innerHTML = `
      <div class="history-title">${escapeHtml(session.title)}</div>
      <div class="history-time">${formatTime(session.createdAt)}</div>
    `;
    item.addEventListener("click", () => {
      activeSessionId = session.id;
      renderMessages();
    });
    historyList.appendChild(item);
  }
}

function renderMessages() {
  chatMessages.innerHTML = "";
  const session = getActiveSession();
  if (!session || !session.messages.length) {
    addMessage({
      role: "bot",
      meta: "DUT Chatbot",
      text: "Xin chào! Đây là DUT Chatbot, trợ lý hỏi đáp cho dữ liệu sinh viên và quy định học vụ. Bạn có thể chọn model rồi hỏi trực tiếp ở đây.",
    });
    return;
  }

  for (const msg of session.messages) {
    addMessage(msg);
  }
}

function addMessage({ role, meta, text, extra }) {
  const node = messageTemplate.content.firstElementChild.cloneNode(true);
  node.classList.add(role);
  node.querySelector(".bubble-meta").textContent = meta || "";
  node.querySelector(".bubble").textContent = text || "";
  const extraNode = node.querySelector(".bubble-extra");
  if (extra) {
    extraNode.textContent = extra;
    extraNode.classList.remove("hidden");
  }
  chatMessages.appendChild(node);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function appendToSession(message) {
  const session = getActiveSession();
  if (!session) {
    return;
  }
  session.messages.push(message);
  if (session.title === "Cuộc trò chuyện mới" && message.role === "user") {
    session.title = message.text.slice(0, 42) || session.title;
  }
  saveHistory();
  renderHistory();
}

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

async function fetchJson(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }
  return response.json();
}

async function loadModels() {
  try {
    const health = await fetchJson("/health");
    const models = await fetchJson("/models");
    modelSelect.innerHTML = "";
    for (const name of models.models || []) {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      if (name === models.default_model) {
        option.selected = true;
      }
      modelSelect.appendChild(option);
    }
    setStatus(
      `Backend sẵn sàng. Neo4j: ${health.graph_enabled ? "bật" : "tắt"} | Reranker: ${health.reranker_enabled ? "bật" : "tắt"} | Provider: ${models.provider}`,
    );
  } catch (error) {
    setStatus(`Không kết nối được backend: ${error.message}`, true);
  }
}

async function sendQuery(path) {
  const query = queryInput.value.trim();
  if (!query) {
    return;
  }

  const model = modelSelect.value;
  const top_k = Number(topKSelect.value);
  const userMessage = {
    role: "user",
    meta: `Bạn | ${model}`,
    text: query,
  };
  appendToSession(userMessage);
  renderMessages();
  queryInput.value = "";
  queryInput.style.height = "auto";
  setStatus(path === "/plan" ? "Đang phân tích route..." : "Đang truy vấn backend...");

  try {
    const data = await fetchJson(path, {
      method: "POST",
      body: JSON.stringify({ query, model, top_k }),
    });

    const meta = path === "/plan" ? `Planner | ${model}` : `DUT Chatbot | ${model}`;
    const route = data.plan?.route || "unknown";
    const intents = Array.isArray(data.plan?.intents) ? data.plan.intents.join(", ") : "unknown";
    const text =
      path === "/plan"
        ? `Route: ${route}\nIntents: ${intents}\nEntities: ${JSON.stringify(data.entities)}`
        : data.answer;
    const extra =
      path === "/plan"
        ? `Vector preview: ${JSON.stringify(data.vector_preview, null, 2)}`
        : `Route: ${route}\nIntents: ${intents}\nTop hits: ${data.vector_hits
            .slice(0, 3)
            .map((item) => `${item.ann_id || "NA"} | ${item.title || "Untitled"}`)
            .join("\n")}`;

    const botMessage = { role: "bot", meta, text, extra };
    appendToSession(botMessage);
    renderMessages();
    setStatus("Hoàn tất.");
  } catch (error) {
    const botMessage = {
      role: "bot",
      meta: `Lỗi | ${model}`,
      text: `Không truy vấn được backend.`,
      extra: error.message,
    };
    appendToSession(botMessage);
    renderMessages();
    setStatus(`Lỗi backend: ${error.message}`, true);
  }
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  await sendQuery("/query");
});

planBtn.addEventListener("click", async () => {
  await sendQuery("/plan");
});

clearHistoryBtn.addEventListener("click", () => {
  sessions = [];
  activeSessionId = createSession();
  saveHistory();
  renderHistory();
  renderMessages();
});

queryInput.addEventListener("input", () => {
  queryInput.style.height = "auto";
  queryInput.style.height = `${Math.min(queryInput.scrollHeight, 220)}px`;
});

renderHistory();
renderMessages();
loadModels();
