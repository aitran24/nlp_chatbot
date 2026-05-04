const API_BASE = "http://127.0.0.1:8000";
const HISTORY_KEY = "dut_chat_history_v1";

const chatMessages = document.getElementById("chatMessages");
const historyList = document.getElementById("historyList");
const modelSelect = document.getElementById("modelSelect");
const topKSelect = document.getElementById("topKSelect");
const queryInput = document.getElementById("queryInput");
const chatForm = document.getElementById("chatForm");
const newConvBtn = document.getElementById("newConvBtn");
const messageTemplate = document.getElementById("messageTemplate");

// Modal elements
const deleteModal = document.getElementById("deleteModal");
const cancelBtn = document.getElementById("cancelBtn");
const confirmDeleteBtn = document.getElementById("confirmDeleteBtn");
const loadingIndicator = document.getElementById("loadingIndicator");
let sessionToDelete = null;

let sessions = loadHistory();
let activeSessionId;
let currentAbortController = null;

// Nếu cuộc trò chuyện gần nhất chưa có tin nhắn từ user (chỉ có welcome message)
// thì dùng cuộc đó, không tạo mới
if (sessions.length > 0 && !hasUserMessage(sessions[0])) {
  activeSessionId = sessions[0].id;
} else {
  // Tạo cuộc trò chuyện mới nếu không có cuộc nào hoặc cuộc gần nhất đã có tin nhắn
  activeSessionId = createSession();
}

// Hủy request khi page unload/navigate
window.addEventListener('beforeunload', () => {
  if (currentAbortController) {
    currentAbortController.abort();
  }
});

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
  const welcomeMessage = {
    role: "bot",
    meta: "DUT Chatbot",
    text: "Xin chào! Đây là DUT Chatbot, trợ lý hỏi đáp cho dữ liệu sinh viên và quy định học vụ. Bạn có thể chọn model rồi hỏi trực tiếp ở đây.",
  };
  sessions.unshift({
    id,
    title: "Cuộc trò chuyện mới",
    createdAt: new Date().toISOString(),
    pinned: false,
    messages: [welcomeMessage], // Thêm tin nhắn chào mừng vào đây
  });
  saveHistory();
  return id;
}

function getActiveSession() {
  return sessions.find((item) => item.id === activeSessionId);
}

function hasUserMessage(session) {
  return !!session && session.messages.some((message) => message.role === "user");
}

function formatTime(value) {
  return new Date(value).toLocaleString("vi-VN", {
    hour: "2-digit",
    minute: "2-digit",
    day: "2-digit",
    month: "2-digit",
  });
}

function getHistorySessions() {
  return [...sessions].sort((a, b) => {
    if (a.pinned !== b.pinned) {
      return a.pinned ? -1 : 1;
    }

    return new Date(b.createdAt) - new Date(a.createdAt);
  });
}

function setStatus(text, isError = false) {
  // Status bar removed
}

// Xóa một session cụ thể
function deleteSession(id) {
  sessions = sessions.filter((s) => s.id !== id);
  if (activeSessionId === id) {
    if (sessions.length > 0) {
      activeSessionId = sessions[0].id;
    } else {
      activeSessionId = createSession();
    }
  }
  saveHistory();
  renderHistory();
  renderMessages();
}

  // Modal functions
  function showDeleteModal(sessionId) {
    sessionToDelete = sessionId;
    deleteModal.classList.remove("hidden");
  }

  function hideDeleteModal() {
    deleteModal.classList.add("hidden");
    sessionToDelete = null;
  }

  function confirmDelete() {
    if (sessionToDelete) {
      deleteSession(sessionToDelete);
      hideDeleteModal();
    }
  }

// Ghim/bỏ ghim session
function togglePin(id) {
  const session = sessions.find((s) => s.id === id);
  if (!session) return;
  session.pinned = !session.pinned;

  saveHistory();
  renderHistory();
}

function renderHistory() {
  historyList.innerHTML = "";

  if (!sessions.length) {
    historyList.innerHTML = '<div class="empty-state">Hiện chưa có cuộc hội thoại nào.</div>';
    return;
  }

  for (const session of getHistorySessions()) {
    const item = document.createElement("div");
    item.className = "history-item" + (session.id === activeSessionId ? " active" : "") + (session.pinned ? " pinned" : "");
    item.dataset.id = session.id;

    const pinIcon = session.pinned ? '<img src="../assets/star2.png" alt="pinned" class="pin-icon">' : '<img src="../assets/star1.png" alt="unpinned" class="pin-icon">';
    const deleteIcon = '<img src="../assets/trash.png" alt="delete" class="pin-icon">';
    
    item.innerHTML = `
      <div class="history-content">
        <div class="history-text">
          <div class="history-title">${escapeHtml(session.title)}</div>
          <div class="history-time">${formatTime(session.createdAt)}</div>
        </div>
        <div class="history-actions">
          <button class="history-action-btn delete-btn" data-id="${session.id}" title="Xóa" type="button">${deleteIcon}</button>
          <button class="history-action-btn pin-btn" data-id="${session.id}" title="Ghim" type="button">${pinIcon}</button>
        </div>
      </div>
    `;

    // Click vào item để mở session
    item.addEventListener("click", (e) => {
      if (e.target.closest(".history-action-btn")) return;
      activeSessionId = session.id;
      renderHistory();
      renderMessages();
    });

    // Pin button
    const pinBtn = item.querySelector(".pin-btn");
    pinBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      togglePin(session.id);
    });

    // Delete button
    const deleteBtn = item.querySelector(".delete-btn");
    deleteBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      showDeleteModal(session.id);
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
  try {
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
  } catch (err) {
    console.error("Error in addMessage:", err);
  }
}

function appendToSession(message) {
  const session = getActiveSession();
  if (!session) return;
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
      if (name === models.default_model) option.selected = true;
      modelSelect.appendChild(option);
    }
  } catch (error) {
    console.error("Backend not available:", error.message);
  }
}

async function sendQuery(path) {
  const query = queryInput.value.trim();
  if (!query) return;

  // Hủy request cũ nếu có
  if (currentAbortController) {
    currentAbortController.abort();
  }
  currentAbortController = new AbortController();

  const model = modelSelect.value;
  const top_k = Number(topKSelect.value);
  const userMessage = { role: "user", meta: ``, text: query };
  appendToSession(userMessage);
  renderMessages();
  queryInput.value = "";
  queryInput.style.height = "40px";

  // Show loading indicator
  loadingIndicator.classList.remove("hidden");
  chatMessages.appendChild(loadingIndicator);
  chatMessages.scrollTop = chatMessages.scrollHeight;

  try {
    const data = await fetchJson(path, {
      method: "POST",
      body: JSON.stringify({ query, model, top_k }),
      signal: currentAbortController.signal,
    });

    const meta = path === "/plan" ? `Planner | ${model}` : `DUT Chatbot | ${model}`;
    const route = data.plan?.route || "unknown";
    const intents = Array.isArray(data.plan?.intents) ? data.plan.intents.join(", ") : "unknown";
    const text =
      path === "/plan"
        ? `Route: ${route}\nIntents: ${intents}\nEntities: ${JSON.stringify(data.entities)}`
        : data.answer;
    const extra = path === "/plan"
        ? `Vector preview: ${JSON.stringify(data.vector_preview, null, 2)}`
        : `Route: ${route}\nIntents: ${intents}\nTop hits: ${data.vector_hits
            .slice(0, 3)
            .map((item) => `${item.ann_id || "NA"} | ${item.title || "Untitled"}`)
            .join("\n")}`;

    const botMessage = { role: "bot", meta, text, extra };
    appendToSession(botMessage);
    renderMessages();
  } catch (error) {
    // Bỏ qua lỗi nếu request bị hủy (user navigate)
    if (error.name === "AbortError") {
      console.log("Request cancelled - user navigated");
      return;
    }

    const botMessage = {
      role: "bot",
      meta: `Lỗi | ${model}`,
      text: `Không truy vấn được backend.`,
      extra: error.message,
    };
    appendToSession(botMessage);
    renderMessages();
  } finally {
    // Hide loading indicator
    loadingIndicator.classList.add("hidden");
  }
}

// ===== EVENT LISTENERS =====

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  await sendQuery("/query");
});

newConvBtn.addEventListener("click", () => {
  // Kiểm tra cuộc trò chuyện gần nhất (chứ không phải cuộc hiện tại)
  const recentSession = sessions[0];

  if (recentSession && !hasUserMessage(recentSession)) {
    // Nếu cuộc gần nhất chưa có user message → chuyển sang cuộc đó
    activeSessionId = recentSession.id;
    queryInput.value = "";
    queryInput.style.height = "40px";
    renderHistory();
    renderMessages();
    queryInput.focus();
    return;
  }

  // Nếu cuộc gần nhất đã có user message → tạo mới
  activeSessionId = createSession();
  queryInput.value = "";
  queryInput.style.height = "40px";
  saveHistory();
  renderHistory();
  renderMessages();
  queryInput.focus();
});

// Modal event listeners
cancelBtn.addEventListener("click", hideDeleteModal);
confirmDeleteBtn.addEventListener("click", confirmDelete);
deleteModal.addEventListener("click", (e) => {
  if (e.target === deleteModal) {
    hideDeleteModal();
  }
});

// Auto-resize textarea
queryInput.addEventListener("input", () => {
  queryInput.style.height = "40px";
  queryInput.style.height = `${Math.min(queryInput.scrollHeight, 150)}px`;
});

// Gửi bằng Enter (Shift+Enter = xuống dòng)
queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    chatForm.dispatchEvent(new Event("submit"));
  }
});

// ===== KHỞI ĐỘNG =====
renderHistory();
renderMessages();
loadModels().catch(console.error);