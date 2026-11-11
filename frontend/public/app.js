// frontend/public/app.js
// NovaCart Essentials - Customer Support RAG Client Engine
// Connects UI to FastAPI backend with real-time Redis telemetry, hybrid search, and cache invalidation.

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const chatFeed = document.getElementById("chatFeed");
  const welcomeBox = document.getElementById("welcomeBox");
  const chatInput = document.getElementById("chatInput");
  const sendBtn = document.getElementById("sendBtn");
  const clearChatBtn = document.getElementById("clearChatBtn");
  const testRepeatBtn = document.getElementById("testRepeatBtn");
  const simulateUpdateBtn = document.getElementById("simulateUpdateBtn");
  const clearCacheBtn = document.getElementById("clearCacheBtn");
  const toastHub = document.getElementById("toastHub");

  // Evidence Modal Elements
  const evidenceModal = document.getElementById("evidenceModal");
  const modalCloseBtn = document.getElementById("modalCloseBtn");
  const modalDocTitle = document.getElementById("modalDocTitle");
  const modalPageTag = document.getElementById("modalPageTag");
  const modalChunkTag = document.getElementById("modalChunkTag");
  const modalPdfLink = document.getElementById("modalPdfLink");
  const modalPassageText = document.getElementById("modalPassageText");

  function openEvidenceModal(src) {
    if (!evidenceModal) return;
    if (modalDocTitle) modalDocTitle.textContent = src.title || "Policy Document";
    if (modalPageTag) modalPageTag.textContent = `Page ${src.page_number || 1} · ${src.section || "General"}`;
    if (modalChunkTag) modalChunkTag.textContent = src.chunk_id || "";
    if (modalPdfLink) {
      const url = src.source_url || "";
      const isUsableUrl = url && (url.startsWith("http://") || url.startsWith("https://"));
      if (isUsableUrl) {
        modalPdfLink.href = url;
        modalPdfLink.textContent = `📄 View PDF Page ${src.page_number || 1} →`;
        modalPdfLink.style.display = "inline-flex";
      } else {
        modalPdfLink.style.display = "none";
      }
    }
    if (modalPassageText) modalPassageText.textContent = src.supporting_passage || src.text_snippet || "Supporting policy excerpt.";
    evidenceModal.style.display = "flex";
  }

  function closeEvidenceModal() {
    if (evidenceModal) evidenceModal.style.display = "none";
  }

  if (modalCloseBtn) modalCloseBtn.addEventListener("click", closeEvidenceModal);
  if (evidenceModal) {
    evidenceModal.addEventListener("click", (e) => {
      if (e.target === evidenceModal) closeEvidenceModal();
    });
  }

  // Dashboard Telemetry Elements
  const dashHitRate = document.getElementById("dashHitRate");
  const dashHitBar = document.getElementById("dashHitBar");
  const dashTotalQuestions = document.getElementById("dashTotalQuestions");
  const dashHits = document.getElementById("dashHits");
  const dashMisses = document.getElementById("dashMisses");
  const dashAvoidedLLM = document.getElementById("dashAvoidedLLM");
  const dashHitLatency = document.getElementById("dashHitLatency");
  const dashMissLatency = document.getElementById("dashMissLatency");
  const dashCostSaved = document.getElementById("dashCostSaved");
  const dashKbVersion = document.getElementById("dashKbVersion");
  const dashChunksCount = document.getElementById("dashChunksCount");
  const policiesList = document.getElementById("policiesList");
  const redisMode = document.getElementById("redisMode");

  let isSending = false;
  let policyExtended = false;

  // --------------------------------------------------------------------------
  // Utility & Helper Functions
  // --------------------------------------------------------------------------
  function escapeHtml(text) {
    if (!text) return "";
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function formatTime() {
    const now = new Date();
    return now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }

  function showToast(message, type = "info") {
    if (!toastHub) return;
    const toast = document.createElement("div");
    toast.className = `toast-msg toast-${type}`;
    toast.textContent = message;
    toastHub.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateY(8px)";
      toast.style.transition = "all 0.3s ease";
      setTimeout(() => toast.remove(), 300);
    }, 3800);
  }

  function scrollToBottom() {
    chatFeed.scrollTo({
      top: chatFeed.scrollHeight,
      behavior: "smooth",
    });
  }

  // --------------------------------------------------------------------------
  // Telemetry & Knowledge Base Sync
  // --------------------------------------------------------------------------
  async function refreshTelemetry() {
    try {
      const res = await fetch("/api/metrics");
      if (!res.ok) return;
      const data = await res.json();

      if (dashTotalQuestions) dashTotalQuestions.textContent = data.total_questions;
      if (dashHits) dashHits.textContent = data.cache_hits;
      if (dashMisses) dashMisses.textContent = data.cache_misses;
      if (dashAvoidedLLM) dashAvoidedLLM.textContent = data.llm_calls_avoided;

      const hitRate = Number(data.hit_rate_pct || 0);
      if (dashHitRate) dashHitRate.textContent = `${hitRate.toFixed(1)}% Hit Rate`;
      if (dashHitBar) dashHitBar.style.width = `${Math.min(100, Math.max(0, hitRate))}%`;

      if (dashHitLatency) dashHitLatency.textContent = `${Number(data.avg_hit_latency_ms || 0).toFixed(1)} ms`;
      if (dashMissLatency) dashMissLatency.textContent = `${Number(data.avg_miss_latency_ms || 0).toFixed(1)} ms`;
      if (dashCostSaved) dashCostSaved.textContent = `$${Number(data.estimated_cost_saved_usd || 0).toFixed(4)}`;
    } catch (err) {
      console.warn("Telemetry refresh error:", err);
    }
  }

  async function refreshKnowledgeBase() {
    try {
      const res = await fetch("/api/kb");
      if (!res.ok) return;
      const data = await res.json();

      if (dashKbVersion) {
        const shortHash = (data.version_hash || "v1.0.0").substring(0, 8);
        dashKbVersion.textContent = `KB: ${shortHash}`;
        dashKbVersion.title = `Full KB SHA-256: ${data.version_hash}`;
      }

      if (dashChunksCount) {
        dashChunksCount.textContent = `${data.total_chunks || 0} chunks`;
      }

      if (policiesList && data.documents) {
        policiesList.innerHTML = "";
        data.documents.forEach((doc) => {
          const item = document.createElement("div");
          item.className = "policy-item";
          item.innerHTML = `
            <div style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:180px;">
              <span style="font-weight:600; color:var(--text-main); font-size:0.75rem;">${escapeHtml(doc.title)}</span>
            </div>
            <span class="chunk-badge" style="font-size:0.62rem;">${escapeHtml(doc.source_ref)}</span>
          `;
          policiesList.appendChild(item);
        });
      }
    } catch (err) {
      console.warn("Knowledge base sync error:", err);
    }
  }

  async function checkHealth() {
    try {
      const res = await fetch("/api/health");
      if (!res.ok) return;
      const data = await res.json();
      if (redisMode) {
        const eviction = data.eviction_policy || "allkeys-lfu";
        if (data.is_emulator) {
          redisMode.textContent = `In-Memory (${eviction})`;
          redisMode.style.color = "#38BDF8";
        } else {
          redisMode.textContent = `Redis 6379 (${eviction})`;
          redisMode.style.color = "#34D399";
        }
      }
    } catch (err) {
      console.warn("Health check error:", err);
    }
  }

  // --------------------------------------------------------------------------
  // Message Rendering
  // --------------------------------------------------------------------------
  function appendUserMessage(text) {
    if (welcomeBox) welcomeBox.style.display = "none";

    const row = document.createElement("div");
    row.className = "message-row user-row";
    row.innerHTML = `
      <div class="msg-bubble user-bubble">
        ${escapeHtml(text)}
      </div>
    `;
    // Attach citation inspection click handlers
    row.querySelectorAll(".source-badge-clickable").forEach((badge) => {
      badge.addEventListener("click", () => {
        const idx = parseInt(badge.getAttribute("data-idx"), 10);
        const srcObj = activeSources[idx];
        if (srcObj) openEvidenceModal(srcObj);
      });
    });

    chatFeed.appendChild(row);
    scrollToBottom();
  }

  function showLoadingIndicator() {
    const row = document.createElement("div");
    row.className = "message-row";
    row.id = "thinkingIndicator";
    row.innerHTML = `
      <div class="agent-avatar" style="width:32px;height:32px;flex-shrink:0;">AI</div>
      <div class="msg-bubble assistant-bubble" style="padding: 12px 18px;">
        <div class="soundwave-wrap">
          <div class="soundwave">
            <div class="wave-line"></div>
            <div class="wave-line"></div>
            <div class="wave-line"></div>
          </div>
          <span style="font-size:0.8rem;color:var(--text-muted);font-weight:500;">Searching knowledge base &amp; reasoning...</span>
        </div>
      </div>
    `;
    // Attach citation inspection click handlers
    row.querySelectorAll(".source-badge-clickable").forEach((badge) => {
      badge.addEventListener("click", () => {
        const idx = parseInt(badge.getAttribute("data-idx"), 10);
        const srcObj = activeSources[idx];
        if (srcObj) openEvidenceModal(srcObj);
      });
    });

    chatFeed.appendChild(row);
    scrollToBottom();
    return row;
  }

  function removeLoadingIndicator() {
    const indicator = document.getElementById("thinkingIndicator");
    if (indicator) indicator.remove();
  }

  function appendAssistantResponse(data) {
    removeLoadingIndicator();

    const row = document.createElement("div");
    row.className = "message-row";

    const isHit = !!data.cache_hit;
    const latency = Number(data.latency_ms || 0);
    const bubbleClass = isHit ? "msg-bubble assistant-bubble is-cache-hit" : "msg-bubble assistant-bubble";

    // Format citations / source references (Tracked and Validated)
    const activeSources = data.citations || data.source_references || data.sources || [];
    let citationsHtml = "";
    if (activeSources.length > 0) {
      const sourceBadges = activeSources.map((src, idx) => {
        const title = escapeHtml(src.title || src.doc_id || "Policy Document");
        const page = src.page_number ? `Page ${src.page_number}` : (src.source_ref || "");
        const chunkId = escapeHtml(src.chunk_id || "");
        return `
          <button class="source-badge source-badge-clickable" data-idx="${idx}" type="button" title="Click to view supporting passage &amp; PDF page">
            <span>📄 ${title}</span>
            <span style="opacity:0.85;font-size:0.65rem;background:rgba(56,189,248,0.18);padding:1px 5px;border-radius:4px;">${escapeHtml(page)}</span>
            <span style="color:#34D399;font-size:0.65rem;font-weight:700;">✓ Validated</span>
          </button>
        `;
      }).join("");

      citationsHtml = `
        <div class="citations-wrap">
          <span class="citations-label">📚 Actively Cited Evidence (${activeSources.length} Verified References)</span>
          <div class="sources-chips-row">
            ${sourceBadges}
          </div>
        </div>
      `;
    }

    let badgeHtml = "";
    if (isHit) {
      if (data.cache_type === "semantic") {
        const simPct = Math.round((data.similarity_score || 0.9) * 100);
        badgeHtml = `<span class="cache-pill-hit" style="background:rgba(147,51,234,0.18); border-color:rgba(168,85,247,0.4); color:#C084FC;">🎯 Semantic Cache Hit · ${latency.toFixed(1)}ms · Equivalence (${simPct}%)</span>`;
      } else {
        badgeHtml = `<span class="cache-pill-hit">⚡ Exact Cache Hit · ${latency.toFixed(1)}ms · Redis (LLM Skipped)</span>`;
      }
    } else {
      badgeHtml = `<span class="cache-pill-miss">🔍 Cache Miss · ${latency.toFixed(1)}ms · Hybrid + Cross-Encoder</span>`;
    }

    // Process answer text (convert newlines to <br> and bold markers)
    const formattedAnswer = escapeHtml(data.answer)
      .replace(/\n\n/g, "<br><br>")
      .replace(/\n/g, "<br>")
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");

    row.innerHTML = `
      <div class="agent-avatar" style="width:32px;height:32px;flex-shrink:0;">AI</div>
      <div class="${bubbleClass}">
        <button class="copy-btn" title="Copy answer" aria-label="Copy answer">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
        </button>
        <div class="bubble-meta-header">
          ${badgeHtml}
          <span style="font-size:0.68rem; color:var(--text-faint); font-family:var(--font-mono);">${formatTime()}</span>
        </div>
        <div class="answer-content" style="color:var(--text-main); font-size:0.92rem; line-height:1.65;">
          ${formattedAnswer}
        </div>
        ${citationsHtml}
      </div>
    `;

    // Attach copy button action
    const copyBtn = row.querySelector(".copy-btn");
    if (copyBtn) {
      copyBtn.addEventListener("click", () => {
        navigator.clipboard.writeText(data.answer).then(() => {
          copyBtn.classList.add("copied");
          showToast("Answer copied to clipboard!", "info");
          setTimeout(() => copyBtn.classList.remove("copied"), 2000);
        });
      });
    }

    // Attach citation inspection click handlers
    row.querySelectorAll(".source-badge-clickable").forEach((badge) => {
      badge.addEventListener("click", () => {
        const idx = parseInt(badge.getAttribute("data-idx"), 10);
        const srcObj = activeSources[idx];
        if (srcObj) openEvidenceModal(srcObj);
      });
    });

    chatFeed.appendChild(row);
    scrollToBottom();
  }

  // --------------------------------------------------------------------------
  // Core Chat Execution
  // --------------------------------------------------------------------------
  async function submitQuestion(questionText) {
    const q = (questionText || "").trim();
    if (!q || isSending) return;

    isSending = true;
    chatInput.value = "";
    chatInput.style.height = "auto";
    sendBtn.disabled = true;

    appendUserMessage(q);
    showLoadingIndicator();

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: "Network error occurred." }));
        throw new Error(errorData.detail || `Server error ${response.status}`);
      }

      const result = await response.json();
      appendAssistantResponse(result);
    } catch (err) {
      removeLoadingIndicator();
      const errorRow = document.createElement("div");
      errorRow.className = "message-row";
      errorRow.innerHTML = `
        <div class="agent-avatar" style="width:32px;height:32px;flex-shrink:0;background:#EF4444;">!</div>
        <div class="msg-bubble assistant-bubble" style="border-color:rgba(239,68,68,0.4);">
          <div style="color:#F87171;font-weight:600;font-size:0.88rem;">Error Generating Response</div>
          <div style="font-size:0.82rem;color:var(--text-muted);margin-top:4px;">${escapeHtml(err.message)}</div>
        </div>
      `;
      chatFeed.appendChild(errorRow);
      scrollToBottom();
      showToast(`Error: ${err.message}`, "error");
    } finally {
      isSending = false;
      sendBtn.disabled = false;
      refreshTelemetry();
      chatInput.focus();
    }
  }

  // --------------------------------------------------------------------------
  // Event Listeners
  // --------------------------------------------------------------------------
  sendBtn.addEventListener("click", () => {
    submitQuestion(chatInput.value);
  });

  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submitQuestion(chatInput.value);
    }
  });

  chatInput.addEventListener("input", () => {
    chatInput.style.height = "auto";
    chatInput.style.height = `${Math.min(chatInput.scrollHeight, 120)}px`;
  });

  // FAQ Chips click handler
  document.querySelectorAll(".faq-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const question = chip.getAttribute("data-question");
      if (question) {
        submitQuestion(question);
      }
    });
  });

  // Clear Chat Feed
  clearChatBtn.addEventListener("click", () => {
    chatFeed.innerHTML = "";
    if (welcomeBox) {
      chatFeed.appendChild(welcomeBox);
      welcomeBox.style.display = "block";
    }
    showToast("Chat feed cleared.", "info");
  });

  // Cold Miss vs Warm Hit Live Demonstration
  testRepeatBtn.addEventListener("click", async () => {
    if (isSending) return;
    const testQuery = "How long do I have to return an item and who pays return shipping?";
    showToast("Running Cold vs Hit test: Step 1 (Cold Query)...", "info");

    await submitQuestion(testQuery);

    // Short delay then re-send exact query to trigger warm cache hit
    setTimeout(async () => {
      showToast("Step 2: Sending identical query (Expecting instantaneous Cache Hit)...", "info");
      await submitQuestion(testQuery);
      showToast("Demo complete! Notice the instantaneous response and emerald Cache Hit badge.", "info");
    }, 800);
  });

  // Cache Invalidation Demo (Policy Update)
  simulateUpdateBtn.addEventListener("click", async () => {
    simulateUpdateBtn.disabled = true;
    policyExtended = !policyExtended;

    const newReturnDays = policyExtended ? "45" : "30";
    const updatePayload = {
      chunk_id: "ret_01",
      new_policy_text: `NovaCart standard return window: Customers may initiate a return within ${newReturnDays} calendar days of package delivery for a full refund to the original payment method. Items must be in new, unworn, and unwashed condition with original tags attached. We provide a prepaid return shipping label for domestic returns; a \$4.99 restocking fee is deducted unless the return is due to our error.`,
    };

    try {
      const res = await fetch("/api/kb/update-policy", {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          "X-Admin-Token": "novacart-admin-key-2026",
        },
        body: JSON.stringify(updatePayload),
      });

      if (!res.ok) throw new Error("Policy update failed");
      const data = await res.json();

      simulateUpdateBtn.querySelector("span").textContent = policyExtended
        ? "Revert Policy (45d → 30d)"
        : "Update Policy (30d → 45d)";

      showToast(`Knowledge Base updated to ${newReturnDays} days! KB hash changed. Cached answers invalidated.`, "info");
      await refreshKnowledgeBase();
      await refreshTelemetry();
    } catch (err) {
      showToast(`Policy update failed: ${err.message}`, "error");
    } finally {
      simulateUpdateBtn.disabled = false;
    }
  });

  // Purge Redis Cache
  clearCacheBtn.addEventListener("click", async () => {
    clearCacheBtn.disabled = true;
    try {
      const res = await fetch("/api/cache/clear", { 
        method: "POST",
        headers: {
          "X-Admin-Token": "novacart-admin-key-2026",
        },
      });
      if (!res.ok) throw new Error("Cache purge failed");
      showToast("Redis answer cache flushed successfully.", "info");
      await refreshTelemetry();
    } catch (err) {
      showToast(`Cache purge failed: ${err.message}`, "error");
    } finally {
      clearCacheBtn.disabled = false;
    }
  });

  // --------------------------------------------------------------------------
  // Initialization & Auto-Polling
  // --------------------------------------------------------------------------
  refreshTelemetry();
  refreshKnowledgeBase();
  checkHealth();

  // Poll metrics every 6 seconds to keep telemetry fresh
  setInterval(refreshTelemetry, 6000);
});

