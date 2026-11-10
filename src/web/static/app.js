/**
 * Minimalist & Reactive Client Application for PDF-Based RAG
 * Supports Multi-Document Ingestion, Incremental Uploads, and Scoped Retrieval.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const chatContainer = document.getElementById("chatContainer");
  const welcomeScreen = document.getElementById("welcomeScreen");
  const chatInput = document.getElementById("chatInput");
  const sendBtn = document.getElementById("sendBtn");
  const fileInput = document.getElementById("fileInput");
  const uploadCard = document.getElementById("uploadCard");
  const documentList = document.getElementById("documentList");
  const docCountBadge = document.getElementById("docCount");
  const clearBtn = document.getElementById("clearBtn");
  const mobileMenuBtn = document.getElementById("mobileMenuBtn");
  const sidebar = document.getElementById("sidebar");
  const sidebarBackdrop = document.getElementById("sidebarBackdrop");
  
  // Scope controls
  const selectAllDocsBtn = document.getElementById("selectAllDocsBtn");
  const clearScopeDocsBtn = document.getElementById("clearScopeDocsBtn");
  const topScopeBadge = document.getElementById("topScopeBadge");
  const topScopeText = document.getElementById("topScopeText");

  // Footer stats
  const statModel = document.getElementById("statModel");
  const statChunks = document.getElementById("statChunks");
  const statCache = document.getElementById("statCache");
  const topModelBadge = document.getElementById("topModelBadge");

  // State
  let isGenerating = false;
  let selectedDocs = new Set();
  let availableDocs = [];

  // -------------------------------------------------------------------------
  // 1. Scope Filter UI
  // -------------------------------------------------------------------------
  function updateScopeUI() {
    if (selectedDocs.size === 0) {
      topScopeText.textContent = "Scope: All Documents";
      topScopeBadge.classList.remove("active-filtered");
      topScopeBadge.title = "Click to reset filter (currently searching all documents)";
    } else {
      const names = Array.from(selectedDocs).map(f => f.length > 15 ? f.substring(0, 12) + "..." : f);
      topScopeText.textContent = `Scope: ${selectedDocs.size} Selected (${names.join(", ")})`;
      topScopeBadge.classList.add("active-filtered");
      topScopeBadge.title = "Filtered search active. Click to reset and search all documents.";
    }

    // Update checkboxes in sidebar
    document.querySelectorAll(".doc-checkbox").forEach(cb => {
      const fname = cb.dataset.filename;
      cb.checked = selectedDocs.has(fname);
      const parent = cb.closest(".doc-item");
      if (parent) {
        if (selectedDocs.has(fname)) {
          parent.classList.add("selected");
        } else {
          parent.classList.remove("selected");
        }
      }
    });
  }

  if (selectAllDocsBtn) {
    selectAllDocsBtn.addEventListener("click", () => {
      availableDocs.forEach(d => selectedDocs.add(d.filename));
      updateScopeUI();
    });
  }

  if (clearScopeDocsBtn) {
    clearScopeDocsBtn.addEventListener("click", () => {
      selectedDocs.clear();
      updateScopeUI();
    });
  }

  if (topScopeBadge) {
    topScopeBadge.addEventListener("click", () => {
      selectedDocs.clear();
      updateScopeUI();
    });
  }

  // -------------------------------------------------------------------------
  // 2. Initial State & Data Fetching
  // -------------------------------------------------------------------------
  async function fetchStatus() {
    try {
      const res = await fetch("/api/status");
      const data = await res.json();
      
      const providerLabel = data.llm_provider === "groq" ? "Groq Cloud" : "Local Ollama";
      statModel.textContent = `${providerLabel} (${data.llm_model})`;
      statChunks.textContent = `${data.total_records} chunk${data.total_records === 1 ? '' : 's'}`;
      if (statCache && data.cache) {
        statCache.textContent = `${data.cache.total_keys} keys · ${data.cache_strategy || 'aside'}`;
      }
      topModelBadge.innerHTML = `Model: <strong>${data.llm_model}</strong> · ${providerLabel}`;
    } catch (err) {
      console.error("Failed to load status:", err);
    }
  }

  async function fetchDocuments() {
    try {
      const res = await fetch("/api/documents");
      const data = await res.json();
      availableDocs = data.documents || [];
      renderDocumentList(availableDocs);
    } catch (err) {
      console.error("Failed to fetch documents:", err);
    }
  }

  function renderDocumentList(docs) {
    documentList.innerHTML = "";
    docCountBadge.textContent = docs.length;

    if (!docs || docs.length === 0) {
      documentList.innerHTML = `
        <div style="font-size: 0.76rem; color: var(--text-muted); text-align: center; padding: 12px 0;">
          No PDFs ingested yet.<br>Drop files in the box above!
        </div>
      `;
      selectedDocs.clear();
      updateScopeUI();
      return;
    }

    // Remove any stale selections of files that were deleted
    const validNames = new Set(docs.map(d => d.filename));
    for (const name of selectedDocs) {
      if (!validNames.has(name)) {
        selectedDocs.delete(name);
      }
    }

    docs.forEach(doc => {
      const item = document.createElement("div");
      item.className = "doc-item" + (selectedDocs.has(doc.filename) ? " selected" : "");
      item.innerHTML = `
        <div class="doc-item-left" data-filename="${doc.filename}">
          <input type="checkbox" class="doc-checkbox" data-filename="${doc.filename}" ${selectedDocs.has(doc.filename) ? 'checked' : ''}>
          <div class="doc-info">
            <div class="doc-name" title="${doc.filename}">📄 ${doc.filename}</div>
            <div class="doc-meta">
              <span>${doc.total_chunks} chunk${doc.total_chunks === 1 ? '' : 's'}</span>
              <span>·</span>
              <span>${doc.total_pages} page${doc.total_pages === 1 ? '' : 's'}</span>
            </div>
          </div>
        </div>
        <button class="doc-delete-btn" title="Delete document" data-filename="${doc.filename}">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="3 6 5 6 21 6"></polyline>
            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
          </svg>
        </button>
      `;

      // Toggle selection on checkbox or row click
      const leftRow = item.querySelector(".doc-item-left");
      leftRow.addEventListener("click", (e) => {
        const fname = doc.filename;
        if (selectedDocs.has(fname)) {
          selectedDocs.delete(fname);
        } else {
          selectedDocs.add(fname);
        }
        updateScopeUI();
      });

      // Prevent duplicate click when clicking checkbox directly
      const checkbox = item.querySelector(".doc-checkbox");
      checkbox.addEventListener("click", (e) => {
        e.stopPropagation();
        const fname = doc.filename;
        if (checkbox.checked) {
          selectedDocs.add(fname);
        } else {
          selectedDocs.delete(fname);
        }
        updateScopeUI();
      });

      // Delete document
      item.querySelector(".doc-delete-btn").addEventListener("click", async (e) => {
        e.stopPropagation();
        const fname = e.currentTarget.dataset.filename;
        if (confirm(`Remove '${fname}' from the knowledge base?`)) {
          await deleteDocument(fname);
        }
      });

      documentList.appendChild(item);
    });

    updateScopeUI();
  }

  async function deleteDocument(filename) {
    try {
      const res = await fetch(`/api/documents/${encodeURIComponent(filename)}`, {
        method: "DELETE",
      });
      if (res.ok) {
        selectedDocs.delete(filename);
        await fetchDocuments();
        await fetchStatus();
      } else {
        alert("Failed to delete document.");
      }
    } catch (err) {
      console.error("Error deleting doc:", err);
    }
  }

  // -------------------------------------------------------------------------
  // 3. Drag & Drop Multi-File Upload
  // -------------------------------------------------------------------------
  uploadCard.addEventListener("click", () => fileInput.click());

  uploadCard.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadCard.classList.add("drag-over");
  });

  uploadCard.addEventListener("dragleave", () => {
    uploadCard.classList.remove("drag-over");
  });

  uploadCard.addEventListener("drop", async (e) => {
    e.preventDefault();
    uploadCard.classList.remove("drag-over");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      await handleFileUpload(e.dataTransfer.files);
    }
  });

  fileInput.addEventListener("change", async () => {
    if (fileInput.files && fileInput.files.length > 0) {
      await handleFileUpload(fileInput.files);
      fileInput.value = "";
    }
  });

  async function handleFileUpload(fileList) {
    const formData = new FormData();
    let pdfCount = 0;

    for (let i = 0; i < fileList.length; i++) {
      if (fileList[i].name.toLowerCase().endsWith(".pdf")) {
        formData.append("files", fileList[i]);
        pdfCount++;
      }
    }

    if (pdfCount === 0) {
      alert("Please select PDF files (.pdf) only.");
      return;
    }

    const titleEl = uploadCard.querySelector(".upload-title");
    const origText = titleEl.textContent;
    titleEl.textContent = `Indexing ${pdfCount} file(s)...`;
    uploadCard.style.pointerEvents = "none";

    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (res.ok) {
        await fetchDocuments();
        await fetchStatus();
      } else {
        alert(data.detail || "Upload failed.");
      }
    } catch (err) {
      alert("Error uploading files: " + err);
    } finally {
      titleEl.textContent = origText;
      uploadCard.style.pointerEvents = "auto";
    }
  }

  // -------------------------------------------------------------------------
  // 4. Question Answering & Messaging
  // -------------------------------------------------------------------------
  function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }

  function appendUserMessage(text) {
    if (welcomeScreen) {
      welcomeScreen.style.display = "none";
    }

    const row = document.createElement("div");
    row.className = "message-row user-row";
    row.innerHTML = `
      <div class="message-bubble user-bubble">${escapeHtml(text)}</div>
    `;
    chatContainer.appendChild(row);
    scrollToBottom();
  }

  function appendAssistantLoading() {
    const row = document.createElement("div");
    row.className = "message-row assistant-row";
    row.id = "loadingMessageRow";
    row.innerHTML = `
      <div class="avatar assistant-avatar">AI</div>
      <div class="message-bubble assistant-bubble" style="display: flex; align-items: center; gap: 8px;">
        <span class="spinner" style="width: 14px; height: 14px;"></span>
        <span style="font-size: 0.82rem; color: var(--text-secondary);">Searching passages & synthesizing answer...</span>
      </div>
    `;
    chatContainer.appendChild(row);
    scrollToBottom();
    return row;
  }

  function formatAnswerMarkdown(text) {
    // 1. Escape HTML
    let out = escapeHtml(text);

    // 2. Bold text **text**
    out = out.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // 3. Bracketed citations [filename.pdf, Page X]
    out = out.replace(/\[([a-zA-Z0-9_\-\. ]+),\s*Page\s*(\d+)\]/g, (match, file, page) => {
      return `<span class="citation-badge" title="Source: ${file}, Page ${page}">📄 ${file} · p.${page}</span>`;
    });

    // 4. Linebreaks
    out = out.replace(/\n\n/g, '</p><p>').replace(/\n/g, '<br>');
    return `<p>${out}</p>`;
  }

  function appendAssistantMessage(data) {
    const loadingRow = document.getElementById("loadingMessageRow");
    if (loadingRow) {
      loadingRow.remove();
    }

    const row = document.createElement("div");
    row.className = "message-row assistant-row";

    // Scope header & Cache tag
    let scopeBadgeHtml = "";
    if (data.document_scope && data.document_scope.length > 0) {
      scopeBadgeHtml = `
        <div style="font-size: 0.72rem; color: #F59E0B; font-weight: 600; display: flex; align-items: center; gap: 4px;">
          <span>🎯 Scoped search restricted to: ${data.document_scope.join(", ")}</span>
        </div>
      `;
    }

    let cacheBadgeHtml = "";
    if (data.latency_ms !== undefined && data.latency_ms !== null) {
      if (data.is_cached) {
        cacheBadgeHtml = `<span style="font-size:0.70rem; color: #10B981; font-weight:600; background:rgba(16,185,129,0.12); padding:2px 8px; border-radius:12px;">⚡ Cache HIT (${data.latency_ms}ms)</span>`;
      } else {
        cacheBadgeHtml = `<span style="font-size:0.70rem; color: #94A3B8; font-weight:600; background:rgba(148,163,184,0.12); padding:2px 8px; border-radius:12px;">⏱ Fresh RAG (${data.latency_ms}ms)</span>`;
      }
    }

    let topInfoBar = "";
    if (scopeBadgeHtml || cacheBadgeHtml) {
      topInfoBar = `
        <div style="display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 8px;">
          <div>${scopeBadgeHtml}</div>
          <div>${cacheBadgeHtml}</div>
        </div>
      `;
    }

    let sourcesHtml = "";
    if (data.has_evidence && data.sources && data.sources.length > 0) {
      const sourceBadges = data.sources.map(s => {
        const fname = s.doc_name || s.filename;
        return `
          <span class="citation-badge">
            📄 ${fname} (p.${s.page_number}) · ${Math.round(s.similarity * 100)}% match
          </span>
        `;
      }).join("");

      sourcesHtml = `
        <div class="citations-footer">
          <span class="sources-label">Sources:</span>
          ${sourceBadges}
        </div>
      `;
    }

    let passagesHtml = "";
    if (data.retrieved_passages && data.retrieved_passages.length > 0) {
      const cards = data.retrieved_passages.map((p, idx) => {
        const fname = p.doc_name || p.filename;
        return `
          <div class="passage-card">
            <div class="passage-header">
              <span><strong>#${idx + 1}</strong> ${fname} · Page ${p.page_number}</span>
              <span>Distance: ${p.distance.toFixed(3)}</span>
            </div>
            <div class="passage-text">"${escapeHtml(p.text)}"</div>
          </div>
        `;
      }).join("");

      passagesHtml = `
        <div class="evidence-drawer">
          <button class="evidence-toggle" type="button">
            <span>Retrieved Evidence Passages (${data.retrieved_passages.length})</span>
            <span class="chevron">▼</span>
          </button>
          <div class="evidence-content" style="display: none;">
            ${cards}
          </div>
        </div>
      `;
    }

    row.innerHTML = `
      <div class="avatar assistant-avatar">AI</div>
      <div class="message-bubble assistant-bubble">
        ${topInfoBar}
        ${formatAnswerMarkdown(data.answer)}
        ${sourcesHtml}
        ${passagesHtml}
      </div>
    `;

    // Hook up accordion toggle
    const toggleBtn = row.querySelector(".evidence-toggle");
    if (toggleBtn) {
      const content = row.querySelector(".evidence-content");
      const chevron = row.querySelector(".chevron");
      toggleBtn.addEventListener("click", () => {
        const isHidden = content.style.display === "none";
        content.style.display = isHidden ? "flex" : "none";
        chevron.textContent = isHidden ? "▲" : "▼";
      });
    }

    chatContainer.appendChild(row);
    scrollToBottom();
  }

  async function handleSend() {
    const text = chatInput.value.trim();
    if (!text || isGenerating) return;

    chatInput.value = "";
    chatInput.style.height = "auto";
    isGenerating = true;
    sendBtn.disabled = true;

    appendUserMessage(text);
    appendAssistantLoading();

    // Prepare payload with optional document_ids scope
    const payload = {
      question: text,
      document_ids: selectedDocs.size > 0 ? Array.from(selectedDocs) : null
    };

    try {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json();
      if (res.ok) {
        appendAssistantMessage(data);
      } else {
        appendAssistantMessage({
          answer: `[Error] ${data.detail || "Failed to generate answer."}`,
          has_evidence: false,
          sources: [],
          retrieved_passages: []
        });
      }
    } catch (err) {
      appendAssistantMessage({
        answer: `[Network Error] Unable to communicate with the RAG server: ${err}`,
        has_evidence: false,
        sources: [],
        retrieved_passages: []
      });
    } finally {
      isGenerating = false;
      sendBtn.disabled = false;
      chatInput.focus();
      fetchStatus();
    }
  }

  // -------------------------------------------------------------------------
  // 5. Input Events & Micro-interactions
  // -------------------------------------------------------------------------
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });

  chatInput.addEventListener("input", () => {
    chatInput.style.height = "auto";
    chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + "px";
  });

  sendBtn.addEventListener("click", handleSend);

  // Suggestion chips
  document.querySelectorAll(".suggestion-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      chatInput.value = chip.dataset.query;
      handleSend();
    });
  });

  // Clear chat
  clearBtn.addEventListener("click", () => {
    const messages = chatContainer.querySelectorAll(".message-row");
    messages.forEach(m => m.remove());
    if (welcomeScreen) {
      welcomeScreen.style.display = "block";
    }
  });

  // Mobile menu toggle
  if (mobileMenuBtn) {
    mobileMenuBtn.addEventListener("click", () => {
      sidebar.classList.toggle("open");
      sidebarBackdrop.classList.toggle("active");
    });

    sidebarBackdrop.addEventListener("click", () => {
      sidebar.classList.remove("open");
      sidebarBackdrop.classList.remove("active");
    });
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Initial Load
  fetchStatus();
  fetchDocuments();
});
