"use strict";

const $ = (id) => document.getElementById(id);
let sending = false;
let noticeTimer;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function notify(text) {
  clearTimeout(noticeTimer);
  $("notice").textContent = text;
  noticeTimer = setTimeout(() => { $("notice").textContent = ""; }, 3500);
}

async function request(path, options = {}) {
  const response = await fetch(path, { ...options, signal: AbortSignal.timeout(60000) });
  if (response.status === 429) throw new Error("You've sent several questions. Please wait a minute before trying again.");
  if (!response.ok) throw new Error("We couldn't reach support. Please try again.");
  return response.json();
}

function openEvidence(title, sources) {
  $("evidenceTitle").textContent = title;
  $("evidenceBody").replaceChildren();
  for (const source of sources) {
    $("evidenceBody").append(
      element("h3", "", source.section || "Policy excerpt"),
      element("p", "source-meta", source.source_ref || `Page ${source.page_number || 1}`),
      element("p", "policy-excerpt", source.supporting_passage || source.text || "")
    );
  }
  $("evidence").showModal();
}

function scrollMessages() { $("messages").scrollTop = $("messages").scrollHeight; }

function renderAnswer(data) {
  const row = element("article", `message assistant${data.error ? " error" : ""}`);
  row.append(element("p", "message-label", data.is_supported === false ? "More help is needed" : "From the policy library"));
  const answer = element("p", "answer");
  const sources = data.citations || [];
  // Keep original numbers: a lone [3] must not open source [1].
  const byNumber = new Map(sources.map((source) => [String(source.citation_number), source]));
  for (const part of (data.answer || "No answer was returned. Please try again.").split(/(\[\d+\])/)) {
    const source = byNumber.get(part.slice(1, -1));
    if (/^\[\d+\]$/.test(part) && source) {
      const link = element("button", "citation", part);
      link.type = "button";
      link.setAttribute("aria-label", `Read source ${part}: ${source.title}`);
      link.addEventListener("click", () => openEvidence(source.title, [source]));
      answer.append(link);
    } else answer.append(document.createTextNode(part));
  }
  row.append(answer);
  if (data.is_supported === false) {
    const handoff = element("p", "muted", "For help with your order, contact the support team through your account portal.");
    if (data.support_contact_url) {
      try {
        const url = new URL(data.support_contact_url);
        if (url.protocol === "https:" && !url.username && !url.password) {
          const link = element("a", "text-button", "Contact support");
          link.href = url.href;
          handoff.append(document.createTextNode(" "), link);
        }
      } catch { /* Keep the account-portal guidance if configuration is invalid. */ }
    }
    row.append(handoff);
  }
  if (sources.length) {
    const references = element("div", "sources");
    for (const source of sources) {
      const button = element("button", "source-button", `${source.citation_number ? `[${source.citation_number}] ` : ""}${source.section || source.title}`);
      button.type = "button";
      button.addEventListener("click", () => openEvidence(source.title, [source]));
      references.append(button);
    }
    row.append(references);
  }
  const actions = element("div", "message-actions");
  const elapsed = Number(data.latency_ms || 0);
  const duration = elapsed < 1000 ? `${Math.round(elapsed)} ms` : `${(elapsed / 1000).toFixed(1)} s`;
  actions.append(element("span", "", `${data.cache_hit ? "Saved answer" : "Policy lookup"} · ${duration}`));
  const copy = element("button", "", "Copy answer");
  copy.type = "button";
  copy.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(data.answer); notify("Answer copied."); }
    catch { notify("Couldn't copy. Select the answer text to copy it."); }
  });
  actions.append(copy);
  row.append(actions);
  return row;
}

async function submitQuestion(question) {
  const text = question.trim();
  if (!text || sending) return;
  sending = true;
  $("send").disabled = true;
  $("newConversation").disabled = true;
  $("welcome").hidden = true;
  $("question").value = "";
  $("question").style.height = "auto";
  const pending = element("p", "pending", "Looking through our policies…");
  $("messages").append(element("div", "message user", text), pending);
  scrollMessages();
  try {
    const data = await request("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: text }) });
    pending.replaceWith(renderAnswer(data));
  } catch (error) {
    const row = element("div", "message assistant error");
    row.append(element("p", "", error.name === "TimeoutError" ? "This is taking longer than expected. Please try again." : error.message));
    const retry = element("button", "text-button", "Try again ↗");
    retry.type = "button";
    retry.addEventListener("click", () => { if (!sending) { row.remove(); submitQuestion(text); } });
    row.append(retry);
    pending.replaceWith(row);
  } finally {
    sending = false;
    $("send").disabled = false;
    $("newConversation").disabled = false;
    scrollMessages();
    $("question").focus({ preventScroll: true });
    if ($("performance").open) refreshMetrics();
  }
}

async function refreshMetrics() {
  try {
    const data = await request("/api/metrics");
    const grid = element("div", "metric-grid");
    for (const [value, label] of [[data.total_questions, "Questions processed"], [data.cache_hits, "Answers reused"], [`${Number(data.hit_rate_pct).toFixed(0)}%`, "Cache hit rate"]]) {
      const item = element("div");
      item.append(element("strong", "", String(value)), element("span", "", label));
      grid.append(item);
    }
    $("metrics").replaceChildren(grid, element("p", "metrics-note", `${!data.is_connected ? "Redis unavailable; serving live answers" : "Redis cache"} · Counts since server start or cache reset.`));
  } catch { $("metrics").textContent = "Performance details are temporarily unavailable."; }
}

$("questionForm").addEventListener("submit", (event) => { event.preventDefault(); submitQuestion($("question").value); });
$("question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); submitQuestion(event.target.value); }
});
$("question").addEventListener("input", () => { $("question").style.height = "auto"; $("question").style.height = `${Math.min($("question").scrollHeight, 180)}px`; });
document.querySelectorAll("[data-question]").forEach((button) => button.addEventListener("click", () => submitQuestion(button.dataset.question)));
$("newConversation").addEventListener("click", () => { if (sending) return; $("messages").replaceChildren(); $("welcome").hidden = false; $("question").value = ""; $("question").focus(); });
$("closeEvidence").addEventListener("click", () => $("evidence").close());
$("evidence").addEventListener("click", (event) => {
  if (event.target === $("evidence")) {
    const bounds = $("evidence").getBoundingClientRect();
    if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) $("evidence").close();
  }
});
$("performance").addEventListener("toggle", () => { if ($("performance").open) refreshMetrics(); });
if (matchMedia("(max-width: 680px)").matches) $("policiesPanel").open = false;

request("/api/kb").then((data) => {
  $("policyList").replaceChildren();
  for (const doc of data.documents) {
    const button = element("button", "policy-button", doc.title.replace(/ Policy$/, ""));
    button.type = "button";
    button.addEventListener("click", () => openEvidence(doc.title, data.chunks.filter((chunk) => chunk.doc_id === doc.doc_id)));
    $("policyList").append(button);
  }
}).catch(() => { $("policyList").textContent = "Policies are temporarily unavailable. Refresh to try again."; });
request("/api/health").catch(() => {
  $("modeNote").textContent = "Support is temporarily unavailable. You can try sending your question again.";
});
