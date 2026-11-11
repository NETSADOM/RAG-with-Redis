import React, { useState, useEffect, useRef } from "react";
import Head from "next/head";
import { 
  Zap, 
  Search, 
  RotateCw, 
  Trash2, 
  Send, 
  Copy, 
  Check, 
  FileText, 
  ExternalLink,
  ChevronDown,
  Sparkles,
  ShieldAlert
} from "lucide-react";
import { Message, ChatResponse, CacheMetrics, KnowledgeBaseResponse, SourceReference } from "@/types";

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [policyExtended, setPolicyExtended] = useState(false);
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  const [metrics, setMetrics] = useState<CacheMetrics>({
    is_connected: false,
    is_emulator: true,
    total_questions: 0,
    cache_hits: 0,
    cache_misses: 0,
    hit_rate_pct: 0,
    avg_hit_latency_ms: 0,
    avg_miss_latency_ms: 0,
    llm_calls_avoided: 0,
    estimated_cost_saved_usd: 0,
    total_keys_in_redis: 0,
    default_ttl_seconds: 3600,
  });

  const [kb, setKb] = useState<KnowledgeBaseResponse>({
    company: "NovaCart Essentials",
    version_hash: "loading...",
    total_chunks: 13,
    documents: [],
  });

  const chatEndRef = useRef<HTMLDivElement>(null);

  const showToast = (msg: string) => {
    setToastMessage(msg);
    setTimeout(() => setToastMessage(null), 3500);
  };

  const fetchMetrics = async () => {
    try {
      const res = await fetch("/api/metrics");
      if (res.ok) {
        const data = await res.json();
        setMetrics(data);
      }
    } catch (err) {
      console.error("Failed to fetch metrics", err);
    }
  };

  const fetchKb = async () => {
    try {
      const res = await fetch("/api/kb");
      if (res.ok) {
        const data = await res.json();
        setKb(data);
      }
    } catch (err) {
      console.error("Failed to fetch KB", err);
    }
  };

  useEffect(() => {
    fetchMetrics();
    fetchKb();
    const interval = setInterval(fetchMetrics, 6000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const handleSend = async (overrideText?: string) => {
    const textToSend = (overrideText || input).trim();
    if (!textToSend || loading) return;

    const userMsg: Message = {
      id: Date.now().toString(),
      role: "user",
      content: textToSend,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };

    setMessages((prev) => [...prev, userMsg]);
    if (!overrideText) setInput("");
    setLoading(true);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: textToSend }),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: ChatResponse = await res.json();

      const assistantMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: data.answer,
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        cacheHit: data.cache_hit,
        cacheType: data.cache_type,
        similarityScore: data.similarity_score,
        latencyMs: data.latency_ms,
        sources: data.source_references || data.sources,
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err: any) {
      const errorMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: `Error generating response: ${err.message}`,
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
      fetchMetrics();
    }
  };

  const handleColdVsHitTest = async () => {
    if (loading) return;
    const testQ = "How long do I have to return an item and who pays return shipping?";
    showToast("Step 1: Running cold query through Hybrid Search + Cross-Encoder...");
    await handleSend(testQ);

    setTimeout(async () => {
      showToast("Step 2: Sending identical query (Expecting instant Redis Cache Hit)...");
      await handleSend(testQ);
      showToast("Demo complete! Notice the instantaneous response and emerald badge.");
    }, 800);
  };

  const handleUpdatePolicy = async () => {
    const nextExtended = !policyExtended;
    const days = nextExtended ? "45" : "30";
    try {
      const res = await fetch("/api/kb/update-policy", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          chunk_id: "ret_01",
          new_policy_text: `NovaCart standard return window: Customers may initiate a return within ${days} calendar days of package delivery for a full refund to original payment method. Items must be unworn and in original condition.`,
        }),
      });
      if (res.ok) {
        setPolicyExtended(nextExtended);
        showToast(`Knowledge Base updated to ${days} days! Version hash changed; stale cache bypassed.`);
        fetchKb();
        fetchMetrics();
      }
    } catch (err: any) {
      showToast(`Policy update failed: ${err.message}`);
    }
  };

  const handleClearCache = async () => {
    try {
      const res = await fetch("/api/cache/clear", { method: "POST" });
      if (res.ok) {
        showToast("Redis answer cache flushed successfully.");
        fetchMetrics();
      }
    } catch (err: any) {
      showToast(`Flush error: ${err.message}`);
    }
  };

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const faqs = [
    { text: "How long do I have to return an item and who pays return shipping?", label: "Return window & shipping fees" },
    { text: "What are your shipping options and rates for standard delivery?", label: "Shipping speeds & pricing" },
    { text: "Can I cancel my order after placing it?", label: "Order cancellation cutoff" },
    { text: "My package arrived with damaged items, what should I do?", label: "Damaged item replacement" },
    { text: "What payment methods and digital wallets do you accept?", label: "Supported payment options" },
  ];

  return (
    <>
      <Head>
        <title>NovaCart Essentials · Customer Support RAG</title>
        <meta name="description" content="Customer Support RAG with Hybrid Search, Cross-Encoder Reranking, and Redis Caching" />
      </Head>

      <div className="flex h-screen w-screen overflow-hidden bg-[#070A10] text-slate-100 font-sans">
        {/* Sidebar Dashboard */}
        <aside className="w-80 h-full border-r border-white/10 bg-[#0B101D]/90 backdrop-blur-xl flex flex-col flex-shrink-0 z-30">
          <div className="p-4 border-b border-white/10 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-lg bg-gradient-to-tr from-sky-500 to-indigo-600 flex items-center justify-center font-extrabold text-white text-sm shadow-lg shadow-sky-500/20">
                NC
              </div>
              <div>
                <h1 className="font-bold text-sm tracking-tight text-white">NovaCart Support</h1>
                <span className="text-[10px] font-semibold text-sky-400 uppercase tracking-wider block">Enterprise RAG</span>
              </div>
            </div>
            <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[11px] font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
              Live
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {/* Performance Telemetry Widget */}
            <div className="rounded-xl border border-white/10 bg-white/[0.02] p-3.5 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Performance Telemetry</span>
                <span className="px-2 py-0.5 rounded-full text-xs font-bold text-emerald-400 bg-emerald-500/10 border border-emerald-500/20">
                  {metrics.hit_rate_pct.toFixed(1)}% Hit Rate
                </span>
              </div>

              {/* Progress bar */}
              <div className="w-full h-1.5 bg-white/5 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-gradient-to-r from-emerald-500 to-sky-400 rounded-full transition-all duration-500" 
                  style={{ width: `${Math.min(100, Math.max(0, metrics.hit_rate_pct))}%` }}
                />
              </div>

              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="p-2 rounded bg-white/[0.02] border border-white/5">
                  <div className="font-mono font-bold text-sm text-white">{metrics.total_questions}</div>
                  <div className="text-[10px] text-slate-400 uppercase">Inquiries</div>
                </div>
                <div className="p-2 rounded bg-white/[0.02] border border-white/5">
                  <div className="font-mono font-bold text-sm text-emerald-400">{metrics.cache_hits}</div>
                  <div className="text-[10px] text-slate-400 uppercase">Cache Hits</div>
                </div>
                <div className="p-2 rounded bg-white/[0.02] border border-white/5">
                  <div className="font-mono font-bold text-sm text-sky-400">{metrics.cache_misses}</div>
                  <div className="text-[10px] text-slate-400 uppercase">Cache Misses</div>
                </div>
                <div className="p-2 rounded bg-white/[0.02] border border-white/5">
                  <div className="font-mono font-bold text-sm text-emerald-400">{metrics.llm_calls_avoided}</div>
                  <div className="text-[10px] text-slate-400 uppercase">LLM Avoided</div>
                </div>
                <div className="p-2 rounded bg-white/[0.02] border border-white/5">
                  <div className="font-mono font-bold text-xs text-white">{metrics.avg_hit_latency_ms.toFixed(1)} ms</div>
                  <div className="text-[10px] text-slate-400 uppercase">Avg Hit Time</div>
                </div>
                <div className="p-2 rounded bg-white/[0.02] border border-white/5">
                  <div className="font-mono font-bold text-xs text-white">{metrics.avg_miss_latency_ms.toFixed(1)} ms</div>
                  <div className="text-[10px] text-slate-400 uppercase">Avg Miss Time</div>
                </div>
              </div>

              <div className="pt-2 border-t border-white/5 flex items-center justify-between text-xs text-slate-400">
                <span>Estimated Cost Saved:</span>
                <span className="font-mono font-bold text-emerald-400">${metrics.estimated_cost_saved_usd.toFixed(4)}</span>
              </div>
            </div>

            {/* Invalidation Demo Controls */}
            <div className="rounded-xl border border-white/10 bg-white/[0.02] p-3.5 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Invalidation Demo</span>
                <span className="font-mono text-[10px] bg-white/5 text-slate-300 px-1.5 py-0.5 rounded">
                  KB: {kb.version_hash.substring(0, 7)}
                </span>
              </div>
              <p className="text-[11px] text-slate-400 leading-relaxed">
                Test Section 8 cache invalidation: updating the return policy dynamically re-hashes the KB and prevents stale Redis hits.
              </p>
              <div className="space-y-2">
                <button
                  onClick={handleUpdatePolicy}
                  className="w-full flex items-center justify-center gap-2 py-2 px-3 rounded-lg text-xs font-semibold bg-sky-500/10 text-sky-400 border border-sky-500/20 hover:bg-sky-500/20 transition-all"
                >
                  <RotateCw className="w-3.5 h-3.5" />
                  {policyExtended ? "Revert Policy (45d → 30d)" : "Update Policy (30d → 45d)"}
                </button>
                <button
                  onClick={handleClearCache}
                  className="w-full flex items-center justify-center gap-2 py-2 px-3 rounded-lg text-xs font-semibold bg-white/5 text-slate-400 border border-white/10 hover:text-red-400 hover:border-red-500/20 hover:bg-red-500/10 transition-all"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  Purge Redis Cache
                </button>
              </div>
            </div>

            {/* Knowledge Base Documents List */}
            <div className="rounded-xl border border-white/10 bg-white/[0.02] p-3.5 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Policy Documents</span>
                <span className="text-[10px] font-mono text-slate-400">{kb.total_chunks} Chunks</span>
              </div>
              <div className="max-h-40 overflow-y-auto space-y-1.5 pr-1">
                {kb.documents.map((doc) => (
                  <div key={doc.doc_id} className="p-2 rounded bg-white/[0.02] border border-white/5 flex items-center justify-between text-xs">
                    <span className="truncate max-w-[170px] text-slate-200 font-medium">{doc.title}</span>
                    <span className="text-[10px] font-mono text-slate-500">{doc.source_ref}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="p-3 border-t border-white/10 text-[11px] text-slate-500 space-y-1">
            <div className="flex justify-between">
              <span>Pipeline:</span>
              <span className="text-slate-400 font-semibold">BM25 + Dense + Cross-Encoder</span>
            </div>
            <div className="flex justify-between">
              <span>Eviction:</span>
              <span className="text-emerald-400 font-semibold">allkeys-lfu (Popular FAQs)</span>
            </div>
            <div className="flex justify-between">
              <span>Cache:</span>
              <span className={metrics.is_emulator ? "text-sky-400 font-semibold" : "text-emerald-400 font-semibold"}>
                {metrics.is_emulator ? "In-Memory (fakeredis)" : "Redis 6379 (Active)"}
              </span>
            </div>
          </div>
        </aside>

        {/* Main Chat Workspace */}
        <main className="flex-1 flex flex-col h-full bg-[#0E1526]/80 relative overflow-hidden">
          {/* Header */}
          <header className="h-16 px-6 border-b border-white/10 flex items-center justify-between bg-[#0B101D]/60 backdrop-blur-xl z-20">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-sky-500 to-indigo-600 flex items-center justify-center font-bold text-xs text-white shadow-md shadow-sky-500/20">
                AI
              </div>
              <div>
                <h2 className="font-semibold text-sm text-white">NovaCart Virtual Assistant</h2>
                <p className="text-[11px] text-slate-400">Grounded in official policy documentation · Redis accelerated</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={handleColdVsHitTest}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-sky-500/10 text-sky-400 border border-sky-500/20 hover:bg-sky-500/20 transition-all"
              >
                <Zap className="w-3.5 h-3.5" />
                ⚡ Test Cold vs Hit
              </button>
              <button
                onClick={() => setMessages([])}
                className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-white/5 text-slate-400 border border-white/10 hover:text-white hover:bg-white/10 transition-all"
              >
                Clear Chat
              </button>
            </div>
          </header>

          {/* Feed */}
          <div className="flex-1 overflow-y-auto p-6 pb-36 space-y-6">
            {messages.length === 0 && (
              <div className="max-w-xl mx-auto my-12 text-center space-y-6">
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold tracking-wider text-sky-400 bg-sky-500/10 border border-sky-500/20">
                  <Sparkles className="w-3.5 h-3.5" />
                  CUSTOMER SUPPORT PORTAL
                </div>
                <h2 className="text-3xl font-extrabold tracking-tight text-white">
                  How can we assist you today?
                </h2>
                <p className="text-sm text-slate-400 leading-relaxed max-w-md mx-auto">
                  Ask any question about refunds, shipping speeds, damaged deliveries, order cancellations, or account troubleshooting.
                </p>
                <div className="grid grid-cols-1 gap-2 text-left max-w-lg mx-auto pt-2">
                  {faqs.map((faq, i) => (
                    <button
                      key={i}
                      onClick={() => handleSend(faq.text)}
                      className="p-3 rounded-xl border border-white/5 bg-white/[0.02] hover:bg-sky-500/5 hover:border-sky-500/30 text-xs text-slate-300 transition-all flex items-center justify-between group"
                    >
                      <span>{faq.label}</span>
                      <span className="text-sky-400 opacity-0 group-hover:opacity-100 transition-opacity">→</span>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex gap-3 max-w-3xl ${msg.role === "user" ? "ml-auto justify-end" : "mr-auto"}`}
              >
                {msg.role === "assistant" && (
                  <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-sky-500 to-indigo-600 flex items-center justify-center font-bold text-xs text-white flex-shrink-0 shadow-md">
                    AI
                  </div>
                )}
                <div
                  className={`relative p-4 rounded-2xl text-sm leading-relaxed max-w-[85%] ${
                    msg.role === "user"
                      ? "bg-gradient-to-r from-sky-500 to-blue-600 text-white rounded-br-none shadow-lg shadow-sky-500/10"
                      : `bg-[#121B2E]/90 border ${
                          msg.cacheHit
                            ? "border-emerald-500/40 shadow-lg shadow-emerald-500/10"
                            : "border-white/10"
                        } text-slate-200 rounded-bl-none backdrop-blur-md`
                  }`}
                >
                  {msg.role === "assistant" && (
                    <div className="flex items-center justify-between mb-2 text-xs">
                      {msg.cacheHit ? (
                        msg.cacheType === "semantic" ? (
                          <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full font-bold text-[11px] text-purple-300 bg-purple-500/15 border border-purple-500/30">
                            <Zap className="w-3 h-3 text-purple-400" />
                            🎯 Semantic Hit · {msg.latencyMs?.toFixed(1)}ms · Equivalence ({Math.round((msg.similarityScore || 0.9) * 100)}%)
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full font-bold text-[11px] text-emerald-400 bg-emerald-500/10 border border-emerald-500/30">
                            <Zap className="w-3 h-3" />
                            ⚡ Exact Hit · {msg.latencyMs?.toFixed(1)}ms · Redis (LLM Skipped)
                          </span>
                        )
                      ) : (
                        <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full font-medium text-[11px] text-slate-400 bg-white/5 border border-white/10">
                          <Search className="w-3 h-3" />
                          Cache Miss · {msg.latencyMs?.toFixed(1)}ms · Hybrid + Cross-Encoder
                        </span>
                      )}
                      <span className="font-mono text-[10px] text-slate-500 ml-3">{msg.timestamp}</span>
                    </div>
                  )}

                  <div className="whitespace-pre-wrap">{msg.content}</div>

                  {/* Sources section */}
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-white/10 space-y-1.5">
                      <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 block">
                        📚 Verified Source Citations ({msg.sources.length})
                      </span>
                      <div className="flex flex-wrap gap-1.5">
                        {msg.sources.map((src, idx) => (
                          <div
                            key={idx}
                            title={src.text_snippet}
                            className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium bg-sky-500/10 text-sky-400 border border-sky-500/20"
                          >
                            <FileText className="w-3 h-3" />
                            <span>{src.title}</span>
                            <span className="text-[9px] opacity-70">§ {src.chunk_id}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {msg.role === "assistant" && (
                    <button
                      onClick={() => copyToClipboard(msg.content, msg.id)}
                      className="absolute top-3 right-3 p-1 rounded hover:bg-white/10 text-slate-500 hover:text-slate-300 transition-colors"
                      title="Copy response"
                    >
                      {copiedId === msg.id ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                    </button>
                  )}
                </div>
              </div>
            ))}

            {loading && (
              <div className="flex gap-3 max-w-3xl mr-auto">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-sky-500 to-indigo-600 flex items-center justify-center font-bold text-xs text-white flex-shrink-0">
                  AI
                </div>
                <div className="p-3.5 rounded-2xl bg-[#121B2E] border border-white/10 text-sm text-slate-400 flex items-center gap-3">
                  <div className="flex items-center gap-1">
                    <span className="w-1.5 h-3 bg-sky-400 rounded animate-pulse"></span>
                    <span className="w-1.5 h-4 bg-sky-400 rounded animate-pulse delay-100"></span>
                    <span className="w-1.5 h-2 bg-sky-400 rounded animate-pulse delay-200"></span>
                  </div>
                  <span>Searching knowledge base &amp; reasoning...</span>
                </div>
              </div>
            )}

            <div ref={chatEndRef} />
          </div>

          {/* Floating Input */}
          <div className="absolute bottom-0 left-0 right-0 p-6 bg-gradient-to-t from-[#070A10] via-[#070A10]/90 to-transparent flex justify-center z-20">
            <div className="w-full max-w-3xl rounded-2xl border border-white/10 bg-[#121B2E]/95 p-2 px-4 shadow-2xl backdrop-blur-xl flex items-center gap-3 focus-within:border-sky-500/50 transition-all">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSend();
                  }
                }}
                placeholder="Ask about returns, shipping speeds, damaged goods, payment issues..."
                className="flex-1 bg-transparent border-none outline-none text-sm text-slate-100 placeholder-slate-500"
              />
              <button
                onClick={() => handleSend()}
                disabled={loading || !input.trim()}
                className="w-9 h-9 rounded-full bg-gradient-to-r from-sky-500 to-indigo-600 flex items-center justify-center text-white disabled:opacity-40 disabled:cursor-not-allowed hover:scale-105 active:scale-95 transition-all shadow-md shadow-sky-500/20"
              >
                <Send className="w-4 h-4" />
              </button>
            </div>
          </div>
        </main>
      </div>

      {/* Floating Toast Notification */}
      {toastMessage && (
        <div className="fixed bottom-20 right-6 px-4 py-2.5 rounded-xl bg-slate-900/95 border border-sky-500/30 text-xs font-semibold text-slate-200 shadow-2xl backdrop-blur-md z-50 animate-bounce">
          {toastMessage}
        </div>
      )}
    </>
  );
}

