// frontend/src/types/index.ts

export interface SourceReference {
  doc_id: string;
  chunk_id: string;
  title: string;
  section: string;
  source_ref: string;
  text_snippet: string;
}

export interface ChatResponse {
  answer: string;
  cache_hit: boolean;
  latency_ms: number;
  source_references: SourceReference[];
  retrieval_method: string;
  reranked: boolean;
  kb_version: string;
}

export interface CacheMetrics {
  is_connected: boolean;
  is_emulator: boolean;
  total_questions: number;
  cache_hits: number;
  cache_misses: number;
  hit_rate_pct: number;
  avg_hit_latency_ms: number;
  avg_miss_latency_ms: number;
  llm_calls_avoided: number;
  estimated_cost_saved_usd: number;
  total_keys_in_redis: number;
  default_ttl_seconds: number;
}

export interface KnowledgeBaseDocument {
  doc_id: string;
  title: string;
  version: string;
  source_ref: string;
}

export interface KnowledgeBaseResponse {
  company: string;
  version_hash: string;
  total_chunks: number;
  documents: KnowledgeBaseDocument[];
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  cacheHit?: boolean;
  latencyMs?: number;
  sources?: SourceReference[];
}

