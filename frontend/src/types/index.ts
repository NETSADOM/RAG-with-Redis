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
  cache_type?: string; // 'exact' | 'semantic' | 'none'
  similarity_score?: number;
  latency_ms: number;
  sources: SourceReference[];
  source_references?: SourceReference[];
  retrieval_method?: string;
  reranked?: boolean;
  knowledge_base_version?: string;
  kb_version?: string;
}

export interface CacheMetrics {
  is_connected: boolean;
  is_emulator: boolean;
  eviction_policy?: string;
  total_questions: number;
  cache_hits: number;
  exact_hits?: number;
  semantic_hits?: number;
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
  cacheType?: string;
  similarityScore?: number;
  latencyMs?: number;
  sources?: SourceReference[];
}

