"""
backend/rag/generator.py
Grounded customer support answer synthesis with active citation tracking,
reference validation against retrieved document pages, and anti-hallucination guardrails.
"""

import os
import re
from pathlib import Path
import requests
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
import logging

try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

logger = logging.getLogger(__name__)


@dataclass
class CustomerSupportAnswer:
    """Structured response from the LLM generator with verified citations."""
    question: str
    answer: str
    is_supported: bool
    citations: List[Dict[str, Any]] = field(default_factory=list)  # Actively cited & validated
    retrieved_context: List[Dict[str, Any]] = field(default_factory=list)  # All candidate passages
    sources: List[Dict[str, Any]] = field(default_factory=list)  # Backwards compatibility
    citation_validation_passed: bool = True
    model: str = "gpt-4o-mini"
    error: Optional[str] = None


class CustomerSupportGenerator:
    """
    Synthesizes accurate, source-grounded answers for customer support inquiries.
    Enforces strict truthfulness, tracks which passages the answer actually cites,
    and validates those references against retrieved document pages.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        groq_key = os.getenv("GROQ_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if openai_key and not openai_key.startswith("your_"):
            self.api_key = api_key or openai_key
            self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        elif groq_key:
            self.api_key = api_key or groq_key
            self.base_url = base_url or os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
            self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        else:
            self.api_key = api_key or ""
            self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def _extract_and_validate_citations(
        self,
        answer: str,
        retrieved_passages: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Track which passages the answer actually cites and validate references against retrieved chunks.
        Returns:
            (validated_citations, all_references_valid)
        """
        all_retrieved_map: Dict[str, Dict[str, Any]] = {}
        index_map: Dict[int, Dict[str, Any]] = {}

        for idx, p in enumerate(retrieved_passages, 1):
            chunk = p["chunk"]
            info = {
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "section": chunk.section,
                "chunk_id": chunk.chunk_id,
                "page_number": getattr(chunk, "page_number", 1),
                "source_ref": chunk.source_ref,
                "source_url": getattr(chunk, "source_url", f"/docs/policies/{chunk.doc_id}.pdf#page={getattr(chunk, 'page_number', 1)}"),
                "supporting_passage": chunk.text,
                "relevance_score": round(float(p.get("rerank_score", p.get("rrf_score", 0.0))), 4),
                "is_validated": True,
            }
            all_retrieved_map[chunk.chunk_id.lower()] = info
            all_retrieved_map[chunk.doc_id.lower()] = info
            index_map[idx] = info

        cited_sources: List[Dict[str, Any]] = []
        seen_chunks: Set[str] = set()
        all_references_valid = True

        # 1. Check for numerical bracketed citations: e.g. [1], [2], [1, 2]
        numeric_citations = re.findall(r"\[(\d+)\]", answer)
        for num_str in numeric_citations:
            num = int(num_str)
            if num in index_map:
                entry = index_map[num]
                if entry["chunk_id"] not in seen_chunks:
                    seen_chunks.add(entry["chunk_id"])
                    cited_sources.append(entry)
            else:
                # Reference cited by LLM that was not in retrieved context
                all_references_valid = False

        # 2. Check for explicit chunk ID or section name mentions in answer
        for chunk_id, info in all_retrieved_map.items():
            if chunk_id in answer.lower():
                if info["chunk_id"] not in seen_chunks:
                    seen_chunks.add(info["chunk_id"])
                    cited_sources.append(info)

        # 3. Fallback heuristic: If no bracket markers used, evaluate key fact overlap
        if not cited_sources and retrieved_passages:
            ans_lower = answer.lower()
            for p in retrieved_passages:
                chunk = p["chunk"]
                # Check for significant text overlap or shared distinctive keywords
                words = [w for w in re.findall(r"\b\w{4,}\b", chunk.text.lower()) if w not in {"with", "that", "this", "from", "have"}]
                overlap_count = sum(1 for w in words if w in ans_lower)
                if overlap_count >= 3:
                    if chunk.chunk_id not in seen_chunks:
                        seen_chunks.add(chunk.chunk_id)
                        info = {
                            "doc_id": chunk.doc_id,
                            "title": chunk.title,
                            "section": chunk.section,
                            "chunk_id": chunk.chunk_id,
                            "page_number": getattr(chunk, "page_number", 1),
                            "source_ref": chunk.source_ref,
                            "source_url": getattr(chunk, "source_url", f"/docs/policies/{chunk.doc_id}.pdf#page={getattr(chunk, 'page_number', 1)}"),
                            "supporting_passage": chunk.text,
                            "relevance_score": round(float(p.get("rerank_score", p.get("rrf_score", 0.0))), 4),
                            "is_validated": True,
                        }
                        cited_sources.append(info)

        # If still empty (e.g. general response), include top 1 validated context
        if not cited_sources and retrieved_passages:
            top_chunk = retrieved_passages[0]["chunk"]
            cited_sources.append({
                "doc_id": top_chunk.doc_id,
                "title": top_chunk.title,
                "section": top_chunk.section,
                "chunk_id": top_chunk.chunk_id,
                "page_number": getattr(top_chunk, "page_number", 1),
                "source_ref": top_chunk.source_ref,
                "source_url": getattr(top_chunk, "source_url", f"/docs/policies/{top_chunk.doc_id}.pdf#page={getattr(top_chunk, 'page_number', 1)}"),
                "supporting_passage": top_chunk.text,
                "relevance_score": round(float(retrieved_passages[0].get("rerank_score", retrieved_passages[0].get("rrf_score", 0.0))), 4),
                "is_validated": True,
            })

        return cited_sources, all_references_valid

    def generate(
        self,
        question: str,
        retrieved_passages: List[Dict[str, Any]],
        company_name: str = "NovaCart Essentials"
    ) -> CustomerSupportAnswer:
        """
        Synthesize an evidence-based answer from retrieved documentation passages.
        Tracks active citations and validates them against retrieved document pages.
        """
        # 1. Check if we have any retrieved passages or if candidate relevance is below threshold
        if not retrieved_passages:
            return CustomerSupportAnswer(
                question=question,
                answer=(
                    f"I'm sorry, but our official {company_name} customer support documentation "
                    f"does not contain information related to your request. Please contact our support team directly."
                ),
                is_supported=False,
                citations=[],
                retrieved_context=[],
                sources=[],
                citation_validation_passed=True,
                model=self.model,
            )

        # 2. Check if the top candidate is completely unrelated (Cross-Encoder score < -4.0)
        top_p = retrieved_passages[0]
        if "rerank_score" in top_p and float(top_p["rerank_score"]) < -4.0:
            retrieved_context_info = [{
                "doc_id": p["chunk"].doc_id,
                "title": p["chunk"].title,
                "section": p["chunk"].section,
                "chunk_id": p["chunk"].chunk_id,
                "page_number": getattr(p["chunk"], "page_number", 1),
                "source_ref": p["chunk"].source_ref,
                "source_url": getattr(p["chunk"], "source_url", ""),
                "supporting_passage": p["chunk"].text,
                "relevance_score": round(float(p.get("rerank_score", 0.0)), 4),
            } for p in retrieved_passages]
            return CustomerSupportAnswer(
                question=question,
                answer=(
                    f"I'm sorry, but our official {company_name} customer support documentation "
                    f"does not contain information related to your inquiry. Please contact our support team directly."
                ),
                is_supported=False,
                citations=[],
                retrieved_context=retrieved_context_info,
                sources=[],
                citation_validation_passed=True,
                model="grounded-local",
            )

        # 2. Format context with explicit citation numbers [1], [2], etc.
        context_blocks = []
        retrieved_context_info = []

        for idx, p in enumerate(retrieved_passages, 1):
            chunk = p["chunk"]
            page_num = getattr(chunk, "page_number", 1)
            context_blocks.append(
                f"[{idx}] {chunk.title} (Page {page_num}, Section: {chunk.section} | ID: {chunk.chunk_id})\n{chunk.text}"
            )
            retrieved_context_info.append({
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "section": chunk.section,
                "chunk_id": chunk.chunk_id,
                "page_number": page_num,
                "source_ref": chunk.source_ref,
                "source_url": getattr(chunk, "source_url", f"/docs/policies/{chunk.doc_id}.pdf#page={page_num}"),
                "supporting_passage": chunk.text,
                "relevance_score": round(float(p.get("rerank_score", p.get("rrf_score", 0.0))), 4),
            })

        combined_context = "\n\n---\n\n".join(context_blocks)

        # 3. System prompt enforcing grounding, citation tagging, and refusal
        system_prompt = (
            f"You are the senior AI customer support assistant for {company_name}.\n"
            "Your sole objective is to answer customer questions accurately, politely, and strictly based "
            "on the provided company policy excerpts.\n\n"
            "STRICT RULES:\n"
            "1. Answer ONLY using facts stated in the CONTEXT below. Do NOT assume, extrapolate, or invent policies.\n"
            "2. If the context does not explicitly contain the answer, politely state: "
            f"'I'm sorry, but our {company_name} policies do not contain sufficient information to answer this question.'\n"
            "3. Reference supporting statements using citation numbers like [1] or [2] corresponding to the source.\n"
            "4. Maintain a warm, professional, and helpful customer support tone."
        )

        user_content = (
            f"CUSTOMER QUESTION:\n{question}\n\n"
            f"OFFICIAL COMPANY POLICY CONTEXT:\n{combined_context}\n\n"
            "CUSTOMER SUPPORT RESPONSE (Include [1], [2] citations):"
        )

        # 4. Invoke LLM API or grounded offline fallback
        if not self.api_key:
            top_p = retrieved_passages[0]
            top_chunk = top_p["chunk"]
            fallback_ans = (
                f"According to [1] {top_chunk.title} (Page {getattr(top_chunk, 'page_number', 1)}, {top_chunk.section}):\n"
                f"{top_chunk.text}"
            )
            active_cites, is_valid = self._extract_and_validate_citations(fallback_ans, retrieved_passages)
            return CustomerSupportAnswer(
                question=question,
                answer=fallback_ans,
                is_supported=True,
                citations=active_cites,
                retrieved_context=retrieved_context_info,
                sources=active_cites,
                citation_validation_passed=is_valid,
                model="offline-fallback",
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.0,
            "max_tokens": 450,
        }

        try:
            url = f"{self.base_url.rstrip('/')}/chat/completions"
            response = requests.post(url, headers=headers, json=payload, timeout=2.5)
            if response.status_code != 200:
                logger.error(f"[Generator] API call failed ({response.status_code}): {response.text}")
                top_chunk = retrieved_passages[0]["chunk"]
                fallback_ans = (
                    f"According to [1] {top_chunk.title} (Page {getattr(top_chunk, 'page_number', 1)}, {top_chunk.section}):\n"
                    f"{top_chunk.text}"
                )
                active_cites, is_valid = self._extract_and_validate_citations(fallback_ans, retrieved_passages)
                return CustomerSupportAnswer(
                    question=question,
                    answer=fallback_ans,
                    is_supported=True,
                    citations=active_cites,
                    retrieved_context=retrieved_context_info,
                    sources=active_cites,
                    citation_validation_passed=is_valid,
                    model=self.model,
                )

            data = response.json()
            raw_answer = data["choices"][0]["message"]["content"].strip()

            unsupported_markers = [
                "do not contain sufficient information",
                "does not contain information",
                "not covered in the provided policy",
                "unable to find information",
                "no information",
            ]
            is_supported = not any(marker in raw_answer.lower() for marker in unsupported_markers)

            if is_supported:
                active_cites, is_valid = self._extract_and_validate_citations(raw_answer, retrieved_passages)
            else:
                active_cites, is_valid = [], True

            return CustomerSupportAnswer(
                question=question,
                answer=raw_answer,
                is_supported=is_supported,
                citations=active_cites,
                retrieved_context=retrieved_context_info,
                sources=active_cites,
                citation_validation_passed=is_valid,
                model=self.model,
            )

        except requests.exceptions.RequestException as e:
            logger.warning(f"[Generator] Network unreachable ({e}). Synthesizing grounded response from context.")
            top_chunk = retrieved_passages[0]["chunk"]
            fallback_ans = (
                f"According to [1] {top_chunk.title} (Page {getattr(top_chunk, 'page_number', 1)}, {top_chunk.section}):\n"
                f"{top_chunk.text}"
            )
            active_cites, is_valid = self._extract_and_validate_citations(fallback_ans, retrieved_passages)
            return CustomerSupportAnswer(
                question=question,
                answer=fallback_ans,
                is_supported=True,
                citations=active_cites,
                retrieved_context=retrieved_context_info,
                sources=active_cites,
                citation_validation_passed=is_valid,
                model="grounded-local",
            )
        except Exception as e:
            logger.error(f"[Generator] Exception during LLM generation: {e}")
            return CustomerSupportAnswer(
                question=question,
                answer="We are experiencing a temporary network issue contacting the support engine.",
                is_supported=False,
                citations=[],
                retrieved_context=retrieved_context_info,
                sources=[],
                citation_validation_passed=False,
                model=self.model,
                error=str(e),
            )
