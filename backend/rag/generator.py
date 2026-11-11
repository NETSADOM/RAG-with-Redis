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
        Accepts explicit bracketed citations [1], [2], etc. as instructed in the prompt.
        Deduplicates valid references in order of appearance and flags out-of-range numbers.
        Returns:
            (validated_citations, all_references_valid)
        """
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
                "source_url": getattr(chunk, "source_url", ""),
                "supporting_passage": chunk.text,
                "relevance_score": round(float(p.get("rerank_score", p.get("rrf_score", 0.0))), 4),
                "is_validated": True,
            }
            index_map[idx] = info

        cited_sources: List[Dict[str, Any]] = []
        seen_chunks: Set[str] = set()
        all_references_valid = True

        # Extract explicit numerical bracketed citations: e.g. [1], [2]
        numeric_citations = re.findall(r"\[(\d+)\]", answer)
        for num_str in numeric_citations:
            num = int(num_str)
            if num in index_map:
                entry = index_map[num]
                if entry["chunk_id"] not in seen_chunks:
                    seen_chunks.add(entry["chunk_id"])
                    cited_sources.append(entry)
            else:
                # Flag out-of-range or hallucinated citation number
                all_references_valid = False

        return cited_sources, all_references_valid

    def _select_best_supporting_passage(
        self,
        question: str,
        retrieved_passages: List[Dict[str, Any]],
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Identify candidate passage with strongest keyword and relevance alignment.
        """
        if not retrieved_passages:
            return 1, {}
        q_tokens = set(re.findall(r"\b\w{3,}\b", question.lower())) - {
            "what", "when", "where", "which", "how", "the", "for", "and",
            "does", "can", "are", "with", "from", "that", "this", "have"
        }
        best_idx = 1
        best_passage = retrieved_passages[0]
        best_score = -999.0

        for idx, p in enumerate(retrieved_passages, 1):
            chunk = p["chunk"]
            combined = f"{chunk.title} {chunk.section} {chunk.text}".lower()
            match_count = sum(1 for tok in q_tokens if tok in combined)
            rel_score = float(p.get("rerank_score", p.get("rrf_score", 0.0)))
            total_score = match_count * 10.0 + rel_score
            if total_score > best_score:
                best_score = total_score
                best_idx = idx
                best_passage = p

        return best_idx, best_passage

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
            best_idx, best_p = self._select_best_supporting_passage(question, retrieved_passages)
            best_chunk = best_p["chunk"]
            fallback_ans = (
                f"According to [{best_idx}] {best_chunk.title} (Page {getattr(best_chunk, 'page_number', 1)}, {best_chunk.section}):\n"
                f"{best_chunk.text}"
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
                best_idx, best_p = self._select_best_supporting_passage(question, retrieved_passages)
                top_chunk = best_p["chunk"]
                fallback_ans = (
                    f"According to [{best_idx}] {top_chunk.title} (Page {getattr(top_chunk, 'page_number', 1)}, {top_chunk.section}):\n"
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
            best_idx, best_p = self._select_best_supporting_passage(question, retrieved_passages)
            top_chunk = best_p["chunk"]
            fallback_ans = (
                f"According to [{best_idx}] {top_chunk.title} (Page {getattr(top_chunk, 'page_number', 1)}, {top_chunk.section}):\n"
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
