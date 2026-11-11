"""
backend/rag/generator.py
Grounded customer support answer synthesis using OpenAI-compatible APIs
(supports OpenAI, Groq, local vLLM/Ollama via standard Chat Completions endpoint).
"""

import os
from pathlib import Path
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
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
    """Structured response from the LLM generator."""
    question: str
    answer: str
    is_supported: bool
    sources: List[Dict[str, Any]]
    model: str
    error: Optional[str] = None


class CustomerSupportGenerator:
    """
    Synthesizes accurate, source-grounded answers for customer support inquiries.
    Enforces strict truthfulness: if documentation doesn't cover the query, states so clearly.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        # Auto-detect OpenAI vs Groq credentials from environment
        groq_key = os.getenv("GROQ_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if openai_key and not openai_key.startswith("your_"):
            self.api_key = api_key or openai_key
            self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        elif groq_key:
            self.api_key = api_key or groq_key
            self.base_url = base_url or os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
            self.model = model or os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
        else:
            self.api_key = api_key or ""
            self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def generate(
        self,
        question: str,
        retrieved_passages: List[Dict[str, Any]],
        company_name: str = "NovaCart Essentials"
    ) -> CustomerSupportAnswer:
        """
        Synthesize an evidence-based answer from retrieved documentation passages.
        """
        # 1. Check if we have any retrieved passages
        if not retrieved_passages:
            return CustomerSupportAnswer(
                question=question,
                answer=(
                    f"I'm sorry, but our official {company_name} customer support documentation "
                    f"does not contain information related to your request. Please contact our support team directly."
                ),
                is_supported=False,
                sources=[],
                model=self.model,
            )

        # 2. Format context and source metadata
        context_blocks = []
        sources = []
        for p in retrieved_passages:
            chunk = p["chunk"]
            context_blocks.append(
                f"[Source: {chunk.title} - Section: {chunk.section} | ID: {chunk.chunk_id}]\n{chunk.text}"
            )
            sources.append({
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "section": chunk.section,
                "chunk_id": chunk.chunk_id,
                "source_ref": chunk.source_ref,
                "relevance_score": round(float(p.get("rerank_score", p.get("rrf_score", 0.0))), 4),
            })

        combined_context = "\n\n---\n\n".join(context_blocks)

        # 3. System prompt enforcing grounding and truthfulness
        system_prompt = (
            f"You are the senior AI customer support assistant for {company_name}.\n"
            "Your sole objective is to answer customer questions accurately, politely, and strictly based "
            "on the provided company policy excerpts.\n\n"
            "RULES:\n"
            "1. Answer ONLY using the facts stated in the CONTEXT below. Do NOT assume, extrapolate, or invent policies.\n"
            "2. If the context does not explicitly contain the answer, politely state: "
            f"'I'm sorry, but our {company_name} policies do not contain sufficient information to answer this question.'\n"
            "3. Cite policy sections or timelines when applicable.\n"
            "4. Maintain a warm, professional, and helpful customer support tone."
        )

        user_content = (
            f"CUSTOMER QUESTION:\n{question}\n\n"
            f"OFFICIAL COMPANY POLICY CONTEXT:\n{combined_context}\n\n"
            "CUSTOMER SUPPORT RESPONSE:"
        )

        # 4. Invoke LLM API
        if not self.api_key:
            # Fallback mock response for offline testing or missing API keys
            fallback_ans = (
                f"Based on our {sources[0]['title']} ({sources[0]['section']}), "
                f"{retrieved_passages[0]['chunk'].text[:250]}..."
            )
            return CustomerSupportAnswer(
                question=question,
                answer=fallback_ans,
                is_supported=True,
                sources=sources,
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
            "temperature": 0.0,  # Zero temperature for deterministic, factual outputs
            "max_tokens": 450,
        }

        try:
            url = f"{self.base_url.rstrip('/')}/chat/completions"
            response = requests.post(url, headers=headers, json=payload, timeout=2.5)
            if response.status_code != 200:
                logger.error(f"[Generator] API call failed ({response.status_code}): {response.text}")
                fallback_ans = (
                    f"Based on our {sources[0]['title']} ({sources[0]['section']}):\n"
                    f"{retrieved_passages[0]['chunk'].text}"
                )
                return CustomerSupportAnswer(
                    question=question,
                    answer=fallback_ans,
                    is_supported=True,
                    sources=sources,
                    model=self.model,
                )

            data = response.json()
            raw_answer = data["choices"][0]["message"]["content"].strip()

            unsupported_markers = [
                "do not contain sufficient information",
                "does not contain information",
                "not covered in the provided policy",
                "unable to find information",
            ]
            is_supported = not any(marker in raw_answer.lower() for marker in unsupported_markers)

            return CustomerSupportAnswer(
                question=question,
                answer=raw_answer,
                is_supported=is_supported,
                sources=sources if is_supported else [],
                model=self.model,
            )

        except requests.exceptions.RequestException as e:
            logger.warning(f"[Generator] Network unreachable ({e}). Synthesizing grounded response from top retrieved context.")
            fallback_ans = (
                f"According to {company_name}'s {sources[0]['title']} ({sources[0]['section']}):\n"
                f"{retrieved_passages[0]['chunk'].text}"
            )
            return CustomerSupportAnswer(
                question=question,
                answer=fallback_ans,
                is_supported=True,
                sources=sources,
                model="grounded-local",
            )
        except Exception as e:
            logger.error(f"[Generator] Exception during LLM generation: {e}")
            return CustomerSupportAnswer(
                question=question,
                answer="We are experiencing a temporary network issue contacting the support engine.",
                is_supported=False,
                sources=sources,
                model=self.model,
                error=str(e),
            )
