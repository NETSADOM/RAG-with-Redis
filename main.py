"""
main.py
Interactive terminal interface for the local PDF-based RAG application.
Allows users to ingest documents, inspect the vector store, ask questions,
and verify grounded answers with exact source document and page number citations.
"""

import sys
import os
from pathlib import Path

from src.config import config
from src.rag_pipeline import RAGPipeline


def print_banner():
    """Print the application startup banner."""
    print("\n" + "=" * 70)
    print("      LOCAL PDF-BASED RETRIEVAL-AUGMENTED GENERATION (RAG)")
    print("=" * 70)
    print(f" • Documents Directory : {config.documents_dir}")
    print(f" • Vector Database     : {config.chroma_db_dir} ('{config.collection_name}')")
    print(f" • Embedding Model     : {config.embedding_model_name} (384 dims)")
    if config.llm_provider == "groq":
        print(f" • LLM Provider        : Online Cloud via Groq (Model: {config.groq_model})")
    else:
        print(f" • LLM Provider        : Local Ollama (Model: {config.ollama_model} at {config.ollama_base_url})")
    print(f" • Relevance Cutoff    : {config.relevance_threshold} (Cosine distance)")
    print("=" * 70 + "\n")


def check_llm_status(pipeline: RAGPipeline) -> None:
    """Inspect and report LLM connectivity and available models."""
    provider_name = "Groq Cloud" if pipeline.generator.provider == "groq" else "Local Ollama"
    print(f"\n[System Check] Checking {provider_name} service...")
    if pipeline.generator.check_connection():
        models = pipeline.generator.list_available_models()
        print(f"  ✓ {provider_name} connection is ACTIVE.")
        print(f"  ✓ Target model '{pipeline.generator.model_name}' is ready to generate answers.")
        if models:
            print(f"  ✓ Available models: {', '.join(models[:6])}{'...' if len(models) > 6 else ''}")
    else:
        if pipeline.generator.provider == "groq":
            print(
                f"  ✗ Groq API is NOT reachable or API key is invalid.\n"
                f"    Please check GROQ_API_KEY in your .env file or verify internet connection."
            )
        else:
            print(
                f"  ✗ Ollama is NOT reachable at '{pipeline.generator.base_url}'.\n"
                "    To enable answer generation:\n"
                "    1. Install Ollama from https://ollama.com\n"
                "    2. Run 'ollama serve' (or open the Ollama desktop app)\n"
                f"    3. Run 'ollama pull {pipeline.generator.model_name}'"
            )


def handle_ask_question(pipeline: RAGPipeline, show_passages: bool) -> None:
    """Interactive loop for asking questions with multi-document scope selection."""
    db_count = pipeline.vector_store.count()
    if db_count == 0:
        print("\n[Notice] Vector database is currently empty.")
        print("Please ingest your documents first (Menu Option 1) before asking questions.\n")
        return

    docs = pipeline.list_documents()
    print("\n" + "-" * 70)
    print(" QUESTION ANSWERING MODE")
    print("-" * 70)
    print("Document Search Scope:")
    print("  [0] Search across ALL documents (default)")
    for idx, d in enumerate(docs, start=1):
        print(f"  [{idx}] {d['filename']} ({d['total_chunks']} chunks, {d['total_pages']} pages)")

    selected_scope = None
    scope_choice = input("\nSelect document scope [0 for all, or comma-separated e.g. 1,2]: ").strip()
    if scope_choice and scope_choice != "0":
        try:
            indices = [int(x.strip()) for x in scope_choice.split(",") if x.strip().isdigit()]
            valid_indices = [i for i in indices if 1 <= i <= len(docs)]
            if valid_indices:
                selected_scope = [docs[i - 1]["filename"] for i in valid_indices]
                print(f"-> Scoped search active for: {', '.join(selected_scope)}")
            else:
                print("-> Invalid selection. Searching across ALL documents.")
        except Exception:
            print("-> Invalid format. Searching across ALL documents.")
    else:
        print("-> Searching across ALL documents.")

    print("\nType your question and press Enter. (Type 'back' or 'exit' to return to menu)")
    print("-" * 70)

    while True:
        try:
            query = input("\nYour Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting to menu...")
            break

        if not query:
            continue

        if query.lower() in ["back", "exit", "quit", "q"]:
            break

        print("\nSearching relevant passages and generating answer...")
        result = pipeline.answer_question(query, document_ids=selected_scope)
        print(result.display(show_passages=show_passages))


def handle_ingestion(pipeline: RAGPipeline) -> None:
    """Trigger ingestion of PDF documents."""
    print(f"\nScanning '{config.documents_dir}' for PDF files...")
    force = False
    choice = input("Force re-index existing documents? [y/N]: ").strip().lower()
    if choice in ["y", "yes"]:
        force = True

    summary = pipeline.ingest_documents(force_reindex=force)
    print(f"\n[Summary] Pages: {summary['pages_extracted']} | Chunks: {summary['chunks_stored']} | Total DB Records: {summary['total_records']}")


def handle_list_documents(pipeline: RAGPipeline) -> None:
    """Display all documents currently stored in ChromaDB."""
    docs = pipeline.list_documents()
    print("\n" + "-" * 70)
    print(" INGESTED DOCUMENTS IN VECTOR DATABASE")
    print("-" * 70)
    if not docs:
        print("  (No documents currently indexed in ChromaDB)")
    else:
        for idx, d in enumerate(docs, start=1):
            pages_str = ", ".join(str(p) for p in d['indexed_pages'])
            print(f"  {idx}. {d['filename']}")
            print(f"     Total Chunks: {d['total_chunks']} | Total Pages: {d['total_pages']} | Indexed Pages: [{pages_str}]")
    print("-" * 70 + "\n")


def main():
    """Main CLI entrypoint."""
    print_banner()

    # Ensure storage folders exist
    config.ensure_directories()

    # Initialize RAG Pipeline
    print("[Pipeline] Initializing RAG components...")
    pipeline = RAGPipeline()

    show_passages = True

    while True:
        print("\n" + "=" * 50)
        print("                  MAIN MENU")
        print("=" * 50)
        print(" [1] Ingest PDF documents (from documents/ folder)")
        print(" [2] Ask a question")
        print(" [3] List currently indexed documents")
        print(f" [4] Toggle retrieved passage inspection (Current: {'ON' if show_passages else 'OFF'})")
        print(" [5] Check LLM provider status & models")
        print(" [6] Exit")
        print("=" * 50)

        try:
            choice = input("Select an option [1-6]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if choice == "1":
            handle_ingestion(pipeline)
        elif choice == "2":
            handle_ask_question(pipeline, show_passages=show_passages)
        elif choice == "3":
            handle_list_documents(pipeline)
        elif choice == "4":
            show_passages = not show_passages
            print(f"\n[Setting] Passage inspection is now: {'ON' if show_passages else 'OFF'}")
        elif choice == "5":
            check_llm_status(pipeline)
        elif choice == "6":
            print("\nThank you for using the PDF RAG application. Goodbye!")
            sys.exit(0)
        else:
            print("\n[Invalid] Please select a number between 1 and 6.")


if __name__ == "__main__":
    main()

