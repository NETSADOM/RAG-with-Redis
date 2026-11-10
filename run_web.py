"""
run_web.py
Entrypoint to launch the Minimalist PDF RAG web application.
Starts a high-performance local Uvicorn server and provides the clickable local URL.
"""

import sys
import uvicorn
from src.config import config


def main():
    config.ensure_directories()
    print("\n" + "=" * 65)
    print("       STARTING MINIMALIST PDF RAG WEB APPLICATION")
    print("=" * 65)
    print(" • Local Web URL   : http://127.0.0.1:8000")
    model_name = config.groq_model if config.llm_provider == "groq" else config.ollama_model
    print(f" • Active LLM      : {config.llm_provider.upper()} ({model_name})")
    print(f" • Vector Storage  : {config.chroma_db_dir}")
    print("=" * 65)
    print(" Open your browser and navigate to: http://127.0.0.1:8000")
    print(" Press CTRL+C in this terminal to stop the server.\n")

    # Start uvicorn server
    uvicorn.run(
        "src.web.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()

