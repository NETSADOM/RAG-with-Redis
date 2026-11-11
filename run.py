"""
run.py
Convenience runner for NovaCart Customer Support RAG System.
Starts the FastAPI application and serves both the REST API and the Frontend Dashboard.
"""

import sys
import uvicorn

if __name__ == "__main__":
    print("=" * 70)
    print("  Starting NovaCart Customer Support RAG Server")
    print("  Access Web Interface at: http://127.0.0.1:8000")
    print("  API Docs available at:   http://127.0.0.1:8000/docs")
    print("=" * 70)
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)

