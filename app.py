"""Hugging Face Spaces entry point launching FastAPI and Gradio together."""

from __future__ import annotations

import threading

import uvicorn

from backend.main import app as fastapi_app
from gradio_app.app import build_interface


def _run_fastapi() -> None:
    """Run FastAPI backend server on internal port 8000."""
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000, log_level="info")


def main() -> None:
    """Launch backend and Gradio UI in one process."""
    api_thread = threading.Thread(target=_run_fastapi, daemon=True)
    api_thread.start()

    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)


if __name__ == "__main__":
    main()
