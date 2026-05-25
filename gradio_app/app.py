"""Gradio frontend for VADAR query interaction and run monitoring."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import gradio as gr
import pandas as pd
import requests


TITLE = "VADAR — Visual Agent for 3D Autonomous Reasoning"
DESCRIPTION = "Type a natural language 3D spatial query. VADAR synthesises and executes a Python program to answer it."


def _api_url() -> str:
    """Return API URL configured for Gradio app requests."""
    return os.getenv("VADAR_API_URL", "http://localhost:8000")


def submit_query(query: str, scene_id: str) -> Tuple[Any, str, Dict[str, Any]]:
    """Submit a query to backend and return result, code, and metadata."""
    response = requests.post(
        f"{_api_url().rstrip('/')}/query",
        json={"query": query, "scene_id": scene_id or "default"},
        timeout=30,
    )
    payload = response.json()

    if response.status_code != 200:
        return {"error": payload}, "", {"status_code": response.status_code}

    metadata = {
        "latency_ms": payload.get("latency_ms"),
        "iterations": payload.get("iterations"),
        "run_id": payload.get("run_id"),
        "success": payload.get("success"),
        "failure_reason": payload.get("failure_reason"),
    }
    return payload.get("result"), payload.get("synthesised_program", ""), metadata


def fetch_recent_runs() -> pd.DataFrame:
    """Fetch recent runs and map to a dataframe for display."""
    response = requests.get(f"{_api_url().rstrip('/')}/runs", timeout=15)
    payload: List[Dict[str, Any]] = response.json() if response.status_code == 200 else []
    return pd.DataFrame(payload[:20])


def fetch_eval_summary() -> Dict[str, Any]:
    """Fetch eval summary metrics from backend."""
    response = requests.get(f"{_api_url().rstrip('/')}/eval/summary", timeout=15)
    if response.status_code != 200:
        return {"error": "Failed to fetch eval summary"}
    return response.json()


def build_interface() -> gr.Blocks:
    """Build and return full Gradio interface."""
    with gr.Blocks(title=TITLE) as demo:
        gr.Markdown(f"# {TITLE}")
        gr.Markdown(DESCRIPTION)

        with gr.Tab("Query"):
            query_input = gr.Textbox(
                label="3D spatial query",
                placeholder="Find all objects within 2 metres of the red chair",
            )
            scene_input = gr.Textbox(label="Scene ID (optional)", value="default")
            submit_button = gr.Button("Submit")

            gr.Examples(
                examples=[
                    ["How many objects are in the scene?", "default"],
                    ["Find the object closest to the origin", "default"],
                    ["Are any objects directly above the table?", "default"],
                    ["What is the largest object by volume?", "default"],
                    ["List all red objects sorted by distance to [0,0,0]", "default"],
                ],
                inputs=[query_input, scene_input],
            )

            result_box = gr.JSON(label="Result")
            program_box = gr.Code(language="python", label="Synthesised program")
            metadata_box = gr.JSON(label="Run metadata")

            submit_button.click(
                submit_query,
                inputs=[query_input, scene_input],
                outputs=[result_box, program_box, metadata_box],
            )

        with gr.Tab("Recent runs"):
            runs_button = gr.Button("Refresh runs")
            runs_table = gr.Dataframe(label="Last 20 runs")
            runs_button.click(fetch_recent_runs, outputs=[runs_table])

        with gr.Tab("Eval summary"):
            eval_button = gr.Button("Refresh eval summary")
            eval_box = gr.JSON(label="Evaluation metrics")
            eval_button.click(fetch_eval_summary, outputs=[eval_box])

    return demo


if __name__ == "__main__":
    build_interface().launch(server_name="0.0.0.0", server_port=7860)
