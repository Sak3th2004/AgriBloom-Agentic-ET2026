"""
AgriBloom Agentic - Main Entry Point
Multi-Agent Agricultural Advisory System using LangGraph

This module orchestrates the 5-agent pipeline:
1. Orchestrator - Routes requests and manages session
2. Vision - Crop disease detection using ViT
3. Knowledge - Weather, market, and agronomic data
4. Compliance - FSSAI/ICAR regulatory checks
5. Output - Multilingual voice and visual response
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

# Add project root to path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import agents
from agents.compliance_agent import run_compliance
from agents.knowledge_agent import run_knowledge
from agents.orchestrator_agent import run_orchestrator
from agents.output_agent import run_output
from agents.vision_agent import run_vision

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("agribloom.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("agribloom")


# Use plain dict for state — avoids Gradio schema parser issues with TypedDict
AgriState = dict


def _route_after_orchestrator(state: AgriState) -> str:
    """
    Determines which agent to invoke after orchestration.

    Routes to:
    - "vision" if an image is provided
    - "knowledge" for text-only queries
    """
    route = state.get("route", "vision")
    logger.debug(f"Routing decision: {route}")
    return route


def build_graph() -> Any:
    """
    Build the LangGraph state machine for the agent pipeline.

    Pipeline flow:
    orchestrator -> (vision | knowledge) -> knowledge -> compliance -> output -> END
    """
    logger.info("Building AgriBloom agent graph...")

    graph = StateGraph(AgriState)

    # Add all agent nodes
    graph.add_node("orchestrator", run_orchestrator)
    graph.add_node("vision", run_vision)
    graph.add_node("knowledge", run_knowledge)
    graph.add_node("compliance", run_compliance)
    graph.add_node("output", run_output)

    # Set entry point
    graph.set_entry_point("orchestrator")

    # Add conditional routing from orchestrator
    graph.add_conditional_edges(
        "orchestrator",
        _route_after_orchestrator,
        {
            "vision_first": "vision",
            "vision": "vision",
            "knowledge_first": "knowledge",
            "knowledge": "knowledge",
        },
    )

    # Linear flow for remaining agents
    graph.add_edge("vision", "knowledge")
    graph.add_edge("knowledge", "compliance")
    graph.add_edge("compliance", "output")
    graph.add_edge("output", END)

    compiled = graph.compile()
    logger.info("Agent graph compiled successfully")

    return compiled


# Build graph at module load time
GRAPH = build_graph()


def run_pipeline(
    image: Any = None,
    image_path: str | None = None,
    user_text: str = "",
    user_language: str = "en",
    lang: str | None = None,
    offline: bool = False,
    lat: float = 17.3850,
    lon: float = 78.4867,
    allow_path_hints: bool = False,
    model_dir: str | None = None,
) -> dict[str, Any]:
    """
    Run the AgriBloom multi-agent pipeline.

    Args:
        image: PIL Image of crop leaf (optional)
        image_path: Path to image file (optional)
        user_text: User's text query
        user_language: Language code (en, hi, kn, te, ta)
        lang: Alternative language code parameter
        offline: Use offline mode (ONNX + cached data)
        lat: Latitude for location-based services
        lon: Longitude for location-based services
        allow_path_hints: Allow filename-based class hints (for demos)
        model_dir: Custom model directory path

    Returns:
        Final state dictionary with all outputs
    """
    start_time = time.time()

    # Normalize language code
    effective_lang = lang or user_language or "en"

    # Build initial state
    initial_state: AgriState = {
        "image": image,
        "image_path": image_path or "",
        "user_text": user_text,
        "user_language": effective_lang,
        "lang": effective_lang,
        "offline": offline,
        "lat": lat,
        "lon": lon,
        "chat_history": [],
        "status": "received",
        "allow_path_hints": allow_path_hints,
        "model_dir": model_dir or "",
    }

    logger.info(
        f"Pipeline started: lang={effective_lang}, offline={offline}, "
        f"has_image={image is not None}, text_len={len(user_text)}"
    )

    try:
        # Invoke the graph
        final_state = GRAPH.invoke(initial_state)

        elapsed = time.time() - start_time
        final_status = final_state.get("status", "unknown")

        logger.info(
            f"Pipeline completed: status={final_status}, "
            f"elapsed={elapsed:.2f}s"
        )

        # Add timing to state
        final_state["elapsed_seconds"] = elapsed

        return final_state

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Pipeline failed after {elapsed:.2f}s: {e}", exc_info=True)

        # Return error state
        return {
            **initial_state,
            "status": "error",
            "error": str(e),
            "elapsed_seconds": elapsed,
            "final_response": f"Error: {str(e)}",
        }


def main() -> None:
    """Main entry point - launches the Gradio UI."""
    logger.info("=" * 60)
    logger.info("AgriBloom Agentic - Starting Application")
    logger.info("=" * 60)

    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU detected: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            logger.warning("No GPU detected - running on CPU")
    except ImportError:
        logger.warning("PyTorch not available")

    # Launch UI
    from ui.app import launch_app
    launch_app(run_pipeline)


if __name__ == "__main__":
    main()
