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


def _build_active_graph() -> Any:
    """Pick the pipeline: V1 linear (default) or the V2 ReAct graph.

    Set ``AGRIBLOOM_USE_REACT=1`` to route through the dynamic ReAct
    orchestrator (graph/react_graph.py). Defaults to the proven linear graph so
    the winning V1 behaviour is never disturbed by accident.
    """
    import os

    if os.getenv("AGRIBLOOM_USE_REACT", "0") == "1":
        try:
            from graph.react_graph import build_react_graph

            logger.info("AGRIBLOOM_USE_REACT=1 → using ReAct dynamic graph")
            return build_react_graph()
        except Exception as e:
            logger.error("ReAct graph failed to build (%s) — falling back to linear", e)
    return build_graph()


# Build graph at module load time
GRAPH = _build_active_graph()


def build_initial_state(
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
    auto_detect_offline: bool = True,
) -> AgriState:
    """Build the seed pipeline state shared by every graph (V1 linear, V2 ReAct).

    Kept as a standalone function so ``run_pipeline`` (V1/Gradio) and
    ``backend.pipeline.run_v2_pipeline`` (the API) build the exact same shape
    of initial state and never drift apart.

    Automatic offline detection: an explicit ``offline=True`` is always
    honored. Otherwise, when ``auto_detect_offline`` is True (the default), a
    fast (~1.5s) network health check runs and — ONLY when the network is
    genuinely poor — upgrades the request to offline mode on its own, so a
    farmer on a bad connection isn't left waiting on doomed API calls. A fine
    network is never downgraded to offline; this only ever tightens, never
    loosens, the caller's request. Set ``auto_detect_offline=False`` to skip
    the check entirely (e.g. in tests, or when the caller already knows).
    """
    effective_lang = lang or user_language or "en"
    effective_offline = offline
    if not effective_offline and auto_detect_offline:
        from utils.network import should_force_offline

        effective_offline = should_force_offline(False)

    return {
        "image": image,
        "image_path": image_path or "",
        "user_text": user_text,
        "user_language": effective_lang,
        "lang": effective_lang,
        "offline": effective_offline,
        "lat": lat,
        "lon": lon,
        "chat_history": [],
        "status": "received",
        "allow_path_hints": allow_path_hints,
        "model_dir": model_dir or "",
    }


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

    initial_state = build_initial_state(
        image=image, image_path=image_path, user_text=user_text,
        user_language=user_language, lang=lang, offline=offline,
        lat=lat, lon=lon, allow_path_hints=allow_path_hints, model_dir=model_dir,
    )
    effective_lang = initial_state["lang"]
    effective_offline = initial_state["offline"]

    logger.info(
        f"Pipeline started: lang={effective_lang}, offline={effective_offline}"
        + (" (auto-detected: poor network)" if effective_offline and not offline else "")
        + f", has_image={image is not None}, text_len={len(user_text)}"
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
