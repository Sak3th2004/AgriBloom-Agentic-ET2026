"""
AgriBloom Agentic - Multi-Agent Agricultural Advisory System
5 Agents: Orchestrator, Vision, Knowledge, Compliance, Output
"""
# Lazy imports to avoid import errors during partial installs


def run_orchestrator(state):
    from agents.orchestrator_agent import run_orchestrator as _fn
    return _fn(state)


def run_vision(state):
    from agents.vision_agent import run_vision as _fn
    return _fn(state)


def run_knowledge(state):
    from agents.knowledge_agent import run_knowledge as _fn
    return _fn(state)


def run_compliance(state):
    from agents.compliance_agent import run_compliance as _fn
    return _fn(state)


def run_output(state):
    from agents.output_agent import run_output as _fn
    return _fn(state)


__all__ = [
    "run_orchestrator",
    "run_vision",
    "run_knowledge",
    "run_compliance",
    "run_output",
]
