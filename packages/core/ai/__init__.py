"""AI advisor and approval gate modules."""

from importlib import import_module
from typing import Any

from packages.core.ai.advisor import AIAdvisor, AIProposal, get_ai_advisor
from packages.core.ai.approval_gate import ApprovalGate, ApprovalGateError, get_approval_gate

__all__ = [
    "AIAdvisor",
    "AIProposal",
    "ApprovalGate",
    "ApprovalGateError",
    "DRLOptimizer",
    "DRLProposal",
    "get_ai_advisor",
    "get_approval_gate",
]


def __getattr__(name: str) -> Any:
    """Lazy-load DRL optimizer symbols to avoid heavy import at module load."""
    if name in {"DRLOptimizer", "DRLProposal"}:
        module = import_module("packages.core.ai.drl_optimizer")
        return getattr(module, name)
    raise AttributeError(name)
