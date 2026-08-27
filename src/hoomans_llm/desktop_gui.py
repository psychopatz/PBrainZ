"""Compatibility entry point for HoomansLLM's native desktop panel.

The implementation lives in :mod:`hoomans_llm.gui`; this module keeps the
historical import path stable for launchers and downstream integrations.
"""

from .gui import HoomansLLMControlPanel

__all__ = ["HoomansLLMControlPanel"]
