"""Agents domain — LLM-driven report and chat generation.

An :class:`Agent` base plus specialized agents (research, summarizer,
synthesis, citation, media, flashcard, quiz) coordinated by an orchestrator.
"""

from deepvision.agents.agent_orchestrator import AgentOrchestrator
from deepvision.agents.base import Agent, AgentContext
from deepvision.agents.citation_agent import CitationAgent
from deepvision.agents.media_agent import MediaAgent
from deepvision.agents.research_agent import ResearchAgent
from deepvision.agents.summarizer_agent import SummarizerAgent
from deepvision.agents.synthesis_agent import SynthesisAgent

__all__ = [
    "Agent",
    "AgentContext",
    "ResearchAgent",
    "SummarizerAgent",
    "SynthesisAgent",
    "CitationAgent",
    "MediaAgent",
    "AgentOrchestrator",
]
