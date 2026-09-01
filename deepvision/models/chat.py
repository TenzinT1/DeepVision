"""Chat models — grounded Q&A over a single paper.

Chat answers are grounded strictly in one paper. Every assistant message can
carry citation references (resolving to the same citation popover as the report)
and figure references (opening the same lightbox). These models back
``POST /chat``.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from deepvision.models.report import Citation, MediaRef

__all__ = [
    "ChatRole",
    "AnswerKind",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
]


class ChatRole(str, Enum):
    """Author of a chat message."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class AnswerKind(str, Enum):
    """How an assistant answer was produced. Wire contract — never rename.

    Chat questions do not all want the same machinery, and treating them as if
    they did is what made the chat feel useless: "give me an APA citation" was
    being answered by semantic search over the paper's body text, which cannot
    possibly contain the answer, so the model invented one.

    - ``content`` — a normal grounded RAG answer over retrieved passages. The
      only kind that carries ``citations``.
    - ``citation`` — a formatted bibliographic citation, built **deterministically**
      from :class:`~deepvision.models.paper.PaperMeta` by
      ``report/citation_styles.py``. No model call, so it cannot be
      mis-punctuated or hallucinated.
    - ``metadata`` — a fact about the paper itself (authors, year, venue, id),
      read from ``PaperMeta`` rather than retrieved.
    - ``structure`` — a fact about the document (page count, figure count,
      chapters), read from the database.

    The UI uses this to render differently — a citation is offered as
    copyable text rather than as prose.
    """

    CONTENT = "content"
    CITATION = "citation"
    METADATA = "metadata"
    STRUCTURE = "structure"


class ChatMessage(BaseModel):
    """A single chat message.

    Assistant messages may include ``citations`` (grounding back-references) and
    ``figures`` (figure/table cards). User messages leave both empty.
    """

    model_config = ConfigDict(use_enum_values=False)

    id: str = Field(..., description="Message id (see utils.ids.message_id).")
    paper_id: str = Field(..., description="Paper this conversation is grounded in.")
    role: ChatRole = Field(..., description="user / assistant / system.")
    text: str = Field(default="", description="Message text.")
    answer_kind: Optional[AnswerKind] = Field(
        default=None,
        description=(
            "How an assistant answer was produced (see AnswerKind). None for "
            "user messages and for answers persisted before this field existed."
        ),
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Grounding citations attached to an assistant answer.",
    )
    figures: list[MediaRef] = Field(
        default_factory=list, description="Figure/table refs attached to an answer."
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ChatRequest(BaseModel):
    """Request body for ``POST /chat``."""

    paper_id: str = Field(..., description="Paper to ground the answer in.")
    message: str = Field(..., min_length=1, description="The user's question.")
    session_id: Optional[str] = Field(
        default=None,
        description="Optional chat-session id to continue an existing thread.",
    )
    history: list[ChatMessage] = Field(
        default_factory=list,
        description="Optional prior turns for multi-turn context (client-supplied).",
    )
    top_k: int = Field(
        default=6, ge=1, le=50, description="How many chunks to retrieve for grounding."
    )


class ChatResponse(BaseModel):
    """Response body for ``POST /chat`` — one grounded assistant answer."""

    session_id: str = Field(..., description="Chat-session id (new or continued).")
    message: ChatMessage = Field(..., description="The assistant's grounded answer.")
    retrieved_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Ids of chunks used for grounding (debug/traceability).",
    )
