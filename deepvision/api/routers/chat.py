"""Chat router — POST /chat.

Owned by: AGENTS domain (research/citation agents) + RAG (retrieval); wired
here by REPORT. Retrieves grounding chunks via a concrete
:class:`~deepvision.rag.retrieval.Retriever`, asks a concrete
:class:`~deepvision.agents.research_agent.ResearchAgent` for a grounded
answer, persists both turns as :class:`~deepvision.db.schema.ChatRow`\\ s, and
returns a single grounded :class:`ChatResponse`.

Chat is strictly single-paper grounded (``request.paper_id`` is required by
the schema). If no concrete Retriever/ResearchAgent is registered yet (or
either raises), this degrades to a best-effort answer built from whatever was
retrieved instead of failing the request outright.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from deepvision.agents.base import AgentContext
from deepvision.agents.chat_intent import Intent, classify
from deepvision.agents.citation_agent import CitationAgent
from deepvision.agents.media_agent import MediaAgent
from deepvision.agents.research_agent import ResearchAgent
from deepvision.api.deps import get_settings
from deepvision.api.schemas import ChatRequest, ChatResponse
from deepvision.db import session_scope
from deepvision.db.schema import ChatRow, PaperRow
from deepvision.models import ChatMessage, ChatRole
from deepvision.providers.factory import build_llm
from deepvision.report.agent_bridge import build_retriever
from deepvision.utils import get_logger
from deepvision.utils.ids import chat_id, message_id

router = APIRouter(tags=["chat"])

log = get_logger(__name__)

#: How many times `top_k` to actually retrieve for a content question. The agent
#: filters furniture out and then de-duplicates by meaning, so it needs a pool
#: several times the size of what it will finally use.
_OVERFETCH = 4


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Answer ``request.message`` grounded strictly in ``request.paper_id``."""
    clog = log.bind(paper_id=request.paper_id)

    with session_scope() as session:
        paper_exists = session.get(PaperRow, request.paper_id) is not None
    if not paper_exists:
        raise HTTPException(status_code=404, detail=f"paper not found: {request.paper_id}")

    settings = get_settings()
    session_id = request.session_id or chat_id()

    chunks: list = []
    retrieved_chunk_ids: list[str] = []
    intent = classify(request.message)
    try:
        # Skip retrieval entirely for a question the paper's *record* answers.
        # Embedding "cite this in APA" and searching the body text cannot
        # succeed — the text does not contain the paper's own bibliographic
        # record — and on the local model that pointless round trip is seconds
        # of latency before a deterministic answer that needed none of it.
        if intent is Intent.CONTENT:
            # ChunkRetriever needs (embeddings, vector_store) — build it via the
            # bridge helper rather than the no-arg resolve/instantiate path (which
            # could never construct it).
            retriever = build_retriever(settings)
            # Deliberately over-fetch: ResearchAgent filters page furniture out
            # and then keeps only chunks that say *different* things, so it needs
            # a pool to choose from. Fetching exactly top_k left it nothing to
            # select and two unrelated questions shared 4 of their 6 chunks.
            chunks = retriever.retrieve(
                request.message, request.paper_id, request.top_k * _OVERFETCH
            )
            retrieved_chunk_ids = [c.id for c in chunks]
    except Exception as exc:
        clog.error("retrieval failed", extra={"error": str(exc)})

    answer: ChatMessage | None = None
    try:
        # ResearchAgent is the concrete class named in; it takes
        # the LLM directly.
        from deepvision.config import get_config

        llm = build_llm(settings, strict=get_config().strict_providers)
        agent = ResearchAgent(llm)
        ctx = AgentContext(
            paper_id=request.paper_id,
            settings=settings,
            chunks=chunks,
            extra={
                "question": request.message,
                "history": request.history,
                "session_id": session_id,
                # The agent selects its final grounding set from the
                # over-fetched pool; this is the size it should cut down to.
                "top_k": request.top_k,
            },
        )
        answer = agent.run(ctx)
    except Exception as exc:
        clog.error("research agent failed; using fallback answer", extra={"error": str(exc)})

    if answer is None:
        answer = _fallback_answer(request, chunks)

    if not answer.id:
        answer.id = message_id()
    answer.paper_id = request.paper_id
    answer.role = ChatRole.ASSISTANT

    _persist_turn(session_id, request, answer)

    # Report the chunks the answer is actually grounded in, not the whole
    # over-fetched pool. The pool is `top_k * _OVERFETCH` wide and the agent
    # discards most of it, so returning all of it would make this traceability
    # field claim grounding that never happened. Citations carry the chunk the
    # agent really used; fall back to the pool only when there are none (a
    # record-answered question retrieves nothing at all, and correctly reports
    # an empty list).
    grounded_ids = [c.chunk_id for c in answer.citations if c.chunk_id]
    if grounded_ids:
        retrieved_chunk_ids = list(dict.fromkeys(grounded_ids))

    return ChatResponse(
        session_id=session_id,
        message=answer,
        retrieved_chunk_ids=retrieved_chunk_ids,
    )


def _fallback_answer(request: ChatRequest, chunks: list) -> ChatMessage:
    """Best-effort answer when the research agent isn't wired up yet.

    Never fails the request outright -- surfaces the most relevant retrieved
    excerpt (if any) instead of a grounded LLM answer, still attaching
    citations/figures when a concrete CitationAgent/MediaAgent is available.
    """
    if chunks:
        snippet = chunks[0].text[:400].strip()
        text = (
            "I found relevant passages, but the answer-generation agent isn't "
            "configured yet, so here is the most relevant excerpt instead:\n\n"
            f"> {snippet}"
        )
    else:
        text = (
            "I couldn't find relevant passages for that question, and the "
            "answer-generation agent isn't configured yet."
        )

    citations = []
    figures = []
    if chunks:
        try:
            citations = CitationAgent().cite(text, chunks)
        except Exception as exc:
            log.error("citation agent failed in chat fallback", extra={"error": str(exc)})
        try:
            figures = MediaAgent().build_media(chunks)[:3]
        except Exception as exc:
            log.error("media agent failed in chat fallback", extra={"error": str(exc)})

    return ChatMessage(
        id=message_id(),
        paper_id=request.paper_id,
        role=ChatRole.ASSISTANT,
        text=text,
        citations=citations,
        figures=figures,
    )


def _persist_turn(session_id: str, request: ChatRequest, answer: ChatMessage) -> None:
    """Persist both the user's question and the assistant's answer as ChatRows."""
    with session_scope() as session:
        session.add(
            ChatRow(
                id=message_id(),
                session_id=session_id,
                paper_id=request.paper_id,
                role=ChatRole.USER.value,
                text=request.message,
                citations=[],
                figures=[],
            )
        )
        session.add(
            ChatRow(
                id=answer.id,
                session_id=session_id,
                paper_id=request.paper_id,
                role=ChatRole.ASSISTANT.value,
                text=answer.text,
                citations=[c.model_dump(mode="json") for c in answer.citations],
                figures=[f.model_dump(mode="json") for f in answer.figures],
            )
        )
