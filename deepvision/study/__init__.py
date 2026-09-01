"""Study layer — the engine room behind the app's third top-level screen.

Split by responsibility, deliberately:

- :mod:`deepvision.study.session_scheduler` — the session scheduler. **Pure
  functions**: no database, no I/O, no global clock (``now`` is always a
  parameter). That is what makes the one piece of arithmetic the whole feature
  rests on exhaustively testable, and it is why the local-day boundary can be
  checked without a database.
- :mod:`deepvision.study.card_queries` — row↔model conversion and the SQL the
  deck/queue views run, plus the two quantities *derived* from the append-only
  review log (a card's strength, and how many distinct cards were studied
  today). Queue order cannot be an SQL ``ORDER BY`` any more — strength is not
  a column — so it sorts in Python; see ``due_rows`` for why that is acceptable
  at this scale and when it would stop being.
- :mod:`deepvision.study.deck_generator` — builds a deck from the paper's
  persisted report and upserts it on ``(paper_id, content_key)`` so a
  regeneration preserves the schedule of every card that survives it.

Intentionally free of imports at package level: ``deck_generator`` pulls in the
report/agent stack, and importing that transitively just to call
the scheduler would make the review path — which must never touch a model
— depend on the model stack.
"""

from __future__ import annotations

__all__: list[str] = []
