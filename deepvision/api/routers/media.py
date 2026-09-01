"""Media router — GET /media/{chunk_id}.

Owned by: REPORT domain (media serving) backed by the local file store. Serves
raw image bytes for a figure/table/page chunk, or its metadata when
``?meta=1``. The chunk is resolved by scanning persisted reports for a
MediaRef/Citation carrying that ``chunk_id`` (see
:mod:`deepvision.report.interactive_sections`).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from deepvision.api.schemas import MediaMetadata
from deepvision.report.interactive_sections import DefaultMediaService

router = APIRouter(tags=["media"])

_service = DefaultMediaService()


@router.get(
    "/media/{chunk_id}",
    response_model=None,
    responses={
        200: {
            "content": {"image/png": {}, "application/json": {}},
            "description": "Raw image bytes, or MediaMetadata when ?meta=1.",
        }
    },
)
def get_media(
    chunk_id: str,
    meta: bool = Query(default=False, description="Return MediaMetadata JSON instead of bytes."),
) -> Response | MediaMetadata:
    """Serve the image bytes (or metadata) for media chunk ``chunk_id``."""
    try:
        if meta:
            return _service.get_metadata(chunk_id)
        data, content_type = _service.get_bytes(chunk_id)
        return Response(content=data, media_type=content_type)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
