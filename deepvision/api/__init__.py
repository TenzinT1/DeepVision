"""HTTP API layer: schemas, dependencies, routers, and the FastAPI app.

``schemas.py`` is the wire contract; ``main.py`` wires the app together.
"""

from deepvision.api import deps, schemas

__all__ = ["schemas", "deps"]
