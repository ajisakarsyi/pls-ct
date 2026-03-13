"""
app/api/routes/reference.py
────────────────────────────
Read-only reference endpoints (no side effects).
"""

from fastapi import APIRouter

from app.core.cognitive import VALID_COGNITIVE_TYPES, cognitive_label
from app.models.schemas import CognitiveTypeItem, CognitiveTypesResponse

router = APIRouter(tags=["Reference"])


@router.get(
    "/cognitive-types",
    response_model=CognitiveTypesResponse,
    summary="Daftar semua 48 kode tipe kognitif yang valid",
)
def list_cognitive_types() -> CognitiveTypesResponse:
    return CognitiveTypesResponse(
        cognitive_types=[
            CognitiveTypeItem(code=ct, label=cognitive_label(ct))
            for ct in VALID_COGNITIVE_TYPES
        ]
    )
