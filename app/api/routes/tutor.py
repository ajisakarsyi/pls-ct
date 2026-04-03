"""
app/api/routes/tutor.py
────────────────────────
Endpoints that drive the AI tutoring loop.
"""

from fastapi import APIRouter, HTTPException, status

from app.core.cognitive import is_valid, DEFAULT_COGNITIVE_TYPE
from app.models.schemas import ChatRequest, ChatResponse, EvalRequest, EvalResponse
from app.services.tutor import generate_reply, evaluate_student_answer

router = APIRouter(tags=["Tutor"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a question and receive a personalised tutor response",
)
def chat(req: ChatRequest) -> ChatResponse:
    cognitive = req.cognitive.upper()
    if not is_valid(cognitive):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Cognitive type '{cognitive}' is not valid. See GET /cognitive-types.",
        )

    result = generate_reply(
        message=req.message,
        cognitive_code=cognitive,
        session_id=req.session_id,
    )
    return ChatResponse(**result)


@router.post(
    "/evaluate",
    response_model=EvalResponse,
    summary="Evaluate a student answer with adaptive scaffolded feedback",
)
def evaluate(req: EvalRequest) -> EvalResponse:
    cognitive = req.cognitive.upper()
    if not is_valid(cognitive):
        cognitive = DEFAULT_COGNITIVE_TYPE

    result = evaluate_student_answer(
        answer=req.answer.strip(),
        correct_answer=req.correct_answer,
        active_question=req.active_question,
        wrong_count=req.wrong_count,
        cognitive_code=cognitive,
        session_id=req.session_id,
    )
    return EvalResponse(**result)
