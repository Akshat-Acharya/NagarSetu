"""
Citizen-facing complaint routes.

POST /api/complaints/analyze — analyze an image using NagarSetu Vision
POST /api/complaints          — file a new complaint
GET  /api/complaints/mine     — get citizen complaints
GET  /api/complaints/{ref}    — get a complaint by reference
"""

import logging
import os
from typing import Optional

import httpx
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.middleware.auth import get_current_user
from app.models.user import User
from app.schemas.complaint import (
    ComplaintCategory,
    ComplaintCreate,
    ComplaintOut,
)
from app.services.complaint_service import (
    create_complaint,
    get_complaint_by_ref,
    list_complaints_for_citizen,
    save_image,
)

router = APIRouter(prefix="/api/complaints", tags=["Complaints"])
logger = logging.getLogger("nagarsetu.complaints")

AI_SERVICE_URL = os.getenv("AI_SERVICE_URL", "http://127.0.0.1:8001").rstrip("/")
AI_TIMEOUT = 15.0
CONFIDENCE_THRESHOLD = 0.70

_AI_LABEL_TO_CATEGORY = {
    "Garbage": ComplaintCategory.Garbage,
    "Pothole": ComplaintCategory.Pothole,
    "Sewage": ComplaintCategory.Waterlogging,
    "Waterlogging": ComplaintCategory.Waterlogging,
    "StreetLight": ComplaintCategory.StreetLight,
    "Other": ComplaintCategory.Other,
}

_AI_LABEL_TO_DEPT = {
    "Garbage": "Municipal Corporation",
    "Pothole": "PWD Department",
    "Sewage": "Municipal Corporation",
    "Waterlogging": "Municipal Corporation",
    "StreetLight": "Electricity Dept",
    "Other": "Municipal Corporation",
}


async def _predict_image_bytes(
    image_bytes: bytes,
    filename: str,
    content_type: str,
) -> Optional[dict]:
    """Forward image bytes to the standalone ML inference service."""

    if not image_bytes:
        return None

    try:
        async with httpx.AsyncClient(timeout=AI_TIMEOUT) as client:
            response = await client.post(
                f"{AI_SERVICE_URL}/predict",
                files={
                    "image": (
                        filename,
                        image_bytes,
                        content_type,
                    )
                },
            )

        if response.status_code != 200:
            logger.error(
                "AI service returned %s: %s",
                response.status_code,
                response.text,
            )
            return None

        result = response.json()
        logger.info(
            "ML prediction: class=%s confidence=%s priority=%s",
            result.get("predicted_class"),
            result.get("confidence"),
            result.get("priority"),
        )
        return result

    except httpx.TimeoutException:
        logger.warning("AI inference timed out.")
        return None
    except httpx.RequestError as exc:
        logger.error("Could not connect to AI service: %s", exc)
        return None
    except Exception:
        logger.exception("Unexpected AI inference error.")
        return None


async def _get_ai_prediction(image_path: str) -> Optional[dict]:
    """Read a saved complaint image and send it to the inference service."""

    full_path = os.path.join(settings.UPLOAD_DIR, image_path)

    if not os.path.exists(full_path):
        logger.warning("Image not found for AI prediction: %s", full_path)
        return None

    try:
        with open(full_path, "rb") as image_file:
            image_bytes = image_file.read()

        ext = os.path.splitext(full_path)[1].lower()
        mime = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }.get(ext, "application/octet-stream")

        return await _predict_image_bytes(
            image_bytes,
            os.path.basename(full_path),
            mime,
        )
    except Exception:
        logger.exception("Failed to read complaint image.")
        return None


def _build_image_url(
    request_base_url: str,
    image_path: Optional[str],
) -> Optional[str]:
    if not image_path:
        return None
    return f"{request_base_url.rstrip('/')}/uploads/{image_path}"


def _complaint_to_out(complaint, base_url: str) -> ComplaintOut:
    data = ComplaintOut.model_validate(complaint)
    data.image_url = _build_image_url(base_url, complaint.image_path)
    return data


@router.post("/analyze")
async def analyze_complaint_image(
    image: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """
    Analyse an uploaded image before complaint submission.
    This endpoint does not save an image or create a database record.
    """

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/bmp",
    }

    if image.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail="Unsupported image format. Use JPEG, PNG, WEBP, or BMP.",
        )

    image_bytes = await image.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image too large. Maximum size is 10 MB.",
        )

    result = await _predict_image_bytes(
        image_bytes,
        image.filename or "complaint_image",
        image.content_type,
    )

    if result is None:
        raise HTTPException(
            status_code=503,
            detail="AI inference service is unavailable. Please try again.",
        )

    return result


@router.post("", response_model=ComplaintOut, status_code=201)
async def file_complaint(
    request: Request,
    title: str = Form(...),
    description: str = Form(...),
    category: ComplaintCategory = Form(...),
    location: Optional[str] = Form(None),
    city: Optional[str] = Form(None),
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # 1. Save the uploaded image.
    image_path = await save_image(image)

    # 2. Run inference on the saved image. If AI is unavailable,
    #    the complaint can still be submitted with citizen-provided data.
    ai_result = await _get_ai_prediction(image_path)

    # 3. Start with the citizen-provided category.
    complaint_data = ComplaintCreate(
        title=title,
        description=description,
        category=category,
        location=location,
        city=city,
    )

    ai_confidence = None
    ai_severity = None
    ai_label = None
    needs_review = False
    ai_priority = None
    ai_dept = None

    # 4. Use the model prediction when available.
    if ai_result:
        ai_label = ai_result.get("predicted_class")
        ai_confidence = ai_result.get("confidence")
        ai_severity = ai_result.get("severity_score")
        ai_priority = ai_result.get("priority")
        needs_review = ai_confidence is not None and ai_confidence < CONFIDENCE_THRESHOLD

        logger.info(
            "AI result for complaint: label=%s confidence=%s priority=%s",
            ai_label,
            ai_confidence,
            ai_priority,
        )

        if (
            ai_label
            and ai_confidence is not None
            and ai_confidence >= CONFIDENCE_THRESHOLD
        ):
            mapped_category = _AI_LABEL_TO_CATEGORY.get(ai_label)

            if mapped_category is not None:
                complaint_data = ComplaintCreate(
                    title=title,
                    description=description,
                    category=mapped_category,
                    location=location,
                    city=city,
                )

            ai_dept = _AI_LABEL_TO_DEPT.get(ai_label)

    # 5. Persist the complaint.
    complaint = create_complaint(
        db=db,
        data=complaint_data,
        citizen=current_user,
        image_path=image_path,
        ai_confidence=ai_confidence,
        ai_severity=ai_severity,
        ai_label=ai_label,
        needs_review=needs_review,
        ai_priority=ai_priority,
        ai_dept=ai_dept,
    )

    return _complaint_to_out(complaint, str(request.base_url))


@router.get("/mine", response_model=list[ComplaintOut])
def my_complaints(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    complaints = list_complaints_for_citizen(db, current_user.id)
    return complaints


@router.get("/{ref}", response_model=ComplaintOut)
def get_complaint(
    ref: str,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    complaint = get_complaint_by_ref(db, ref)

    if (
        current_user.role == "citizen"
        and complaint.citizen_id != current_user.id
    ):
        raise HTTPException(status_code=403, detail="Not your complaint.")

    return _complaint_to_out(complaint, str(request.base_url))
