# pyrefly: ignore [missing-import]
from fastapi import APIRouter, Depends, HTTPException, status
from schemas.webhook import SupabaseWebhookPayload
from services.knowledge import KnowledgeService
from dependencies import get_knowledge_service

router = APIRouter()

@router.post("/webhook/knowledge-file")
async def knowledge_file_webhook(
    payload: SupabaseWebhookPayload,
    service: KnowledgeService = Depends(get_knowledge_service)
):
    try:
        result = service.process_webhook(payload)
        if result is None:
            return {"status": "ignored"}
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing webhook: {str(e)}"
        )