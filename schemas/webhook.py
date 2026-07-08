# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Literal

class SupabaseWebhookPayload(BaseModel):
    type: Literal["INSERT", "UPDATE", "DELETE"]
    table: str
    schema_name: str = Field(..., alias="schema")
    record: Optional[Dict[str, Any]] = None
    old_record: Optional[Dict[str, Any]] = None

    model_config = {
        "populate_by_name": True
    }
