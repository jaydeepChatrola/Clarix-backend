# pyrefly: ignore [missing-import]
from pydantic import BaseModel
from typing import Optional, Union

class KnowledgeFile(BaseModel):
    id: Union[str, int]
    status: str
    bot_id: Optional[str] = None
    file_name: Optional[str] = None
    storage_path: Optional[str] = None
    created_at: Optional[str] = None
    
    # Table fields
    name: Optional[str] = None
    type: Optional[str] = None
    size_bytes: Optional[int] = None
    uploaded_at: Optional[str] = None

    model_config = {
        "extra": "allow"
    }
