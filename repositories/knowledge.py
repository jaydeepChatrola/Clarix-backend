from typing import Optional
from core.supabase import supabase
from models.knowledge import KnowledgeFile

class KnowledgeRepository:

    def update_status(self, file_id: str, status: str) -> Optional[KnowledgeFile]:
        if not supabase:
            return None
        response = (
            supabase
            .table("knowledge_files")
            .update({"status": status})
            .eq("id", file_id)
            .execute()
        )
        if response.data:
            return KnowledgeFile(**response.data[0])
        return None

    def get_by_id(self, file_id: str) -> Optional[KnowledgeFile]:
        if not supabase:
            return None
        response = (
            supabase
            .table("knowledge_files")
            .select("*")
            .eq("id", file_id)
            .execute()
        )
        if response.data:
            return KnowledgeFile(**response.data[0])
        return None