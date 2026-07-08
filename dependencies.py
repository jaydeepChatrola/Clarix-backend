# pyrefly: ignore [missing-import]
from fastapi import Depends
from repositories.knowledge import KnowledgeRepository
from services.knowledge import KnowledgeService

def get_knowledge_repository() -> KnowledgeRepository:
    return KnowledgeRepository()

def get_knowledge_service(
    repository: KnowledgeRepository = Depends(get_knowledge_repository)
) -> KnowledgeService:
    return KnowledgeService(repository=repository)
