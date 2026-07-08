from typing import Optional
import logging
from schemas.webhook import SupabaseWebhookPayload
from repositories.knowledge import KnowledgeRepository
from models.knowledge import KnowledgeFile

logger = logging.getLogger(__name__)

class KnowledgeService:
    def __init__(self, repository: KnowledgeRepository):
        self.repository = repository

    def process_webhook(self, payload: SupabaseWebhookPayload) -> Optional[KnowledgeFile]:
        """
        Process the incoming Supabase webhook event.
        """
        # Print the full JSON payload to the console
        print(f"\n--- Full Webhook Payload ---")
        print(payload.model_dump_json(indent=2))
        print(f"-----------------------------\n")

        if payload.table != "knowledge_files":
            logger.warning(f"Ignored webhook event for unsupported table: {payload.table}")
            return None

        record_data = payload.record
        if not record_data:
            logger.warning("Webhook payload contained no record data")
            return None

        # Validate the record data using the Pydantic model
        knowledge_file = KnowledgeFile(**record_data)
        file_id = str(knowledge_file.id)
        logger.info(f"Received webhook for file {file_id} with status '{knowledge_file.status}' (Event: {payload.type})")

        # Example business logic for INSERT/UPDATE
        if payload.type == "INSERT" and knowledge_file.status == "pending":
            logger.info(f"Transitioning file {file_id} to 'processing' status")
            try:
                self.repository.update_status(file_id, "processing")
            except Exception as e:
                logger.warning(f"Could not update status to processing in Supabase: {e}")
            
            file_name = knowledge_file.file_name or knowledge_file.name or "unknown"
            file_type = knowledge_file.type or (file_name.split('.')[-1] if '.' in file_name else "unknown")
            logger.info(f"Processing knowledge file '{file_name}' (type: {file_type}) for bot '{knowledge_file.bot_id}'")
            
            from core.config import ROOT_DIR
            from core.rag import ingest_file
            
            # If the specific file or storage path exists on disk, use it; otherwise fall back to static_knowledge.txt
            file_path_to_ingest = ROOT_DIR / "static_knowledge.txt"

            if file_path_to_ingest.exists():
                logger.info(f"Ingesting knowledge file from path: {file_path_to_ingest}")
                try:
                    ingest_file(file_id, str(file_path_to_ingest))
                    logger.info("Ingestion completed successfully.")
                except Exception as e:
                    logger.error(f"Failed to ingest file '{file_path_to_ingest}': {e}")
            else:
                logger.warning(f"Knowledge file not found at {file_path_to_ingest}")
                
            try:
                res = self.repository.update_status(file_id, "completed")
                if res:
                    return res
            except Exception as e:
                logger.warning(f"Could not update status to completed in Supabase: {e}")
            
            knowledge_file.status = "completed"
            return knowledge_file
        
        elif payload.type == "UPDATE":
            logger.info(f"Handling update for file {file_id}")
            try:
                res = self.repository.get_by_id(file_id)
                if res:
                    return res
            except Exception as e:
                logger.warning(f"Could not fetch file from Supabase: {e}")
            return knowledge_file

        return knowledge_file
