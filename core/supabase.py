from typing import Optional
# pyrefly: ignore [missing-import]
from supabase import create_client, Client
from core.config import settings
import logging

logger = logging.getLogger(__name__)

supabase: Optional[Client] = None
if settings.SUPABASE_URL and settings.SUPABASE_KEY:
    try:
        supabase = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_KEY,
        )
    except Exception as e:
        logger.warning(f"Could not initialize Supabase client: {e}")
else:
    logger.warning("SUPABASE_URL or SUPABASE_KEY not provided. Supabase client initialized as None.")