from pathlib import Path
from typing import Optional
# pyrefly: ignore [missing-import]
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT_DIR / ".env"

class Settings(BaseSettings):
    SUPABASE_URL: str = "https://uqjthfbembkyzmtfgzzg.supabase.co"
    SUPABASE_KEY: str = "sb_publishable_xEM5xheiOTeFRX1qODbV3g_LkJ5UqxE"
    
    OPENAI_API_KEY: str = ""
    OPENROUTER_API_KEY: str = "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"
    OPENAI_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENAI_MODEL: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    OPENAI_EMBED_MODEL: str = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
    HF_TOKEN: Optional[str] = "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"

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        extra="ignore",
    )


settings = Settings()