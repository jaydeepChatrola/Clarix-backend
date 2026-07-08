import os
import warnings
import logging
from functools import lru_cache
from typing import List, Optional
# pyrefly: ignore [missing-import]
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from core.config import settings

# Configure HuggingFace token or silence unauthenticated warning when token is not provided
if settings.HF_TOKEN:
    os.environ["HF_TOKEN"] = settings.HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = settings.HF_TOKEN
elif not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    warnings.filterwarnings("ignore", message=".*You are sending unauthenticated requests to the HF Hub.*")
    logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

_embedding_fn_instance: Optional[SentenceTransformerEmbeddingFunction] = None


def get_embedding_function() -> Optional[SentenceTransformerEmbeddingFunction]:
    """Singleton getter for SentenceTransformerEmbeddingFunction (all-MiniLM-L6-v2).
    Loads the model into memory only once on first use or import.
    """
    global _embedding_fn_instance
    if _embedding_fn_instance is None:
        try:
            logger.info("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
            _embedding_fn_instance = SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            logger.info("SentenceTransformer model 'all-MiniLM-L6-v2' loaded successfully.")
        except Exception as e:
            logger.error(f"Could not initialize SentenceTransformerEmbeddingFunction: {e}")
            _embedding_fn_instance = None
    return _embedding_fn_instance


@lru_cache(maxsize=10000)
def get_cached_embedding(text: str) -> List[float]:
    """Get embedding for a single text with in-memory LRU caching to prevent
    re-computing vector embeddings for previously processed text or frequent queries.
    """
    fn = get_embedding_function()
    if not fn:
        raise RuntimeError("SentenceTransformerEmbeddingFunction ('all-MiniLM-L6-v2') failed to initialize.")
        
    try:
        embeddings = fn([text])
        if embeddings and len(embeddings) > 0:
            return [float(v) for v in embeddings[0]]
        raise RuntimeError("SentenceTransformer returned empty embedding.")
    except Exception as e:
        logger.error(f"Failed to generate embedding from SentenceTransformer: {e}")
        raise RuntimeError(f"Embedding generation failed: {e}")
