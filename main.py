import os
import io
import csv
import json
import time
# pyrefly: ignore [missing-import]
import httpx
import tempfile
from datetime import datetime, timezone
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException, Request
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from pydantic import BaseModel
# pyrefly: ignore [missing-import]
from supabase import create_client, Client
# pyrefly: ignore [missing-import]
from openai import OpenAI
# pyrefly: ignore [missing-import]
import pypdf
# pyrefly: ignore [missing-import]
import docx
# pyrefly: ignore [missing-import]
from google import genai

load_dotenv()

# ─── Clients ──────────────────────────────────────────────────────────────────
app = FastAPI(title="Clarix API")
genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
openai_chat = OpenAI(api_key=os.getenv("OPENROUTER_API_KEY"), base_url=os.getenv("OPENROUTER_BASE_URL"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")  # service key — not anon key!
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Constants ────────────────────────────────────────────────────────────────
CHAT_MODEL    = os.getenv("OPENAI_MODEL")
CHUNK_SIZE    = 500   # characters per chunk
CHUNK_OVERLAP = 50    # overlap between chunks
TOP_K         = 5     # how many chunks to retrieve


# ══════════════════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    bot_id: str
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    answer: str
    was_answered: bool
    response_time_ms: int


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def extract_text(file_bytes: bytes, file_name: str) -> str:
    """Extract raw text from PDF, DOCX, TXT, JSON, or CSV."""
    ext = file_name.split(".")[-1].lower()

    if ext == "pdf":
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    elif ext == "docx":
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(para.text for para in doc.paragraphs)

    elif ext == "txt":
        return file_bytes.decode("utf-8", errors="ignore")

    elif ext == "csv":
        text = file_bytes.decode("utf-8", errors="ignore")
        reader = csv.reader(io.StringIO(text))
        return "\n".join(", ".join(row) for row in reader)

    elif ext == "json":
        text = file_bytes.decode("utf-8", errors="ignore")
        data = json.loads(text)
        lines: list[str] = []

        def flatten(obj, prefix: str = ""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    flatten(v, f"{prefix}{k}: " if not prefix else f"{prefix}{k}: ")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    flatten(item, f"{prefix}[{i}]: ")
            else:
                lines.append(f"{prefix}{obj}")

        flatten(data)
        return "\n".join(lines)

    else:
        raise ValueError(f"Unsupported file type: {ext}")


def get_embedding(text: str) -> list[float]:
    """Convert text to embedding vector using Gemini."""
    result = genai_client.models.embed_content(
        model="gemini-embedding-2",
        contents=text.replace("\n", " ")
    )
    return result.embeddings[0].values

def store_chunks(bot_id: str, file_id: str, chunks: list[str]):
    """Embed each chunk and store in Supabase pgvector."""
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        supabase.table("document_chunks").insert({
            "bot_id": bot_id,
            "file_id": file_id,
            "content": chunk,
            "embedding": embedding,
            "metadata": {"bot_id": bot_id, "file_id": file_id},
            "chunk_index": i,
        }).execute()


def search_chunks(bot_id: str, question: str) -> list[str]:
    """Find most relevant chunks for a question."""
    question_embedding = get_embedding(question)
    result = supabase.rpc("match_chunks", {
        "query_embedding": question_embedding,
        "filter": {"bot_id": bot_id},
    }).execute()
    return [row["content"] for row in result.data]


def update_file_status(file_id: str, status: str, error: str = ""):
    """Update knowledge_files status in Supabase."""
    payload = {"status": status}
    if error:
        payload["error_message"] = error
    supabase.table("knowledge_files").update(payload).eq("id", file_id).execute()


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 1 — HEALTH CHECK
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 2 — WEBHOOK (Supabase triggers this on file insert)
# ══════════════════════════════════════════════════════════════════════════════

#@app.post("/webhook/knowledge-file")
@app.post("/webhooks/supabase")
async def supabase_webhook(request: Request):
    """
    Supabase calls this when a new row is inserted in knowledge_files.
    We download the file, extract text, chunk it, embed it, store in pgvector.
    """
    payload = await request.json()

    if not payload:
        return {"status": "error", "message": "Empty payload"}

    type_   = payload.get("type")
    if type_ == "DELETE":
        record = payload.get("old_record") or {}
    else:
        record = payload.get("record") or {}

    file_id = record.get("id")
    bot_id  = record.get("bot_id")
    file_name = record.get("name")
    file_type = record.get("type")

    if not all([file_id, bot_id, file_name]):
        raise HTTPException(status_code=400, detail="Missing required fields")

    # Skip URL type — handled separately
    if file_type == "url":
        await process_url(file_id=file_id, bot_id=bot_id, url=file_name)
        return {"status": "processing url"}

    try:
        # 1. Download file from Supabase Storage
        path = f"{bot_id}/{file_name}"
        response = supabase.storage.from_("knowledge").download(path)
        file_bytes = response  # returns bytes

        # 2. Extract text
        text = extract_text(file_bytes, file_name)
        if not text.strip():
            update_file_status(file_id, "failed", "Could not extract text from file")
            return {"status": "failed", "reason": "empty text"}

        # 3. Chunk text
        chunks = chunk_text(text)

        # 4. Embed + store in pgvector
        store_chunks(bot_id=bot_id, file_id=file_id, chunks=chunks)

        # 5. Mark as processed ✅
        update_file_status(file_id, "processed", error="")

        return {"status": "processed", "chunks": len(chunks)}

    except Exception as e:
        # Mark as failed so client sees it in Flutter
        update_file_status(file_id, "failed", str(e))
        raise HTTPException(status_code=500, detail=str(e))


async def process_url(file_id: str, bot_id: str, url: str):
    """Scrape a URL, chunk it, embed it, store in pgvector."""
    try:
        # 1. Scrape the page
        async with httpx.AsyncClient(timeout=30) as client:
            res = await client.get(url, follow_redirects=True)
            res.raise_for_status()

        # 2. Simple HTML strip — extract visible text
        from html.parser import HTMLParser

        class TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.text_parts = []
                self._skip = False

            def handle_starttag(self, tag, attrs):
                if tag in ("script", "style", "nav", "footer"):
                    self._skip = True

            def handle_endtag(self, tag):
                if tag in ("script", "style", "nav", "footer"):
                    self._skip = False

            def handle_data(self, data):
                if not self._skip and data.strip():
                    self.text_parts.append(data.strip())

        parser = TextExtractor()
        parser.feed(res.text)
        text = "\n".join(parser.text_parts)

        # 3. Chunk + embed + store
        chunks = chunk_text(text)
        store_chunks(bot_id=bot_id, file_id=file_id, chunks=chunks)

        # 4. Mark processed ✅
        update_file_status(file_id, "processed")

    except Exception as e:
        update_file_status(file_id, "failed", str(e))


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 3 — CHAT (JS Widget calls this)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    JS Widget sends question → we search pgvector → GPT-4o answers.
    Every message is saved to Supabase for analytics.
    """
    start = time.time()

    try:
        # 1. Search relevant chunks from this bot's knowledge
        chunks = search_chunks(bot_id=req.bot_id, question=req.message)
        was_answered = len(chunks) > 0

        # 2. Build context from chunks
        context = "\n\n".join(chunks) if chunks else ""

        # 3. Build system prompt
        system_prompt = """You are a helpful AI assistant. 
        Answer the user's question using ONLY the context provided below.
        If the context does not contain enough information to answer, 
        say "I don't have information about that. Please contact support."
        Be concise, friendly, and accurate.

        Context:
        """ + context

        # 4. Call Nvidia Model
        response = openai_chat.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.message},
            ],
            temperature=0.3,  # low temp = more factual answers
            max_tokens=500,
        )

        answer = response.choices[0].message.content
        response_time_ms = int((time.time() - start) * 1000)

        # 5. Save message to Supabase for analytics
        supabase.table("messages").insert({
            "bot_id": req.bot_id,
            "session_id": req.session_id,
            "question": req.message,
            "answer": answer,
            "was_answered": was_answered,
            "response_time_ms": response_time_ms,
        }).execute()

        return ChatResponse(
            answer=answer,
            was_answered=was_answered,
            response_time_ms=response_time_ms,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))