# pyrefly: ignore [missing-import]
from fastapi import FastAPI
from api.webhook import router

app = FastAPI()

app.include_router(router)

from core.rag import generate_answer

@app.get("/query")
async def query_rag(q: str):
    return generate_answer(q)

#http://127.0.0.1:8000/webhook/knowledge-file

#{"Content-Type": "application/json"}

"""
{
  "type":"INSERT",
  "table":"knowledge_files",
  "schema":"public",
  "record":{
      "id":1,
      "bot_id":"bot_123",
      "user_id":"user_123",
      "file_name":"python.pdf",
      "storage_path":"knowledge/python.pdf",
      "status":"pending"
  },
  "old_record":null
}

"""
