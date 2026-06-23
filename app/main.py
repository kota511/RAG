from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.documents import (
    chunk_text,
    extract_text_from_bytes,
    generate_chunk_id,
    generate_document_id,
)

app = FastAPI(title="RAG API")


class DocumentChunk(BaseModel):
    chunk_index: int
    text: str
    chunk_id: str


class DocumentUploadResponse(BaseModel):
    filename: str | None
    content_type: str | None
    size_bytes: int
    text: str
    character_count: int
    chunk_count: int
    chunks: list[DocumentChunk]
    document_id: str


stored_documents: dict[str, DocumentUploadResponse] = {}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    if file.content_type != "text/plain":
        raise HTTPException(status_code=415, detail="Unsupported file type")

    contents = await file.read()
    text = extract_text_from_bytes(contents)
    if len(text) == 0:
        raise HTTPException(status_code=400, detail="File is empty")

    document_id = generate_document_id()
    chunks = [
        DocumentChunk(
            chunk_id=generate_chunk_id(document_id, index),
            chunk_index=index,
            text=chunk,
        )
        for index, chunk in enumerate(chunk_text(text))
    ]

    response = DocumentUploadResponse(
        document_id=document_id,
        filename=file.filename,
        content_type=file.content_type,
        size_bytes=len(contents),
        text=text,
        character_count=len(text),
        chunk_count=len(chunks),
        chunks=chunks,
    )

    stored_documents[document_id] = response
    return response


@app.get("/documents/{document_id}")
def get_document(document_id: str) -> DocumentUploadResponse:
    document = stored_documents.get(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    return document
