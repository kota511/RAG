from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from pydantic import BaseModel

from app.documents import (
    calculate_lexical_score,
    chunk_text,
    extract_text_from_bytes,
    generate_chunk_id,
    generate_document_id,
)
from app.embeddings import cosine_similarity, create_embeddings
from app.vector_store import create_qdrant_client, ensure_collection

qdrant_client = create_qdrant_client()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    try:
        ensure_collection(qdrant_client)
        yield
    finally:
        qdrant_client.close()


app = FastAPI(title="RAG API", lifespan=lifespan)


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


class SearchResult(BaseModel):
    document_id: str
    chunk_id: str
    chunk_index: int
    text: str
    score: float


class DocumentSummary(BaseModel):
    document_id: str
    filename: str | None
    content_type: str | None
    size_bytes: int
    character_count: int
    chunk_count: int


stored_documents: dict[str, DocumentUploadResponse] = {}
stored_embeddings: dict[str, list[float]] = {}


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
        for index, chunk in enumerate(
            chunk_text(text, chunk_size=100, chunk_overlap=20)
        )
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

    embeddings = create_embeddings([chunk.text for chunk in chunks])

    stored_documents[document_id] = response
    stored_embeddings.update(
        {
            chunk.chunk_id: embedding
            for chunk, embedding in zip(chunks, embeddings)
        }
    )
    return response


@app.get("/documents")
def list_documents() -> list[DocumentSummary]:
    return [
        DocumentSummary(
            document_id=document.document_id,
            filename=document.filename,
            content_type=document.content_type,
            size_bytes=document.size_bytes,
            character_count=document.character_count,
            chunk_count=document.chunk_count,
        )
        for document in stored_documents.values()
    ]


@app.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: str) -> Response:
    document = stored_documents.get(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    for chunk in document.chunks:
        stored_embeddings.pop(chunk.chunk_id, None)

    del stored_documents[document_id]
    return Response(status_code=204)


@app.get("/documents/{document_id}")
def get_document(document_id: str) -> DocumentUploadResponse:
    document = stored_documents.get(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    return document


@app.get("/search")
def search_documents(
    query: str = Query(min_length=1),
    limit: int = Query(default=5, ge=1, le=20),
) -> list[SearchResult]:
    results: list[SearchResult] = []

    for document in stored_documents.values():
        for chunk in document.chunks:
            score = calculate_lexical_score(query, chunk.text)
            if score == 0:
                continue

            results.append(
                SearchResult(
                    document_id=document.document_id,
                    chunk_id=chunk.chunk_id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    score=float(score),
                )
            )

    results.sort(key=lambda result: result.score, reverse=True)
    return results[:limit]


@app.get("/semantic-search")
def semantic_search(
    query: str = Query(min_length=1),
    limit: int = Query(default=5, ge=1, le=20),
) -> list[SearchResult]:
    query_embedding = create_embeddings([query])[0]
    results: list[SearchResult] = []

    for document in stored_documents.values():
        for chunk in document.chunks:
            score = cosine_similarity(
                query_embedding,
                stored_embeddings[chunk.chunk_id],
            )
            results.append(
                SearchResult(
                    document_id=document.document_id,
                    chunk_id=chunk.chunk_id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    score=score,
                )
            )

    results.sort(key=lambda result: result.score, reverse=True)
    return results[:limit]
