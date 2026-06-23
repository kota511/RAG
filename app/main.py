from fastapi import FastAPI, File, HTTPException, UploadFile

app = FastAPI(title="RAG API")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


def extract_text_from_bytes(contents: bytes) -> str:
    return contents.decode("utf-8")


def chunk_text(text: str, chunk_size: int = 100) -> list[str]:
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)) -> dict[str, str | int | list[str] | None]:
    if file.content_type != "text/plain":
        raise HTTPException(status_code=415, detail="Unsupported file type")

    contents = await file.read()
    text = extract_text_from_bytes(contents)
    if len(text) == 0:
        raise HTTPException(status_code=400, detail="File is empty")

    chunks = chunk_text(text)

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": len(contents),
        "text": text,
        "character_count": len(text),
        "chunk_count": len(chunks),
        "chunks": chunks,
    }
