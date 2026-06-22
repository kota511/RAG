from fastapi import FastAPI, File, UploadFile

app = FastAPI(title="RAG API")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


def extract_text_from_bytes(contents: bytes) -> str:
    return contents.decode("utf-8")


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)) -> dict[str, str | int | None]:
    contents = await file.read()
    text = extract_text_from_bytes(contents)

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": len(contents),
        "text": text,
        "character_count": len(text),
    }
