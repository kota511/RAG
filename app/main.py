from fastapi import FastAPI, File, UploadFile

app = FastAPI(title="RAG API")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/documents")
async def upload_document(file: UploadFile = File(...)) -> dict[str, str | int |None]:
    contents = await file.read()

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": len(contents)
    }
