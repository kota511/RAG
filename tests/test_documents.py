from fastapi.testclient import TestClient
from app.main import app, chunk_text, extract_text_from_bytes

client = TestClient(app)


def test_extracts_text() -> None:
    text = extract_text_from_bytes(b"rag test")

    assert text == "rag test"


def test_upload_returns_metadata() -> None:
    response = client.post(
        "/documents",
        files={"file": ("notes.txt", "rag test", "text/plain")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "filename": "notes.txt",
        "content_type": "text/plain",
        "size_bytes": 8,
        "text": "rag test",
        "character_count": 8,
        "chunk_count": 1,
        "chunks": ["rag test"],
    }


def test_upload_rejects_unsupported_type() -> None:
    response = client.post(
        "/documents",
        files={"file": ("notes.pdf", "%PDF-1.7", "application/pdf")},
    )

    assert response.status_code == 415
    assert response.json() == {"detail": "Unsupported file type"}


def test_upload_rejects_empty_file() -> None:
    response = client.post(
        "/documents",
        files={"file": ("empty.txt", "", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "File is empty"}


def test_chunks_text() -> None:
    text = "testing if this will chunk properly"
    chunks = chunk_text(text, chunk_size=10)

    assert chunks == ["testing if", " this will", " chunk pro", "perly"]
