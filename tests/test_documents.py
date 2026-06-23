from uuid import UUID

from fastapi.testclient import TestClient
from app.main import app
from app.documents import chunk_text, extract_text_from_bytes, generate_document_id

client = TestClient(app)


def test_extracts_text() -> None:
    text = extract_text_from_bytes(b"rag test")

    assert text == "rag test"


def test_upload_returns_metadata() -> None:
    response = client.post(
        "/documents",
        files={"file": ("notes.txt", "rag test", "text/plain")},
    )
    response_body = response.json()

    assert response.status_code == 200
    assert isinstance(response_body["document_id"], str)
    UUID(response_body["document_id"])
    assert response_body == {
        "filename": "notes.txt",
        "content_type": "text/plain",
        "size_bytes": 8,
        "text": "rag test",
        "character_count": 8,
        "chunk_count": 1,
        "chunks": ["rag test"],
        "document_id": response_body["document_id"],
    }


def test_fetches_uploaded_document() -> None:
    upload_response = client.post(
        "/documents",
        files={"file": ("notes.txt", "rag test", "text/plain")},
    )
    document_id = upload_response.json()["document_id"]

    fetch_response = client.get(f"/documents/{document_id}")

    assert fetch_response.status_code == 200
    assert fetch_response.json() == upload_response.json()


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


def test_generates_document_id() -> None:
    document_id = generate_document_id()

    assert isinstance(document_id, str)
    UUID(document_id)


def test_fetch_rejects_missing_document() -> None:
    response = client.get("/documents/nonexistent-id")

    assert response.status_code == 404
    assert response.json() == {"detail": "Document not found"}
