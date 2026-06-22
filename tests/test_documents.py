from fastapi.testclient import TestClient
from app.main import app, extract_text_from_bytes

client = TestClient(app)


def test_extract_text_from_bytes_decodes_utf8_text() -> None:
    text = extract_text_from_bytes("rag test")

    assert text == "rag test"


def test_upload_document_metadata() -> None:
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
    }
