from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


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
    }