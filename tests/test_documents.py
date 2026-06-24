from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.documents import (
    chunk_text,
    extract_text_from_bytes,
    generate_chunk_id,
    generate_document_id,
)
from app.main import (
    app,
    stored_documents,
    stored_embeddings,
)

client = TestClient(app)


def test_health_check_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def fake_embeddings(texts: list[str]) -> list[list[float]]:
    return [[1.0, 0.0] for _ in texts]


@pytest.fixture(autouse=True)
def reset_application_state(
    monkeypatch: pytest.MonkeyPatch,
):
    stored_documents.clear()
    stored_embeddings.clear()
    monkeypatch.setattr("app.main.create_embeddings", fake_embeddings)

    yield

    stored_documents.clear()
    stored_embeddings.clear()


def semantic_test_embeddings(texts: list[str]) -> list[list[float]]:
    vectors = {
        "Grounded answers include citations.": [1.0, 0.0],
        "Deployment uses containers.": [0.0, 1.0],
        "how are sources referenced": [1.0, 0.0],
    }
    return [vectors[text] for text in texts]


def test_extracts_text() -> None:
    assert extract_text_from_bytes(b"rag test") == "rag test"


def test_chunks_text() -> None:
    chunks = chunk_text(
        "testing if this will chunk properly",
        chunk_size=10,
        chunk_overlap=0,
    )

    assert chunks == ["testing if", " this will", " chunk pro", "perly"]


def test_chunks_text_with_overlap() -> None:
    chunks = chunk_text("abcdefghij12345", chunk_size=10, chunk_overlap=3)

    assert chunks == ["abcdefghij", "hij12345"]


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [
        (0, 0),
        (10, -1),
        (10, 10),
    ],
)
def test_rejects_invalid_chunk_settings(
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    with pytest.raises(ValueError):
        chunk_text(
            "some text",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


def test_generates_document_id() -> None:
    document_id = generate_document_id()

    assert isinstance(document_id, str)
    UUID(document_id)


def test_generates_chunk_id() -> None:
    assert generate_chunk_id("document-123", 0) == "document-123:0"


def test_upload_returns_metadata() -> None:
    response = client.post(
        "/documents",
        files={"file": ("notes.txt", "rag test", "text/plain")},
    )
    response_body = response.json()

    assert response.status_code == 200
    assert response_body == {
        "filename": "notes.txt",
        "content_type": "text/plain",
        "size_bytes": 8,
        "text": "rag test",
        "character_count": 8,
        "chunk_count": 1,
        "chunks": [
            {
                "chunk_id": f"{response_body['document_id']}:0",
                "chunk_index": 0,
                "text": "rag test",
            }
        ],
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


def test_upload_uses_overlapping_chunks() -> None:
    text = "a" * 100 + "b" * 20

    response = client.post(
        "/documents",
        files={"file": ("overlap.txt", text, "text/plain")},
    )
    chunks = response.json()["chunks"]

    assert response.status_code == 200
    assert [chunk["text"] for chunk in chunks] == [
        "a" * 100,
        "a" * 20 + "b" * 20,
    ]


def test_fetch_rejects_missing_document() -> None:
    response = client.get("/documents/nonexistent-id")

    assert response.status_code == 404
    assert response.json() == {"detail": "Document not found"}


def test_lists_document_summaries() -> None:
    uploaded = client.post(
        "/documents",
        files={"file": ("notes.txt", "rag test", "text/plain")},
    ).json()

    response = client.get("/documents")

    assert response.status_code == 200
    assert response.json() == [
        {
            "document_id": uploaded["document_id"],
            "filename": "notes.txt",
            "content_type": "text/plain",
            "size_bytes": 8,
            "character_count": 8,
            "chunk_count": 1,
        }
    ]


def test_deletes_document() -> None:
    uploaded = client.post(
        "/documents",
        files={"file": ("notes.txt", "rag test", "text/plain")},
    ).json()

    response = client.delete(f"/documents/{uploaded['document_id']}")

    assert response.status_code == 204
    assert response.content == b""
    assert client.get(f"/documents/{uploaded['document_id']}").status_code == 404


def test_delete_rejects_missing_document() -> None:
    response = client.delete("/documents/nonexistent-id")

    assert response.status_code == 404
    assert response.json() == {"detail": "Document not found"}


def test_deleted_document_is_not_searchable() -> None:
    uploaded = client.post(
        "/documents",
        files={"file": ("notes.txt", "unique searchable phrase", "text/plain")},
    ).json()

    client.delete(f"/documents/{uploaded['document_id']}")
    response = client.get("/search", params={"query": "unique"})

    assert response.status_code == 200
    assert response.json() == []


def test_search_chunks_case_insensitively() -> None:
    upload_response = client.post(
        "/documents",
        files={"file": ("search.txt", "answers use citations", "text/plain")},
    )
    uploaded_document = upload_response.json()
    search_response = client.get("/search", params={"query": "ANSWERS"})

    assert search_response.status_code == 200
    assert search_response.json() == [
        {
            "document_id": uploaded_document["document_id"],
            "chunk_id": uploaded_document["chunks"][0]["chunk_id"],
            "chunk_index": 0,
            "text": "answers use citations",
            "score": 1.0,
        }
    ]


def test_search_no_matches() -> None:
    client.post(
        "/documents",
        files={"file": ("search.txt", "answers use citations", "text/plain")},
    )

    search_response = client.get("/search", params={"query": "nonexistent query"})

    assert search_response.status_code == 200
    assert search_response.json() == []


def test_empty_query_rejected() -> None:
    response = client.get("/search", params={"query": ""})

    assert response.status_code == 422


def test_search_ranks_and_limits_results() -> None:
    first = client.post(
        "/documents",
        files={"file": ("first.txt", "citations and evidence", "text/plain")},
    ).json()
    client.post(
        "/documents",
        files={"file": ("second.txt", "citations only", "text/plain")},
    )

    response = client.get(
        "/search",
        params={"query": "citations evidence", "limit": 1},
    )
    results = response.json()

    assert response.status_code == 200
    assert len(results) == 1
    assert results[0]["document_id"] == first["document_id"]
    assert results[0]["score"] == 2.0


def test_semantic_search_ranks_related_chunk_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.main.create_embeddings", semantic_test_embeddings)
    citations_document = client.post(
        "/documents",
        files={
            "file": (
                "citations.txt",
                "Grounded answers include citations.",
                "text/plain",
            )
        },
    ).json()
    client.post(
        "/documents",
        files={"file": ("deployment.txt", "Deployment uses containers.", "text/plain")},
    )

    response = client.get(
        "/semantic-search",
        params={"query": "how are sources referenced", "limit": 1},
    )
    results = response.json()

    assert response.status_code == 200
    assert len(results) == 1
    assert results[0]["document_id"] == citations_document["document_id"]


def test_deleting_document_removes_embeddings() -> None:
    uploaded = client.post(
        "/documents",
        files={"file": ("notes.txt", "rag test", "text/plain")},
    ).json()
    chunk_id = uploaded["chunks"][0]["chunk_id"]

    client.delete(f"/documents/{uploaded['document_id']}")

    assert chunk_id not in stored_embeddings
