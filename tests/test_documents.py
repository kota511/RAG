from collections.abc import Iterator
from unittest.mock import Mock
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient, models

from app.answers import AnswerSource, GeneratedAnswer
from app.documents import (
    chunk_text,
    generate_document_id,
)
from app.main import (
    INSUFFICIENT_CONTEXT_ANSWER,
    SearchResult,
    app,
    stored_documents,
)
from app.vector_store import (
    COLLECTION_NAME,
    EMBEDDING_SIZE,
    ensure_collection,
    generate_point_id,
    reset_collection,
)

client = TestClient(app)


def test_health_check_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def fake_embeddings(texts: list[str]) -> list[list[float]]:
    return [
        [1.0] + [0.0] * (EMBEDDING_SIZE - 1)
        for _ in texts
    ]


@pytest.fixture(autouse=True)
def reset_application_state(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[QdrantClient]:
    stored_documents.clear()
    monkeypatch.setattr("app.main.create_embeddings", fake_embeddings)
    test_qdrant_client = QdrantClient(":memory:")
    monkeypatch.setattr("app.main.qdrant_client", test_qdrant_client)

    with client:
        yield test_qdrant_client

    stored_documents.clear()


def semantic_test_embeddings(texts: list[str]) -> list[list[float]]:
    vectors = {
        "Grounded answers include citations.": [1.0, 0.0]
        + [0.0] * (EMBEDDING_SIZE - 2),
        "Deployment uses containers.": [0.0, 1.0]
        + [0.0] * (EMBEDDING_SIZE - 2),
        "how are sources referenced": [1.0, 0.0]
        + [0.0] * (EMBEDDING_SIZE - 2),
    }
    return [vectors[text] for text in texts]


def test_ensure_collection_keeps_existing_points(
    reset_application_state: QdrantClient,
) -> None:
    reset_application_state.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=1,
                vector=[1.0] * EMBEDDING_SIZE,
            )
        ],
    )

    ensure_collection(reset_application_state)

    collection = reset_application_state.get_collection(COLLECTION_NAME)
    vectors = collection.config.params.vectors
    point_count = reset_application_state.count(
        collection_name=COLLECTION_NAME,
        exact=True,
    ).count

    assert isinstance(vectors, models.VectorParams)
    assert vectors.size == EMBEDDING_SIZE
    assert vectors.distance == models.Distance.COSINE
    assert point_count == 1


def test_reset_collection_removes_stale_points(
    reset_application_state: QdrantClient,
) -> None:
    reset_application_state.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=1,
                vector=[1.0] * EMBEDDING_SIZE,
            )
        ],
    )

    reset_collection(reset_application_state)

    assert reset_application_state.count(
        collection_name=COLLECTION_NAME,
        exact=True,
    ).count == 0


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


def test_generates_deterministic_point_id() -> None:
    first = generate_point_id("document-123:0")
    second = generate_point_id("document-123:0")

    assert first == second
    UUID(first)


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


def test_upload_stores_chunk_point(
    reset_application_state: QdrantClient,
) -> None:
    uploaded = client.post(
        "/documents",
        files={"file": ("notes.txt", "rag test", "text/plain")},
    ).json()
    chunk = uploaded["chunks"][0]

    points = reset_application_state.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[generate_point_id(chunk["chunk_id"])],
        with_payload=True,
        with_vectors=True,
    )

    assert len(points) == 1
    assert points[0].payload == {
        "document_id": uploaded["document_id"],
        "chunk_id": chunk["chunk_id"],
        "chunk_index": 0,
        "text": "rag test",
    }
    assert points[0].vector == fake_embeddings(["rag test"])[0]


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


def test_upload_rejects_invalid_utf8() -> None:
    response = client.post(
        "/documents",
        files={"file": ("bad.txt", b"\xff", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "File must be valid UTF-8"}


def test_upload_rejects_embedding_count_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.main.create_embeddings", lambda _texts: [])

    response = client.post(
        "/documents",
        files={"file": ("notes.txt", "rag test", "text/plain")},
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "Embedding count mismatch"}


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


def test_deleted_document_is_not_lexically_searchable() -> None:
    uploaded = client.post(
        "/documents",
        files={"file": ("notes.txt", "unique searchable phrase", "text/plain")},
    ).json()

    client.delete(f"/documents/{uploaded['document_id']}")
    response = client.get("/search", params={"query": "unique"})

    assert response.status_code == 200
    assert response.json() == []


def test_deleted_document_is_not_semantically_searchable() -> None:
    uploaded = client.post(
        "/documents",
        files={"file": ("notes.txt", "unique semantic phrase", "text/plain")},
    ).json()

    before_delete = client.get(
        "/semantic-search",
        params={"query": "related meaning"},
    )
    client.delete(f"/documents/{uploaded['document_id']}")
    after_delete = client.get(
        "/semantic-search",
        params={"query": "related meaning"},
    )

    assert before_delete.json()[0]["document_id"] == uploaded["document_id"]
    assert after_delete.status_code == 200
    assert after_delete.json() == []


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
    assert results == [
        {
            "document_id": citations_document["document_id"],
            "chunk_id": citations_document["chunks"][0]["chunk_id"],
            "chunk_index": 0,
            "text": "Grounded answers include citations.",
            "score": pytest.approx(1.0),
        }
    ]


def test_answers_returns_model_selected_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semantic_results = [
        SearchResult(
            document_id="document-1",
            chunk_id="document-1:0",
            chunk_index=0,
            text="Grounded answers use retrieved context.",
            score=0.95,
        ),
        SearchResult(
            document_id="document-2",
            chunk_id="document-2:0",
            chunk_index=0,
            text="Citations identify supporting chunks.",
            score=0.90,
        ),
    ]
    retrieve_results = Mock(return_value=semantic_results)
    generate = Mock(
        return_value=GeneratedAnswer(
            answer="Citations identify the supporting chunks.",
            has_sufficient_context=True,
            cited_chunk_ids=["document-2:0", "document-2:0"],
        )
    )
    monkeypatch.setattr("app.main.retrieve_semantic_results", retrieve_results)
    monkeypatch.setattr("app.main.generate_answer", generate)

    response = client.post(
        "/answers",
        json={"question": "How do citations work?", "limit": 2},
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Citations identify the supporting chunks.",
        "insufficient_context": False,
        "citations": [
            {
                "document_id": "document-2",
                "chunk_id": "document-2:0",
                "chunk_index": 0,
                "text": "Citations identify supporting chunks.",
                "score": 0.90,
            }
        ],
    }
    retrieve_results.assert_called_once_with("How do citations work?", 2)
    generate.assert_called_once_with(
        "How do citations work?",
        [
            AnswerSource(
                chunk_id="document-1:0",
                text="Grounded answers use retrieved context.",
            ),
            AnswerSource(
                chunk_id="document-2:0",
                text="Citations identify supporting chunks.",
            ),
        ],
    )


def test_answers_skips_generation_without_retrieved_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.main.retrieve_semantic_results", lambda _query, _limit: [])
    generate = Mock()
    monkeypatch.setattr("app.main.generate_answer", generate)

    response = client.post(
        "/answers",
        json={"question": "What is not documented?"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": INSUFFICIENT_CONTEXT_ANSWER,
        "insufficient_context": True,
        "citations": [],
    }
    generate.assert_not_called()


def test_answers_returns_insufficient_context_when_model_cannot_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.main.retrieve_semantic_results",
        lambda _query, _limit: [
            SearchResult(
                document_id="document-1",
                chunk_id="document-1:0",
                chunk_index=0,
                text="Unrelated context.",
                score=0.20,
            )
        ],
    )
    monkeypatch.setattr(
        "app.main.generate_answer",
        lambda _question, _sources: GeneratedAnswer(
            answer="This answer must be discarded.",
            has_sufficient_context=False,
            cited_chunk_ids=["fabricated:0"],
        ),
    )

    response = client.post(
        "/answers",
        json={"question": "What is not documented?"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": INSUFFICIENT_CONTEXT_ANSWER,
        "insufficient_context": True,
        "citations": [],
    }


def test_answers_rejects_missing_structured_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.main.retrieve_semantic_results",
        lambda _query, _limit: [
            SearchResult(
                document_id="document-1",
                chunk_id="document-1:0",
                chunk_index=0,
                text="Grounded answers use context.",
                score=0.95,
            )
        ],
    )
    monkeypatch.setattr(
        "app.main.generate_answer",
        lambda _question, _sources: None,
    )

    response = client.post(
        "/answers",
        json={"question": "How are answers grounded?"},
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "Answer generation failed"}


@pytest.mark.parametrize(
    "cited_chunk_ids",
    [
        [],
        ["fabricated:0"],
    ],
)
def test_answers_rejects_invalid_citations(
    monkeypatch: pytest.MonkeyPatch,
    cited_chunk_ids: list[str],
) -> None:
    monkeypatch.setattr(
        "app.main.retrieve_semantic_results",
        lambda _query, _limit: [
            SearchResult(
                document_id="document-1",
                chunk_id="document-1:0",
                chunk_index=0,
                text="Grounded answers use context.",
                score=0.95,
            )
        ],
    )
    monkeypatch.setattr(
        "app.main.generate_answer",
        lambda _question, _sources: GeneratedAnswer(
            answer="Grounded answers use context.",
            has_sufficient_context=True,
            cited_chunk_ids=cited_chunk_ids,
        ),
    )

    response = client.post(
        "/answers",
        json={"question": "How are answers grounded?"},
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "Answer contained invalid citations"}


@pytest.mark.parametrize(
    "request_body",
    [
        {"question": ""},
        {"question": "Valid question", "limit": 0},
        {"question": "Valid question", "limit": 21},
    ],
)
def test_answers_rejects_invalid_request(request_body: dict[str, object]) -> None:
    response = client.post("/answers", json=request_body)

    assert response.status_code == 422
