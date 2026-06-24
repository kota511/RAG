from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.embeddings import (
    EMBEDDING_MODEL,
    cosine_similarity,
    create_embeddings,
)


def test_creates_embeddings_in_input_order(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Mock()
    client.embeddings.create.return_value.data = [
        SimpleNamespace(index=1, embedding=[0.0, 1.0]),
        SimpleNamespace(index=0, embedding=[1.0, 0.0]),
    ]
    monkeypatch.setattr("app.embeddings.OpenAI", lambda: client)

    embeddings = create_embeddings(["first", "second"])

    assert embeddings == [[1.0, 0.0], [0.0, 1.0]]
    client.embeddings.create.assert_called_once_with(
        model=EMBEDDING_MODEL,
        input=["first", "second"],
    )


def test_calculates_cosine_similarity() -> None:
    similarity = cosine_similarity([1.0, 0.0], [1.0, 1.0])

    assert similarity == pytest.approx(0.7071, abs=0.0001)


def test_zero_vector_has_no_similarity() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_rejects_different_vector_dimensions() -> None:
    with pytest.raises(ValueError):
        cosine_similarity([1.0], [1.0, 0.0])
