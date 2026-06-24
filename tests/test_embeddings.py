from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.embeddings import (
    EMBEDDING_MODEL,
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
