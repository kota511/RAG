import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.answers import (
    ANSWER_INSTRUCTIONS,
    ANSWER_MODEL,
    AnswerSource,
    GeneratedAnswer,
    generate_answer,
)


def test_generates_structured_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    generated_answer = GeneratedAnswer(
        answer="Grounded answers use citations.",
        has_sufficient_context=True,
        cited_chunk_ids=["document-123:0"],
    )
    client = Mock()
    client.responses.parse.return_value = SimpleNamespace(
        output_parsed=generated_answer
    )
    monkeypatch.setattr("app.answers.OpenAI", lambda: client)
    sources = [
        AnswerSource(
            chunk_id="document-123:0",
            text="Grounded answers use citations.",
        )
    ]

    result = generate_answer("How are answers grounded?", sources)

    assert result == generated_answer
    client.responses.parse.assert_called_once_with(
        model=ANSWER_MODEL,
        reasoning={"effort": "low"},
        instructions=ANSWER_INSTRUCTIONS,
        input=json.dumps(
            {
                "question": "How are answers grounded?",
                "sources": [
                    {
                        "chunk_id": "document-123:0",
                        "text": "Grounded answers use citations.",
                    }
                ],
            }
        ),
        text_format=GeneratedAnswer,
        verbosity="low",
        store=False,
    )


def test_returns_none_without_parsed_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Mock()
    client.responses.parse.return_value = SimpleNamespace(output_parsed=None)
    monkeypatch.setattr("app.answers.OpenAI", lambda: client)

    result = generate_answer(
        "How are answers grounded?",
        [
            AnswerSource(
                chunk_id="document-123:0",
                text="Grounded answers use citations.",
            )
        ],
    )

    assert result is None
