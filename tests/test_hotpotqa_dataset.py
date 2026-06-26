import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from benchmarks.hotpotqa_dataset import (
    DATASET_URL,
    BenchmarkDocument,
    HotpotQACase,
    create_hotpotqa_case,
    load_hotpotqa_cases,
    select_hotpotqa_cases,
)


def create_dataset_row(dataset_id: str) -> dict[str, object]:
    return {
        "id": dataset_id,
        "question": "Which fact connects the two documents?",
        "answer": "The shared fact",
        "type": "bridge",
        "level": "medium",
        "context": {
            "title": ["First document", "Second document"],
            "sentences": [
                ["Unrelated sentence.", "The first supporting sentence."],
                ["The second supporting sentence."],
            ],
        },
        "supporting_facts": {
            "title": [
                "First document",
                "First document",
                "Second document",
            ],
            "sent_id": [0, 1, 0],
        },
    }


def test_creates_case_with_unique_supporting_document_titles() -> None:
    case = create_hotpotqa_case(create_dataset_row("case-1"))

    assert case == HotpotQACase(
        dataset_id="case-1",
        question="Which fact connects the two documents?",
        answer="The shared fact",
        documents=(
            BenchmarkDocument(
                title="First document",
                sentences=(
                    "Unrelated sentence.",
                    "The first supporting sentence.",
                ),
            ),
            BenchmarkDocument(
                title="Second document",
                sentences=("The second supporting sentence.",),
            ),
        ),
        supporting_document_titles=(
            "First document",
            "Second document",
        ),
    )


def test_selects_same_cases_regardless_of_input_order() -> None:
    rows = [
        create_dataset_row("case-1"),
        create_dataset_row("case-2"),
        create_dataset_row("case-3"),
        create_dataset_row("case-4"),
    ]

    first_selection = select_hotpotqa_cases(rows, count=2, seed=123)
    second_selection = select_hotpotqa_cases(
        reversed(rows),
        count=2,
        seed=123,
    )

    expected_ids = ["case-1", "case-4"]

    assert [case.dataset_id for case in first_selection] == expected_ids
    assert [case.dataset_id for case in second_selection] == expected_ids


def test_loads_pinned_validation_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_dataset = Mock(return_value=[create_dataset_row("case-1")])
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=load_dataset),
    )

    cases = load_hotpotqa_cases(count=1, seed=5114)

    assert [case.dataset_id for case in cases] == ["case-1"]
    load_dataset.assert_called_once_with(
        "parquet",
        data_files={"validation": DATASET_URL},
        split="validation",
    )


def test_requires_explicit_case_count() -> None:
    with pytest.raises(TypeError, match="count"):
        load_hotpotqa_cases()


@pytest.mark.parametrize("count", [0, 5])
def test_rejects_invalid_case_count(count: int) -> None:
    rows = [create_dataset_row("case-1")]

    with pytest.raises(ValueError):
        select_hotpotqa_cases(rows, count=count, seed=123)
