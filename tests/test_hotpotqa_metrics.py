import pytest

from benchmarks.hotpotqa_metrics import (
    AggregateRetrievalMetrics,
    RetrievalMetrics,
    aggregate_retrieval_metrics,
    calculate_retrieval_metrics,
)


def test_scores_unique_supporting_documents() -> None:
    metrics = calculate_retrieval_metrics(
        ["First", "Second"],
        ["Distractor", "First", "First", "Second"],
    )

    assert metrics == RetrievalMetrics(
        limit=5,
        supporting_document_count=2,
        matched_supporting_document_count=2,
        supporting_document_recall=1.0,
        complete_support=True,
        reciprocal_rank=0.5,
    )


def test_honours_retrieval_limit() -> None:
    metrics = calculate_retrieval_metrics(
        ["First", "Second"],
        ["First", "Distractor", "Second"],
        limit=2,
    )

    assert metrics.supporting_document_recall == 0.5
    assert metrics.complete_support is False
    assert metrics.reciprocal_rank == 1.0


def test_scores_no_supporting_documents_retrieved() -> None:
    metrics = calculate_retrieval_metrics(
        ["First"],
        ["Distractor"],
    )

    assert metrics.matched_supporting_document_count == 0
    assert metrics.supporting_document_recall == 0.0
    assert metrics.complete_support is False
    assert metrics.reciprocal_rank == 0.0


def test_aggregates_case_metrics() -> None:
    aggregate = aggregate_retrieval_metrics(
        [
            RetrievalMetrics(5, 2, 2, 1.0, True, 1.0),
            RetrievalMetrics(5, 2, 1, 0.5, False, 0.5),
        ]
    )

    assert aggregate == AggregateRetrievalMetrics(
        limit=5,
        case_count=2,
        mean_supporting_document_recall=0.75,
        complete_support_rate=0.5,
        mean_reciprocal_rank=0.75,
    )


def test_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        calculate_retrieval_metrics([], [], limit=5)

    with pytest.raises(ValueError):
        calculate_retrieval_metrics(["Document"], [], limit=0)

    with pytest.raises(ValueError):
        aggregate_retrieval_metrics(
            [
                RetrievalMetrics(1, 1, 1, 1.0, True, 1.0),
                RetrievalMetrics(5, 1, 1, 1.0, True, 1.0),
            ]
        )
