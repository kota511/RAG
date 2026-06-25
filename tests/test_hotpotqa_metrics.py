import pytest

from benchmarks.hotpotqa_chunks import (
    PreparedDocument,
    PreparedSupportingFact,
)
from benchmarks.hotpotqa_metrics import (
    AggregateRetrievalMetrics,
    RetrievedChunkReference,
    RetrievalMetrics,
    aggregate_retrieval_metrics,
    calculate_retrieval_metrics,
)


def prepared_documents() -> tuple[PreparedDocument, ...]:
    return (
        PreparedDocument(
            title="First",
            text="",
            chunks=(),
            supporting_facts=(
                PreparedSupportingFact(
                    sentence_index=0,
                    text="First fact.",
                    start=0,
                    end=11,
                    chunk_indexes=(2, 3),
                ),
            ),
        ),
        PreparedDocument(
            title="Second",
            text="",
            chunks=(),
            supporting_facts=(
                PreparedSupportingFact(
                    sentence_index=1,
                    text="Second fact.",
                    start=12,
                    end=24,
                    chunk_indexes=(7,),
                ),
            ),
        ),
    )


def reference(title: str, chunk_index: int) -> RetrievedChunkReference:
    return RetrievedChunkReference(
        document_title=title,
        chunk_index=chunk_index,
    )


def test_scores_complete_retrieval() -> None:
    metrics = calculate_retrieval_metrics(
        prepared_documents(),
        [
            reference("Distractor", 0),
            reference("First", 2),
            reference("Second", 7),
        ],
    )

    assert metrics == RetrievalMetrics(
        limit=5,
        supporting_fact_count=2,
        matched_supporting_fact_count=2,
        supporting_fact_recall=1.0,
        complete_support=True,
        reciprocal_rank=0.5,
    )


def test_scores_partial_retrieval_and_honours_limit() -> None:
    metrics = calculate_retrieval_metrics(
        prepared_documents(),
        [
            reference("First", 3),
            reference("Distractor", 0),
            reference("Second", 7),
        ],
        limit=2,
    )

    assert metrics.supporting_fact_recall == 0.5
    assert metrics.complete_support is False
    assert metrics.reciprocal_rank == 1.0


def test_scores_no_relevant_retrieval() -> None:
    metrics = calculate_retrieval_metrics(
        prepared_documents(),
        [reference("Distractor", 0)],
    )

    assert metrics.matched_supporting_fact_count == 0
    assert metrics.supporting_fact_recall == 0.0
    assert metrics.complete_support is False
    assert metrics.reciprocal_rank == 0.0


def test_aggregates_case_metrics() -> None:
    aggregate = aggregate_retrieval_metrics(
        [
            RetrievalMetrics(
                limit=5,
                supporting_fact_count=2,
                matched_supporting_fact_count=2,
                supporting_fact_recall=1.0,
                complete_support=True,
                reciprocal_rank=1.0,
            ),
            RetrievalMetrics(
                limit=5,
                supporting_fact_count=2,
                matched_supporting_fact_count=1,
                supporting_fact_recall=0.5,
                complete_support=False,
                reciprocal_rank=0.5,
            ),
        ]
    )

    assert aggregate == AggregateRetrievalMetrics(
        limit=5,
        case_count=2,
        mean_supporting_fact_recall=0.75,
        complete_support_rate=0.5,
        mean_reciprocal_rank=0.75,
    )


def test_rejects_missing_supporting_facts() -> None:
    with pytest.raises(ValueError):
        calculate_retrieval_metrics(
            [
                PreparedDocument(
                    title="Document",
                    text="",
                    chunks=(),
                    supporting_facts=(),
                )
            ],
            [],
        )


def test_rejects_mixed_aggregate_limits() -> None:
    with pytest.raises(ValueError):
        aggregate_retrieval_metrics(
            [
                RetrievalMetrics(1, 1, 1, 1.0, True, 1.0),
                RetrievalMetrics(5, 1, 1, 1.0, True, 1.0),
            ]
        )
