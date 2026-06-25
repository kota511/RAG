from dataclasses import dataclass
from typing import Iterable, Sequence

from benchmarks.hotpotqa_chunks import PreparedDocument


@dataclass(frozen=True)
class RetrievedChunkReference:
    document_title: str
    chunk_index: int


@dataclass(frozen=True)
class RetrievalMetrics:
    limit: int
    supporting_fact_count: int
    matched_supporting_fact_count: int
    supporting_fact_recall: float
    complete_support: bool
    reciprocal_rank: float


@dataclass(frozen=True)
class AggregateRetrievalMetrics:
    limit: int
    case_count: int
    mean_supporting_fact_recall: float
    complete_support_rate: float
    mean_reciprocal_rank: float


def calculate_retrieval_metrics(
    documents: Iterable[PreparedDocument],
    retrieved_chunks: Sequence[RetrievedChunkReference],
    limit: int = 5,
) -> RetrievalMetrics:
    if limit <= 0:
        raise ValueError("limit must be greater than zero")

    supporting_chunk_sets = [
        {
            RetrievedChunkReference(
                document_title=document.title,
                chunk_index=chunk_index,
            )
            for chunk_index in fact.chunk_indexes
        }
        for document in documents
        for fact in document.supporting_facts
    ]
    if not supporting_chunk_sets:
        raise ValueError("documents must contain at least one supporting fact")

    top_results = retrieved_chunks[:limit]
    retrieved_set = set(top_results)
    matched_fact_count = sum(
        bool(supporting_chunks & retrieved_set)
        for supporting_chunks in supporting_chunk_sets
    )
    supporting_chunks = set().union(*supporting_chunk_sets)
    first_relevant_rank = next(
        (
            rank
            for rank, retrieved_chunk in enumerate(top_results, start=1)
            if retrieved_chunk in supporting_chunks
        ),
        None,
    )

    supporting_fact_count = len(supporting_chunk_sets)
    return RetrievalMetrics(
        limit=limit,
        supporting_fact_count=supporting_fact_count,
        matched_supporting_fact_count=matched_fact_count,
        supporting_fact_recall=matched_fact_count / supporting_fact_count,
        complete_support=matched_fact_count == supporting_fact_count,
        reciprocal_rank=(
            0.0
            if first_relevant_rank is None
            else 1 / first_relevant_rank
        ),
    )


def aggregate_retrieval_metrics(
    metrics: Sequence[RetrievalMetrics],
) -> AggregateRetrievalMetrics:
    if not metrics:
        raise ValueError("metrics must not be empty")

    limits = {result.limit for result in metrics}
    if len(limits) != 1:
        raise ValueError("all metrics must use the same limit")

    case_count = len(metrics)
    return AggregateRetrievalMetrics(
        limit=metrics[0].limit,
        case_count=case_count,
        mean_supporting_fact_recall=(
            sum(result.supporting_fact_recall for result in metrics)
            / case_count
        ),
        complete_support_rate=(
            sum(result.complete_support for result in metrics)
            / case_count
        ),
        mean_reciprocal_rank=(
            sum(result.reciprocal_rank for result in metrics)
            / case_count
        ),
    )
