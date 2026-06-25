from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class RetrievalMetrics:
    limit: int
    supporting_document_count: int
    matched_supporting_document_count: int
    supporting_document_recall: float
    complete_support: bool
    reciprocal_rank: float


@dataclass(frozen=True)
class AggregateRetrievalMetrics:
    limit: int
    case_count: int
    mean_supporting_document_recall: float
    complete_support_rate: float
    mean_reciprocal_rank: float


def calculate_retrieval_metrics(
    supporting_document_titles: Iterable[str],
    retrieved_document_titles: Sequence[str],
    limit: int = 5,
) -> RetrievalMetrics:
    if limit <= 0:
        raise ValueError("limit must be greater than zero")

    supporting_documents = set(supporting_document_titles)
    if not supporting_documents:
        raise ValueError("supporting_document_titles must not be empty")

    top_results = retrieved_document_titles[:limit]
    matched_count = len(supporting_documents & set(top_results))
    first_relevant_rank = next(
        (
            rank
            for rank, title in enumerate(top_results, start=1)
            if title in supporting_documents
        ),
        None,
    )

    supporting_document_count = len(supporting_documents)
    return RetrievalMetrics(
        limit=limit,
        supporting_document_count=supporting_document_count,
        matched_supporting_document_count=matched_count,
        supporting_document_recall=matched_count / supporting_document_count,
        complete_support=matched_count == supporting_document_count,
        reciprocal_rank=(
            0.0 if first_relevant_rank is None else 1 / first_relevant_rank
        ),
    )


def aggregate_retrieval_metrics(
    metrics: Sequence[RetrievalMetrics],
) -> AggregateRetrievalMetrics:
    if not metrics:
        raise ValueError("metrics must not be empty")
    if len({result.limit for result in metrics}) != 1:
        raise ValueError("all metrics must use the same limit")

    case_count = len(metrics)
    return AggregateRetrievalMetrics(
        limit=metrics[0].limit,
        case_count=case_count,
        mean_supporting_document_recall=(
            sum(result.supporting_document_recall for result in metrics)
            / case_count
        ),
        complete_support_rate=(
            sum(result.complete_support for result in metrics) / case_count
        ),
        mean_reciprocal_rank=(
            sum(result.reciprocal_rank for result in metrics) / case_count
        ),
    )
