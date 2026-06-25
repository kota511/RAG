from unittest.mock import Mock, call

import pytest

from app.vector_store import EMBEDDING_SIZE
from benchmarks.hotpotqa_dataset import (
    BenchmarkDocument,
    HotpotQACase,
    SupportingFact,
)
from benchmarks.hotpotqa_metrics import RetrievalMetrics
from benchmarks.hotpotqa_retrieval import evaluate_retrieval_case


def vector(first: float, second: float) -> list[float]:
    return [first, second] + [0.0] * (EMBEDDING_SIZE - 2)


def benchmark_case() -> HotpotQACase:
    return HotpotQACase(
        dataset_id="case-1",
        question="Where is the evidence?",
        answer="Relevant",
        question_type="bridge",
        level="easy",
        documents=(
            BenchmarkDocument(
                title="Relevant",
                sentences=("The evidence is here.",),
            ),
            BenchmarkDocument(
                title="Distractor",
                sentences=("Unrelated information.",),
            ),
        ),
        supporting_facts=(
            SupportingFact(
                title="Relevant",
                sentence_index=0,
                text="The evidence is here.",
            ),
        ),
    )


def test_evaluates_live_retrieval_with_batched_chunk_embeddings() -> None:
    embeddings_by_text = {
        "Relevant\nThe evidence is here.": vector(1.0, 0.0),
        "Distractor\nUnrelated information.": vector(0.0, 1.0),
        "Where is the evidence?": vector(1.0, 0.0),
    }
    create_embeddings = Mock(
        side_effect=lambda texts: [
            embeddings_by_text[text]
            for text in texts
        ]
    )

    metrics = evaluate_retrieval_case(
        benchmark_case(),
        limit=1,
        embedding_function=create_embeddings,
    )

    assert metrics == RetrievalMetrics(
        limit=1,
        supporting_fact_count=1,
        matched_supporting_fact_count=1,
        supporting_fact_recall=1.0,
        complete_support=True,
        reciprocal_rank=1.0,
    )
    assert create_embeddings.call_args_list == [
        call(
            [
                "Relevant\nThe evidence is here.",
                "Distractor\nUnrelated information.",
                "Where is the evidence?",
            ]
        )
    ]


def test_rejects_missing_chunk_embeddings() -> None:
    with pytest.raises(
        ValueError,
        match="embedding count does not match input count",
    ):
        evaluate_retrieval_case(
            benchmark_case(),
            embedding_function=lambda _texts: [],
        )
