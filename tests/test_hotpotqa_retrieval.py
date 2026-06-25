import json
from pathlib import Path
from unittest.mock import Mock, call

import pytest

from app.vector_store import EMBEDDING_SIZE
from benchmarks.hotpotqa_dataset import BenchmarkDocument, HotpotQACase
from benchmarks.hotpotqa_metrics import RetrievalMetrics
from benchmarks.hotpotqa_retrieval import (
    RetrievalCaseResult,
    RetrievedChunk,
    create_report,
    evaluate_retrieval_case,
    write_report,
)


def vector(first: float, second: float) -> list[float]:
    return [first, second] + [0.0] * (EMBEDDING_SIZE - 2)


def benchmark_case() -> HotpotQACase:
    return HotpotQACase(
        dataset_id="case-1",
        question="Where is the evidence?",
        answer="Relevant",
        documents=(
            BenchmarkDocument(
                title="Relevant",
                sentences=("The evidence is here.",),
            ),
            BenchmarkDocument(
                title="Missing",
                sentences=("The second fact is here.",),
            ),
        ),
        supporting_document_titles=("Relevant", "Missing"),
    )


def test_evaluates_retrieval_with_batched_embeddings() -> None:
    embeddings_by_text = {
        "Relevant\nThe evidence is here.": vector(1.0, 0.0),
        "Missing\nThe second fact is here.": vector(0.0, 1.0),
        "Where is the evidence?": vector(1.0, 0.0),
    }
    create_embeddings = Mock(
        side_effect=lambda texts: [
            embeddings_by_text[text]
            for text in texts
        ]
    )

    result = evaluate_retrieval_case(
        benchmark_case(),
        limit=1,
        embedding_function=create_embeddings,
    )

    assert result.metrics == RetrievalMetrics(
        limit=1,
        supporting_document_count=2,
        matched_supporting_document_count=1,
        supporting_document_recall=0.5,
        complete_support=False,
        reciprocal_rank=1.0,
    )
    assert result.retrieved_chunks == (
        RetrievedChunk(
            rank=1,
            document_title="Relevant",
            chunk_index=0,
            score=pytest.approx(1.0),
            text="Relevant\nThe evidence is here.",
        ),
    )
    assert create_embeddings.call_args_list == [
        call(
            [
                "Relevant\nThe evidence is here.",
                "Missing\nThe second fact is here.",
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


def test_writes_json_report(tmp_path: Path) -> None:
    case_result = RetrievalCaseResult(
        dataset_id="case-1",
        question="Question?",
        expected_answer="Answer",
        supporting_document_titles=("Document",),
        metrics=RetrievalMetrics(5, 1, 1, 1.0, True, 1.0),
        retrieved_chunks=(),
    )
    report = create_report([case_result])
    output_path = tmp_path / "report.json"

    write_report(report, output_path)

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["config"]["retrieval_limit"] == 5
    assert saved["aggregate"] == {
        "limit": 5,
        "case_count": 1,
        "mean_supporting_document_recall": 1.0,
        "complete_support_rate": 1.0,
        "mean_reciprocal_rank": 1.0,
    }
    assert saved["cases"][0]["dataset_id"] == "case-1"
