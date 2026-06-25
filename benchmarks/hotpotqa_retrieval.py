import argparse
import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from qdrant_client import QdrantClient, models

from app.documents import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from app.embeddings import EMBEDDING_MODEL, create_embeddings
from app.vector_store import (
    COLLECTION_NAME,
    ensure_collection,
    generate_point_id,
)
from benchmarks.hotpotqa_chunks import prepare_case_documents
from benchmarks.hotpotqa_dataset import (
    DATASET_FILE_PATH,
    DATASET_REPOSITORY,
    DATASET_REVISION,
    DEFAULT_SELECTION_SEED,
    HotpotQACase,
    load_hotpotqa_cases,
)
from benchmarks.hotpotqa_metrics import (
    RetrievalMetrics,
    aggregate_retrieval_metrics,
    calculate_retrieval_metrics,
)

EmbeddingFunction = Callable[[list[str]], list[list[float]]]


@dataclass(frozen=True)
class RetrievedChunk:
    rank: int
    document_title: str
    chunk_index: int
    score: float
    text: str


@dataclass(frozen=True)
class RetrievalCaseResult:
    dataset_id: str
    question: str
    expected_answer: str
    supporting_document_titles: tuple[str, ...]
    metrics: RetrievalMetrics
    retrieved_chunks: tuple[RetrievedChunk, ...]


def evaluate_retrieval_case(
    case: HotpotQACase,
    limit: int = 5,
    embedding_function: EmbeddingFunction = create_embeddings,
) -> RetrievalCaseResult:
    if limit <= 0:
        raise ValueError("limit must be greater than zero")

    documents = prepare_case_documents(case)
    chunks = [
        (document.title, chunk)
        for document in documents
        for chunk in document.chunks
    ]
    embedding_inputs = [chunk.text for _, chunk in chunks] + [case.question]
    embeddings = embedding_function(embedding_inputs)
    if len(embeddings) != len(embedding_inputs):
        raise ValueError("embedding count does not match input count")

    client = QdrantClient(":memory:")
    try:
        ensure_collection(client)
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=generate_point_id(
                        f"{case.dataset_id}:{title}:{chunk.chunk_index}"
                    ),
                    vector=embedding,
                    payload={
                        "document_title": title,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                    },
                )
                for (title, chunk), embedding in zip(
                    chunks,
                    embeddings[:-1],
                    strict=True,
                )
            ],
            wait=True,
        )
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embeddings[-1],
            limit=limit,
            with_payload=True,
        )
    finally:
        client.close()

    retrieved_chunks = tuple(
        RetrievedChunk(
            rank=rank,
            document_title=point.payload["document_title"],
            chunk_index=point.payload["chunk_index"],
            score=point.score,
            text=point.payload["text"],
        )
        for rank, point in enumerate(response.points, start=1)
    )
    metrics = calculate_retrieval_metrics(
        case.supporting_document_titles,
        [chunk.document_title for chunk in retrieved_chunks],
        limit=limit,
    )
    return RetrievalCaseResult(
        dataset_id=case.dataset_id,
        question=case.question,
        expected_answer=case.answer,
        supporting_document_titles=case.supporting_document_titles,
        metrics=metrics,
        retrieved_chunks=retrieved_chunks,
    )


def create_report(
    results: list[RetrievalCaseResult],
    limit: int,
) -> dict[str, object]:
    aggregate = aggregate_retrieval_metrics(
        [result.metrics for result in results]
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "dataset_repository": DATASET_REPOSITORY,
            "dataset_revision": DATASET_REVISION,
            "dataset_file_path": DATASET_FILE_PATH,
            "selection_seed": DEFAULT_SELECTION_SEED,
            "case_count": len(results),
            "retrieval_limit": limit,
            "embedding_model": EMBEDDING_MODEL,
            "chunk_size": DEFAULT_CHUNK_SIZE,
            "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
        },
        "aggregate": asdict(aggregate),
        "cases": [asdict(result) for result in results],
    }


def write_report(report: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run live semantic retrieval against pinned HotpotQA cases."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of cases to run; defaults to 1 to limit API usage.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of chunks retrieved per question.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/hotpotqa_retrieval.json"),
        help="JSON output path.",
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is required. Run with '--env-file .env'."
        )

    cases = load_hotpotqa_cases(count=args.count)
    results = [
        evaluate_retrieval_case(case, limit=args.limit)
        for case in cases
    ]
    for index, result in enumerate(results, start=1):
        metrics = result.metrics
        print(
            f"[{index}/{len(results)}] {result.dataset_id}: "
            f"document recall@{args.limit}="
            f"{metrics.supporting_document_recall:.1%}, "
            f"complete={metrics.complete_support}, "
            f"mrr@{args.limit}={metrics.reciprocal_rank:.3f}"
        )

    report = create_report(results, limit=args.limit)
    aggregate = report["aggregate"]
    print(
        f"Aggregate over {aggregate['case_count']} case(s): "
        f"document recall@{aggregate['limit']}="
        f"{aggregate['mean_supporting_document_recall']:.1%}, "
        f"complete@{aggregate['limit']}="
        f"{aggregate['complete_support_rate']:.1%}, "
        f"mrr@{aggregate['limit']}="
        f"{aggregate['mean_reciprocal_rank']:.3f}"
    )
    write_report(report, args.output)
    print(f"Saved detailed results to {args.output}")


if __name__ == "__main__":
    main()
