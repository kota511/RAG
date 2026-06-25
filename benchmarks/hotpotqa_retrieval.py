import argparse
import os
from collections.abc import Callable

from qdrant_client import QdrantClient, models

from app.embeddings import create_embeddings
from app.vector_store import (
    COLLECTION_NAME,
    ensure_collection,
    generate_point_id,
)
from benchmarks.hotpotqa_chunks import prepare_case_documents
from benchmarks.hotpotqa_dataset import HotpotQACase, load_hotpotqa_cases
from benchmarks.hotpotqa_metrics import (
    RetrievalMetrics,
    RetrievedChunkReference,
    aggregate_retrieval_metrics,
    calculate_retrieval_metrics,
)

EmbeddingFunction = Callable[[list[str]], list[list[float]]]


def evaluate_retrieval_case(
    case: HotpotQACase,
    limit: int = 5,
    embedding_function: EmbeddingFunction = create_embeddings,
) -> RetrievalMetrics:
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
    chunk_embeddings = embeddings[:-1]
    query_embedding = embeddings[-1]

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
                    },
                )
                for (title, chunk), embedding in zip(
                    chunks,
                    chunk_embeddings,
                    strict=True,
                )
            ],
            wait=True,
        )
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=limit,
            with_payload=True,
        )
    finally:
        client.close()

    retrieved_chunks = [
        RetrievedChunkReference(
            document_title=point.payload["document_title"],
            chunk_index=point.payload["chunk_index"],
        )
        for point in response.points
    ]
    return calculate_retrieval_metrics(
        documents,
        retrieved_chunks,
        limit=limit,
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
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is required. Run with '--env-file .env'."
        )

    cases = load_hotpotqa_cases(count=args.count)
    results: list[RetrievalMetrics] = []
    for index, case in enumerate(cases, start=1):
        metrics = evaluate_retrieval_case(case, limit=args.limit)
        results.append(metrics)
        print(
            f"[{index}/{len(cases)}] {case.dataset_id}: "
            f"recall@{args.limit}={metrics.supporting_fact_recall:.1%}, "
            f"complete={metrics.complete_support}, "
            f"mrr@{args.limit}={metrics.reciprocal_rank:.3f}"
        )

    aggregate = aggregate_retrieval_metrics(results)
    print(
        f"Aggregate over {aggregate.case_count} case(s): "
        f"recall@{aggregate.limit}="
        f"{aggregate.mean_supporting_fact_recall:.1%}, "
        f"complete@{aggregate.limit}="
        f"{aggregate.complete_support_rate:.1%}, "
        f"mrr@{aggregate.limit}="
        f"{aggregate.mean_reciprocal_rank:.3f}"
    )


if __name__ == "__main__":
    main()
