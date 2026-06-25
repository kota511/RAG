from benchmarks.hotpotqa_chunks import (
    PreparedChunk,
    PreparedDocument,
    prepare_case_documents,
)
from benchmarks.hotpotqa_dataset import BenchmarkDocument, HotpotQACase


def test_prepares_documents_with_application_chunking() -> None:
    case = HotpotQACase(
        dataset_id="case-1",
        question="Question?",
        answer="Answer",
        documents=(
            BenchmarkDocument(
                title="Doc",
                sentences=("AAAA", " BBBB", " CCCC"),
            ),
        ),
        supporting_document_titles=("Doc",),
    )

    documents = prepare_case_documents(
        case,
        chunk_size=10,
        chunk_overlap=2,
    )

    assert documents == (
        PreparedDocument(
            title="Doc",
            chunks=(
                PreparedChunk(0, "Doc\nAAAA B"),
                PreparedChunk(1, " BBBB CCCC"),
            ),
        ),
    )
