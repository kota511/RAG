from benchmarks.hotpotqa_chunks import (
    PreparedChunk,
    PreparedDocument,
    PreparedSupportingFact,
    prepare_case_documents,
)
from benchmarks.hotpotqa_dataset import (
    BenchmarkDocument,
    HotpotQACase,
    SupportingFact,
)


def test_prepares_ingestion_text_chunks_and_support_mapping() -> None:
    case = HotpotQACase(
        dataset_id="case-1",
        question="What is the supporting fact?",
        answer="BBBB",
        question_type="bridge",
        level="medium",
        documents=(
            BenchmarkDocument(
                title="Doc",
                sentences=("AAAA", " BBBB", " CCCC"),
            ),
        ),
        supporting_facts=(
            SupportingFact(
                title="Doc",
                sentence_index=1,
                text=" BBBB",
            ),
        ),
    )

    documents = prepare_case_documents(
        case,
        chunk_size=10,
        chunk_overlap=2,
    )

    assert documents == (
        PreparedDocument(
            title="Doc",
            text="Doc\nAAAA BBBB CCCC",
            chunks=(
                PreparedChunk(
                    chunk_index=0,
                    text="Doc\nAAAA B",
                    start=0,
                    end=10,
                ),
                PreparedChunk(
                    chunk_index=1,
                    text=" BBBB CCCC",
                    start=8,
                    end=18,
                ),
            ),
            supporting_facts=(
                PreparedSupportingFact(
                    sentence_index=1,
                    text=" BBBB",
                    start=8,
                    end=13,
                    chunk_indexes=(0, 1),
                ),
            ),
        ),
    )


def test_maps_only_labelled_supporting_sentences() -> None:
    case = HotpotQACase(
        dataset_id="case-1",
        question="Which document has evidence?",
        answer="First",
        question_type="comparison",
        level="easy",
        documents=(
            BenchmarkDocument(
                title="First",
                sentences=("Evidence.", " Extra."),
            ),
            BenchmarkDocument(
                title="Second",
                sentences=("Distractor.",),
            ),
        ),
        supporting_facts=(
            SupportingFact(
                title="First",
                sentence_index=0,
                text="Evidence.",
            ),
        ),
    )

    first, second = prepare_case_documents(case)

    assert [fact.sentence_index for fact in first.supporting_facts] == [0]
    assert second.supporting_facts == ()
