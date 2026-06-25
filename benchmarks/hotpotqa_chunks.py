from dataclasses import dataclass

from app.documents import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    chunk_text,
)
from benchmarks.hotpotqa_dataset import (
    BenchmarkDocument,
    HotpotQACase,
)


@dataclass(frozen=True)
class PreparedChunk:
    chunk_index: int
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class PreparedSupportingFact:
    sentence_index: int
    text: str
    start: int
    end: int
    chunk_indexes: tuple[int, ...]


@dataclass(frozen=True)
class PreparedDocument:
    title: str
    text: str
    chunks: tuple[PreparedChunk, ...]
    supporting_facts: tuple[PreparedSupportingFact, ...]


def prepare_case_documents(
    case: HotpotQACase,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[PreparedDocument, ...]:
    supporting_indexes_by_title: dict[str, set[int]] = {}
    for fact in case.supporting_facts:
        supporting_indexes_by_title.setdefault(fact.title, set()).add(
            fact.sentence_index
        )

    return tuple(
        prepare_document(
            document,
            supporting_sentence_indexes=supporting_indexes_by_title.get(
                document.title,
                set(),
            ),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for document in case.documents
    )


def prepare_document(
    document: BenchmarkDocument,
    supporting_sentence_indexes: set[int],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> PreparedDocument:
    text, sentence_spans = render_document(document)
    chunks = create_chunk_spans(
        text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    supporting_facts = tuple(
        create_prepared_supporting_fact(
            text=text,
            sentence_index=sentence_index,
            sentence_text=document.sentences[sentence_index],
            sentence_span=sentence_spans[sentence_index],
            chunks=chunks,
        )
        for sentence_index in sorted(supporting_sentence_indexes)
    )

    return PreparedDocument(
        title=document.title,
        text=text,
        chunks=chunks,
        supporting_facts=supporting_facts,
    )


def render_document(
    document: BenchmarkDocument,
) -> tuple[str, tuple[tuple[int, int], ...]]:
    text = f"{document.title}\n"
    sentence_spans: list[tuple[int, int]] = []

    for sentence in document.sentences:
        start = len(text)
        text += sentence
        sentence_spans.append((start, len(text)))

    return text, tuple(sentence_spans)


def create_chunk_spans(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[PreparedChunk, ...]:
    chunks = chunk_text(
        text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    step = chunk_size - chunk_overlap

    return tuple(
        PreparedChunk(
            chunk_index=index,
            text=chunk,
            start=index * step,
            end=index * step + len(chunk),
        )
        for index, chunk in enumerate(chunks)
    )


def create_prepared_supporting_fact(
    text: str,
    sentence_index: int,
    sentence_text: str,
    sentence_span: tuple[int, int],
    chunks: tuple[PreparedChunk, ...],
) -> PreparedSupportingFact:
    sentence_start, sentence_end = sentence_span
    chunk_indexes = tuple(
        chunk.chunk_index
        for chunk in chunks
        if chunk_contains_evidence(
            text,
            chunk_start=chunk.start,
            chunk_end=chunk.end,
            evidence_start=sentence_start,
            evidence_end=sentence_end,
        )
    )

    return PreparedSupportingFact(
        sentence_index=sentence_index,
        text=sentence_text,
        start=sentence_start,
        end=sentence_end,
        chunk_indexes=chunk_indexes,
    )


def chunk_contains_evidence(
    text: str,
    chunk_start: int,
    chunk_end: int,
    evidence_start: int,
    evidence_end: int,
) -> bool:
    overlap_start = max(chunk_start, evidence_start)
    overlap_end = min(chunk_end, evidence_end)
    return (
        overlap_start < overlap_end
        and bool(text[overlap_start:overlap_end].strip())
    )
