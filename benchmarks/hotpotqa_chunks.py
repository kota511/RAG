from dataclasses import dataclass

from app.documents import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    chunk_text,
)
from benchmarks.hotpotqa_dataset import HotpotQACase


@dataclass(frozen=True)
class PreparedChunk:
    chunk_index: int
    text: str


@dataclass(frozen=True)
class PreparedDocument:
    title: str
    chunks: tuple[PreparedChunk, ...]


def prepare_case_documents(
    case: HotpotQACase,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[PreparedDocument, ...]:
    prepared_documents = []
    for document in case.documents:
        text = f"{document.title}\n{''.join(document.sentences)}"
        chunks = chunk_text(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        prepared_documents.append(
            PreparedDocument(
                title=document.title,
                chunks=tuple(
                    PreparedChunk(index, chunk)
                    for index, chunk in enumerate(chunks)
                ),
            )
        )

    return tuple(prepared_documents)
