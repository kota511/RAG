import re
import uuid

DEFAULT_CHUNK_SIZE = 100
DEFAULT_CHUNK_OVERLAP = 20


def extract_text_from_bytes(contents: bytes) -> str:
    return contents.decode("utf-8")


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between zero and chunk_size")

    chunks: list[str] = []
    step = chunk_size - chunk_overlap

    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size]
        chunks.append(chunk)

        if start + chunk_size >= len(text):
            break

    return chunks


def generate_document_id() -> str:
    return str(uuid.uuid4())


def generate_chunk_id(document_id: str, chunk_index: int) -> str:
    return f"{document_id}:{chunk_index}"


def calculate_lexical_score(query: str, text: str) -> int:
    query_words = set(re.findall(r"\w+", query.casefold()))
    text_words = set(re.findall(r"\w+", text.casefold()))

    return len(query_words & text_words)
