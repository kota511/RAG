import uuid


def extract_text_from_bytes(contents: bytes) -> str:
    return contents.decode("utf-8")


def chunk_text(text: str, chunk_size: int = 100) -> list[str]:
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def generate_document_id() -> str:
    return str(uuid.uuid4())


def generate_chunk_id(document_id: str, chunk_index: int) -> str:
    return f"{document_id}:{chunk_index}"
