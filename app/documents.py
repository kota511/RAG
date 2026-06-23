def extract_text_from_bytes(contents: bytes) -> str:
    return contents.decode("utf-8")


def chunk_text(text: str, chunk_size: int = 100) -> list[str]:
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]