from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"


def create_embeddings(texts: list[str]) -> list[list[float]]:
    client = OpenAI()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    ordered_embeddings = sorted(response.data, key=lambda item: item.index)
    return [item.embedding for item in ordered_embeddings]
