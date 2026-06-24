import math

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


def cosine_similarity(first: list[float], second: list[float]) -> float:
    if len(first) != len(second):
        raise ValueError("vectors must have the same dimensions")

    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm == 0 or second_norm == 0:
        return 0.0

    dot_product = sum(
        first_value * second_value
        for first_value, second_value in zip(first, second)
    )
    return dot_product / (first_norm * second_norm)
