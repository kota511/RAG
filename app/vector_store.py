import os
import uuid

from qdrant_client import QdrantClient, models

COLLECTION_NAME = "document_chunks"
EMBEDDING_SIZE = 1536
DEFAULT_QDRANT_URL = "http://localhost:6333"


def create_qdrant_client() -> QdrantClient:
    return QdrantClient(url=os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL))


def ensure_collection(client: QdrantClient) -> None:
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=EMBEDDING_SIZE,
                distance=models.Distance.COSINE,
            ),
        )


def reset_collection(client: QdrantClient) -> None:
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    ensure_collection(client)


def generate_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))
