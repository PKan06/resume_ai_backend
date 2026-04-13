from langchain_community.embeddings import HuggingFaceEmbeddings
from server.config import settings


class EmbeddingService:

    _instance = None

    def __init__(self):
        self.model = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": settings.EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True}
        )

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = EmbeddingService()
        return cls._instance.model