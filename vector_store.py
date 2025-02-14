from sentence_transformers import SentenceTransformer
import chromadb
from datetime import datetime
from typing import List

class VectorStore:
    def __init__(self, model_name: str, collection_name: str, collection_description: str, path: str,
                 hnsw_space='cosine', hnsw_construction=100, hnsw_search_ef=100, hnsw_M=100):
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": collection_description,
                "created": str(datetime.now()),
                "hnsw:space": hnsw_space,
                "hnsw:construction_ef": hnsw_construction,
                "hnsw:search_ef": hnsw_search_ef,
                "hnsw:M": hnsw_M
            }
        )

    def add_data(self, IDs: List[str], texts: List[str]):
        if len(IDs) != len(texts):
            raise ValueError("IDs and texts lists must have the same length")

        embeddings = self.model.encode(texts, normalize_embeddings=True).tolist()
        self.collection.add(ids=IDs, embeddings=embeddings, documents=texts)