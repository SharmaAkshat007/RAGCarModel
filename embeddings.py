from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
import chromadb
from tqdm import tqdm
from datetime import datetime
from typing import List, Union
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
import pandas as pd
import numpy as np


class CustomOllamaEmbeddingFunction(OllamaEmbeddingFunction):
    def __init__(self, url: str = "http://localhost:11434", model_name: str = "nomic-embed-text:latest"):
        super().__init__(url=url, model_name=model_name)
        # Create a new session with increased timeout
        self._session = httpx.Client(timeout=180.0)  # 60 seconds timeout
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text with retry logic."""
        try:
            response = self._session.post(
                f"{self._api_url}/api/embeddings",
                json={"model": self._model_name, "prompt": text},
                timeout=180.0  # Explicit timeout for this request
            )
            response.raise_for_status()
            
            result = response.json()
            if 'embedding' in result:
                return result['embedding']
            else:
                raise ValueError(f"No embedding found in response: {result}")
                
        except httpx.TimeoutException as e:
            raise RuntimeError(f"Timeout while connecting to Ollama API. Please check if Ollama is running and responsive: {str(e)}")
        except httpx.RequestError as e:
            raise RuntimeError(f"Failed to connect to Ollama API. Please check if Ollama is running: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error while getting embedding: {str(e)}")

    def __call__(self, input: Union[str, List[str]]) -> List[List[float]]:
        texts = input if isinstance(input, list) else [input]
        embeddings = []
        
        for text in texts:
            try:
                embedding = self._get_embedding(text)
                embeddings.append(embedding)
            except Exception as e:
                raise RuntimeError(f"Failed to get embedding for text '{text[:50]}...': {str(e)}")
            
        return embeddings


class VectorStore:
    def __init__(self, model_name, collection_name, collection_description, path, hnsw_space = 'cosine', hnsw_construction = 100, hnsw_search_ef = 100, hnsw_M = 100):
        self.model = CustomOllamaEmbeddingFunction(
            url = "http://localhost:11434", 
            model_name = model_name
        )
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name = collection_name, 
            embedding_function = self.model, 
            metadata = {
                "description": collection_description, 
                "created": str(datetime.now()),
                "hnsw:space": hnsw_space,
                "hnsw:construction_ef": hnsw_construction,
                "hnsw:search_ef": hnsw_search_ef,
                "hnsw:M": hnsw_M
            }
        )    
    def add_data(self, ID: str, text: str):
        embedding = self.create_embedding(text)
        self.collection.add(
            ids = [ID],
            embeddings = [embedding],
            documents = [text]
        )
    def create_embedding(self, text: str):
        return self.model(text)[0]
    def query(self, embedding, n_results = 5):
        return self.collection.query(
            query_embeddings = [embedding],
            n_results = n_results
        )
    

if __name__ == '__main__':
    print("Reading data")
    df = pd.read_csv("train.csv")
    df["Id"] = df["Id"].astype(str)
    ids = np.array(df["Id"])
    keys = np.array(df["Key"])
    vector_store = VectorStore("nomic-embed-text:latest", "car_model", "Car Model Collection", "./vector-embeddings", 'cosine', 5000, 5000, 5000)
    print("Adding data to vector store")
    for i in  tqdm(range(99, ids.shape[0])):
        vector_store.add_data(ids[i], keys[i])
        print(f"Added {i}")