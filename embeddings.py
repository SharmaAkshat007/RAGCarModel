import pandas as pd
from vector_store import VectorStore
from tqdm import tqdm

if __name__ == "__main__":
    df = pd.read_csv("train.csv")
    df["Id"] = df["Id"].astype(str)
    ids = df["Id"].tolist()
    keys = df["Key"].tolist()
    vector_store = VectorStore("BAAI/bge-small-en-v1.5", "car_model", "Car Model Collection", "./vector-embeddings", 'cosine', 500, 500, 500)
    batch_size = 32
    for i in tqdm(range(0, len(ids), batch_size)):
        batch_ids = ids[i:i + batch_size]
        batch_keys = keys[i:i + batch_size]
        vector_store.add_data(batch_ids, batch_keys)