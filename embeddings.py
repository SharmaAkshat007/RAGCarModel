import pandas as pd
from vector_store import VectorStore
from tqdm import tqdm

if __name__ == "__main__":
    df = pd.read_csv("train.csv")
    df["Id"] = df["Id"].astype(str)
    ids = df["Id"].tolist()
    keys = df["Key"].tolist()
    values = df['Value'].tolist()
    values_dict = []
    for index in range(len(values)):
        value = {"value": values[index]}
        values_dict.append(value)
    vector_store = VectorStore("BAAI/bge-small-en-v1.5", "car_model", "Car Model Collection", "./vector_embedding", 'cosine', 500, 500, 500)
    batch_len = 32
    for i in tqdm(range(0, len(ids), batch_len)):
        batch_ids = ids[i:i+batch_len]
        batch_keys = keys[i:i+batch_len]
        batch_values = values_dict[i:i+batch_len]
        vector_store.add_data(batch_ids, batch_keys, batch_values)