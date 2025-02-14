# Car Model Vector Search System

A semantic search system for car models using vector embeddings and LLM-powered response generation.

## Overview

This project implements a vector similarity search system for car models using sentence transformers and ChromaDB. It includes capabilities to:

- Create and store vector embeddings for car model data
- Perform semantic similarity searches
- Generate contextual responses using LLMs (via Ollama)

## Technologies Used

- Python 3.x
- ChromaDB (Vector Database)
- Sentence Transformers (BAAI/bge-small-en-v1.5)
- Ollama (Local LLM integration)
- Pandas (Data handling)
- LangChain (LLM orchestration)

## Project Structure

```
.
├── embeddings.py
├── query.py
├── vector_store.py
├── README.md
├── LICENSE.txt
├── requirements.txt
├── .gitignore
```

## Setup

1. Create a virtual environment:

```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install Ollama and download required model:

```bash
ollama pull deepseek-r1:1.5b
```

4. Prepare your data:

- Place your car model dataset in `train.csv`
- Required columns: Id, Key, Value

## Usage

### 1. Generate Embeddings

Run the embedding generation script:

```bash
python embeddings.py
```

This will:

- Load data from train.csv
- Create vector embeddings
- Store them in ChromaDB

### 2. Perform Queries

Use the query system:

```python
from vector_store import VectorStore
from query import Query

# Initialize
vector_store = VectorStore("BAAI/bge-small-en-v1.5", "car_model", "Car Model Collection", "./vector_embedding")
query = Query(vector_store, "deepseek-r1:1.5b")

# Search
input_query = "aviator"
retrieved_data = query.query(input_query, 10)

# Get LLM-enhanced response
query_obj = {
    "query": input_query,
    "keys": retrieved_data['keys'],
    "values": retrieved_data['values'],
    "distances": retrieved_data['distances']
}
response = query.get_response(query_obj)
```

## Configuration

The vector store can be configured with the following parameters:

- `hnsw_space`: Vector similarity metric (default: 'cosine')
- `hnsw_construction`: HNSW construction parameter (default: 100)
- `hnsw_search_ef`: HNSW search parameter (default: 100)
- `hnsw_M`: HNSW M parameter (default: 100)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is open-sourced under the MIT License - see the LICENSE file for details.
