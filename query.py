from vector_store import VectorStore
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pandas as pd

class Query:
    def __init__(self, vector_store: VectorStore, model_name: str):
        self.vector_store = vector_store
        self.llm = OllamaLLM(model=model_name)
        self.prompt = PromptTemplate(
        input_variables=["input_key", "similar_keys" "similar_values", "distances"],
        template=(
            "Given the input key: {input_key}, think critically and reason through the retrieved vector search results "
            "to determine the most relevant value. {similar_keys} and {similar_values} are key value mapping. All these mappings are of car models.\n\n"
            "### Retrieved Data:\n"
            "- **Similar Keys:** {similar_keys}\n"
            "- **Similar Values:** {similar_values}\n"
            "- **Cosine Distances:** {distances}\n\n"
            "### Instructions:\n"
            "1. Analyze the retrieved keys, values mapping holistically.\n"
            "2. Prioritize values from entries with the lowest cosine distances (highest relevance).\n"
            "3. If multiple values seem relevant, apply logical reasoning to choose the most contextually appropriate one.\n"
            "4. If no exact match exists, infer the best possible value based on {input_key} and {similar_keys} relationships.\n\n"
            "### Output Format:\n"
            "Return only the most relevant key value pair as a response and reasoning behind it, ensuring it aligns accurately with {input_key} and the given context. Give output as key value pair in json everytime."
        ))
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
        
    def query(self, query: str, k: int):
        result = self.vector_store.search(query, k)
        return {"keys": result['documents'][0], "distances": result['distances'][0], "values": result['metadatas'][0]}
    
    def get_response(self, query: dict):
        response = self.chain.run(
            input_key=query['query'],
            similar_keys=query['keys'],
            similar_values=query['values'],
            distances=query['distances']
        )
        return response



if __name__ == "__main__":
    vector_store = VectorStore("BAAI/bge-small-en-v1.5", "car_model", "Car Model Collection", "/content/import shutil")
    query = Query(vector_store, "deepseek-r1:1.5b")
    input = "aviator"
    retrieved_data = query.query(input, 10)
    print(retrieved_data)
    query_obj = {
        "query": input,
        "keys": retrieved_data['keys'],
        "values": retrieved_data['values'],
        "distances": retrieved_data['distances']
    }
    response = query.get_response(query_obj)
    print(response)


    
