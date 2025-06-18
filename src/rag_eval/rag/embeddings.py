import os
from langchain_openai import OpenAIEmbeddings

import dotenv
dotenv.load_dotenv()
def create_embedding(text):
    embedding_model = OpenAIEmbeddings(api_key = os.environ.get("OPENAI_API_KEY"), model="text-embedding-3-large")
    if type(text) == list:
        embeddings = []
        for item in text:
            embeddings.append(embedding_model.embed_query(item))
        return embeddings
    else:
        output = embedding_model.embed_query(text)
        return output


