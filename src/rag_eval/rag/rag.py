import os
from rag_eval.rag.embeddings import create_embedding
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma

from langchain.prompts import ChatPromptTemplate
from rag_eval.utils.client import llm
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

prompt_template = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:
{context}
Answer the question based on the above context: {question}.
Provide a detailed answer.
Don’t justify your answers.
Don’t give information not mentioned in the CONTEXT INFORMATION.
Do not say "according to the context" or "mentioned in the context" or similar.
""")


class rag():
    def __init__(self,chroma_dir=None):
       
        self.llm = llm()
        self.prompt_template = prompt_template
        self.vectordb = Chroma(
            collection_name="rag_eval",
            embedding_function= OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY")),
            persist_directory="./chroma_langchain_db"   
        )
    def chunk_text(self,documents):
        if type(documents) == list:
            return [Document(page_content=item) for item in documents]
        else:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=300, # Size of each chunk in characters
                chunk_overlap=100, # Overlap between consecutive chunks
                length_function=len, # Function to compute the length of the text
                add_start_index=True, # Flag to add start index to each chunk
            )
            return Document(page_content=text_splitter.split_documents(documents))

    def process_docs_to_vectorstore(self,documents):
        chunks = self.chunk_text(documents)
        id = []
        for item in range(len(chunks)):
            id.append(f"{item}")
        #embeddings = create_embedding(chunks, self.api_key)

        self.vectordb.add_documents(documents=chunks, ids = id)

    def retrieve_docs(self,query,k=5):
        docs = self.vectordb.similarity_search_with_score(query, k=5)
        return docs
    
    def query(self,question):

        docs = self.retrieve_docs(question)
       
        if type(docs) == list:
            context_text = "\n\n".join([item[0].page_content for item in docs])
        else:
            context_text = "\n\n".join([doc.page_content for doc, _score in docs])
        prompt = self.prompt_template.format(context=context_text, question=question)

        return self.llm.query(prompt)

