import anthropic
import os
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
import openai

import dotenv

dotenv.load_dotenv()
class llm():
    def __init__(self):
        self.model = "gpt-4o"
        self.api_key = os.environ["OPENAI_API_KEY"]
        self.client = openai.OpenAI(api_key=self.api_key)

    def query(self,content):
        messages =  self.client.responses.create(
            model="gpt-4.1",
            input= content
        )

        return messages.output_text

def create_ragas_llm(model:str = "gpt-4o"):
    if "gpt" in model:
        api_key = os.environ["OPENAI_API_KEY"]
        return LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))


