from langchain.prompts import ChatPromptTemplate
from rag_eval.utils.client import llm
from rag_eval.metrics.prompts import contextual_precision_prompt, contextual_recall_prompt

class evaluation_llm():

    def __init__(self):
        self.llm = llm()
        self.context_precision = contextual_precision_prompt
        self.context_recall = contextual_recall_prompt
     
    def evaluate_context_precision(self,query,response,retrieval_list):
        prompt = self.context_precision.format(query=query, response=response,retrieval_list=retrieval_list)

        return self.llm.query(prompt)
    def evaluate_context_recall(self,query,ground_truth,retrieval_list):
        prompt = self.context_recall.format(query=query, ground_truth = ground_truth, retrieval_list= retrieval_list)
        return self.llm.query(prompt)