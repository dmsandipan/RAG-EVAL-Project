#TODO: implement llm evaluation for contextual precision and recall
from rag_eval.metrics.similarity_metrics import jaccard_similarity, cosine_similarity
from rag_eval.rag.embeddings import create_embedding

from rag_eval.metrics.evaluator import evaluation_llm
import ast

from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithoutReference
from ragas.metrics import LLMContextPrecisionWithReference
from ragas.metrics import LLMContextRecall

from rag_eval.utils.client import create_ragas_llm


def contextual_precision_no_llm(question:str, retrieved_context: list, reference:list = None,threshold:float = 0.5) -> float:
    # get jaccard similarity score for each question, context pair
    context_relevance = []
    rank_sum = 0
    for iter,context in enumerate(retrieved_context,start=1):
        if jaccard_similarity(question,context) >threshold:
            rank_sum +=1
            context_relevance.append(rank_sum/iter)

    if len(retrieved_context) >0:
        return sum(retrieved_context) / len(retrieved_context)
    else:
        return 0
    
def contextual_precision_llm(query:str, response:str ,retrieval_list:list,ground_truth:list = None):
    
    #if reference:
    evaluator_llm = evaluation_llm()
    result = evaluator_llm.evaluate_context_precision(query=query,response=response,retrieval_list=retrieval_list)
    result = ast.literal_eval(result)    
    score_list = []
    precision_k = []
    average_precision = []
    for score in result["classification"]:
        
        score = float(score)
        if score >0:
            score_list.append(1)
        else:
            score_list.append(0)
        
        precision_k.append(sum(score_list) / len(score_list))
        
        if score >0:
            average_precision.append(precision_k[-1])
    
    if len(average_precision) < 1:
        return 0
    else:
        return sum(average_precision) / len(average_precision) , result

    
def contextual_recall_non_llm(retrieved_context:list, reference_context:list) ->float:
    output = 0 
    for context in retrieved_context:
        if context in reference_context:
            output +=1
    if output > 0:
        return output/ len(output)
    else:
        return 0
    
def contextual_recall_llm(query:str, ground_truth:str ,retrieval_list = list)->float:
    
    evaluator_llm = evaluation_llm()
    result = evaluator_llm.evaluate_context_recall(query=query,ground_truth =ground_truth,retrieval_list=retrieval_list)
    result = ast.literal_eval(result) 

    return sum(result["classification"]) /len(result["classification"]), result
    
  
def contextual_relevancy(query:str,retrieval_list:list)-> float:

    
    question_embedding =  create_embedding(query)   
    sim_scores = []
    for context in retrieval_list:
        context_embedding = create_embedding(context)
        sim_scores.append(cosine_similarity(question_embedding, context_embedding))
    return sum(sim_scores) / len(sim_scores)

def mrr(retrieval_list: list [str], reference_list:list [str]) ->float:
    if retrieval_list[0] in reference_list:
        position = reference_list.index(retrieval_list[0]) +1
        score = 1/position
        return score
    else:
        return 0 
    
def evaluate_output(query:str, response:str =None, retrieval_list:list = None, ground_truth:str=None,reference_list:list= None ):
    output_dict = {}


    output_dict["contextual_precision"] = contextual_precision_llm(
        query=query,
        response= response, 
        retrieval_list=retrieval_list,
        ground_truth=ground_truth
        )
    if ground_truth:
        output_dict["contextual_recall"] = contextual_recall_llm(
            query=query,
            ground_truth=ground_truth,
            retrieval_list= retrieval_list
            )
    
    output_dict["contextual_relevancy"] = float(contextual_relevancy(
        query=query,
        retrieval_list=retrieval_list
        ))
    if reference_list:
        output_dict["mmr"] = mrr(
            retrieval_list=retrieval_list,
            reference_list= reference_list
            )
    
    return output_dict




def ragas_contextual_precision_llm(question:str,retrieved_context:list,reference:list = None):
    
    if reference:
        evaluator_llm = create_ragas_llm()
        context_precision = LLMContextPrecisionWithReference(llm=evaluator_llm)
        sample = SingleTurnSample(
            user_input = question,
            reference = reference,
            retrieved_contexts = retrieved_context)
        
    else:
        evaluator_llm = create_ragas_llm()
        context_precision = LLMContextPrecisionWithoutReference(llm=evaluator_llm)
        sample = SingleTurnSample(
            user_input = question,
            retrieved_contexts = retrieved_context)
        
    return context_precision.single_turn_score(sample)

def ragas_contextual_recall_llm(question:str, response:str, reference:str ,retrieved_context = list)->float:
    evaluator_llm = create_ragas_llm()
    sample = SingleTurnSample(
        user_input=question, 
        response= response, 
        reference= reference,
        retrieved_contexts=retrieved_context
    )
    context_recall = LLMContextRecall(llm=evaluator_llm)
    return context_recall.single_turn_score(sample)