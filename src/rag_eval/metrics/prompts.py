from langchain.prompts import ChatPromptTemplate

contextual_precision_prompt = ChatPromptTemplate.from_template("""
<instructions>
You are an evaluator in a Retrieval-Augmented Generation (RAG) system. 
Your task is to assess whether each retrieved text chunk is contextually relevant to the user's
query, based on how well it supports or contributes to the generated answer.
<instructions>

<requirements>
<format>
You will receive:

1. A user query
2. A generated answer (based on the retrieved chunks)
3. A list of retrieved text chunks


Your job is to decide, for each chunk, whether it was actually useful or necessary in producing
the answer. Focus only on the relevance of the content in the chunk with respect to the
answer and the query, a chunk can be partially relevant, if it only contains part of the answer, its still correct.

Classification:
Respond 1 if the chunk contains information that supports, justifies, or directly contributes to the generated answer, the chunk can be partially relevent, it does not need to answer the entire question, it can answer part of it.
Respond 0.5 if the chunk is related to the topic or query but not clearly used in the answer.
Respond 0 if the chunk does not contain information that clearly helps answer the query or is not reflected in the final answer.
<format>
                                                           
<format_rules>
{{"classification": ["a list containing the retrieval relevent to the query?"], "explanation": ["a list of how you came to that response for each chunk given to you"]}}
<requirements>

<input>
query: {query}
response : {response}
retrieval list: {retrieval_list}
<input>""")

contextual_recall_prompt = ChatPromptTemplate.from_template(
"""
<instructions>
You are an evaluator in a retrieval_augmented generation system.
Your task is to assess the ability of a rag system to retrieve all neceessary information from an external source to generate
a complete and accurate response to a query. You must make sure that no crucial detail is missed from a ground truth answer in the information retrieved by the rag system
<instructions>
<requirements>
<format>
You will recieve: 

1. A user query
2. a ground truth answer 
3. a list of retrieved text chunks

Your job is to generate a list of claims made by the ground truth answer, 
then for each claim determine if there is a relevent chunk in the list of retrieved context
given to you. Do not simply repeat the sentence as the claim. An example of how to break down a sentence is below:

<example>
'Governor John R. Rogers High School, named after John Rankin Rogers, is located in the Puyallup School District of Washington, United States.'

This sentence makes several claims:
1. Governor John R. Rogers High School is named after John Rankin Rogers
2. The School is located in Puyallup School District of Washington, which is in the united states. 
<example>

Classification:
Respond 1 if the claim has a chunk that contains relevent information, that the chunk supports, justifies or directly contributest to the claim.
Respond 0.5 if the claim has a chunk that is only partially related, but not clearly used in the answer
Respond 0 if a claim is not supported at all by any chunk in the list of retrieved context
<format>

<format_rules>
{{"claim": ["a list of each claim in the ground truth answer"], "classification": ["a list containing the score related to each claim"], "explanation": ["Why each claim is scored the way it is"]}}
<format_rules>
<requirements>

<input>
query: {query}
ground truth answer to query: {ground_truth}
retrieval list: {retrieval_list}
<input>"""
)