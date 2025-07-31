# RAG-EVAL-Project
# RAG Evaluation on WikiQA Dataset
This repository evaluates Retrieval Augmented Generation (RAG) models on the WikiQA dataset using the RAGAS library. It demonstrates loading, preprocessing, and evaluating a RAG model using metrics like faithfulness, context precision, context recall, and answer relevancy.
## Usage
1. Clone this repository.
2. Install requirements: 
`pip install ragas datasets langchain transformers`
3. Set up your OpenAI API key (if using OpenAI model).
4. Run the code in Colab or your local environment.
5. Download the `updated_dataset.csv` to view results.
## Metrics
- **Faithfulness:** Response alignment with context.
- **Context Precision:** Relevance of context to question.
- **Context Recall:** Proportion of relevant context retrieved.
- **Answer Relevancy:** Relevance of response to question.
## Further Exploration
- Try different RAG models (e.g., LongT5).
- Explore other RAGAS metrics.
- Evaluate on different datasets.
## Enrichment and Research Contributions
Feel free to enrich this repository! Suggestions include:
- Adding new research findings or literature relevant to Retrieval Augmented Generation (RAG) and RAG evaluation
- Sharing code snippets for novel evaluation pipelines or RAG model improvements
- Posting Jupyter notebooks or scripts demonstrating experimental RAG setups
- Contributing datasets or links to public RAG evaluation resources
- Adding explanations of metrics, best practices, or research insights
Your input helps make this project a stronger resource for both practitioners and researchers. Pull requests and research-driven discussions are welcome!
## Useful Links and Resources
- **RAGAS library:** https://github.com/explodinggradients/ragas
- **Facebook AI RAG paper:** https://arxiv.org/abs/2005.11401
- **HuggingFace RAG documentation:** https://huggingface.co/docs/transformers/model_doc/rag
- **Awesome RAG (curated list):** https://github.com/ramanprithivi/awesome-rag
- **Public datasets for RAG evaluation:** https://huggingface.co/datasets/wiki_qa
- **Blog: Introduction to Retrieval Augmented Generation:** https://towardsdatascience.com/introduction-to-retrieval-augmented-generation-rag-81c6fa8a7e20
## Contributing
Contributions are welcome! Open a pull request for issues or new features.
