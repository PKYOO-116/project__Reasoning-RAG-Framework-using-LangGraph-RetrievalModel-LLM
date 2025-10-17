# Reasoning RAG Framework using LangGraph, Retrieval Model, and LLM

Used open-source tools to build LangGraph RAG pipeline <br>

## Model / Framework / Database / Environment

**MTEB Model:** Qwen3-Embedding-0.6B (STS, Zero-shot, Retrieval, Reranking) <br>
**RTEB Model:** bge-m3 (CUREv1 performance while other performance indexes are great as well) <br>
**Vector DB:** Qdrant <br>
**LLM:** Llama3 (8B-Instruct) <br>
**LLM Engine:** Ollama <br>
**Pipeline framwork:** LangGraph + LangChain <br>
**Environment:** Python 3.12 <br>

## Data Flow -----------------------------------------------


                        < LangGraph RAG Pipeline >

                                    │
                                    ▼
                  ┌────────────────────────────────────┐
                  │            Ingest Node             │
                  │       (Load & Chunk Documents)     │
                  └────────────────────────────────────┘
                                    │
                                    ▼
                  ┌────────────────────────────────────┐
                  │           Embedding Node           │
                  │        (Qwen3 + bge Dual Vec)      │
                  └────────────────────────────────────┘
                                    │
                                    ▼
                  ┌────────────────────────────────────┐
                  │           Qdrant Vector DB         │
                  │      (Local Storage / Search)      │
                  └────────────────────────────────────┘
                                    │
                                    ▼
                  ┌────────────────────────────────────┐
                  │        Retrieval Evaluator         │
                  │    (nDCG, MRR, Query Rewrite)      │
                  └────────────────────────────────────┘
                                    │
                                    ▼
                  ┌────────────────────────────────────┐
                  │           LLM Generator            │
                  │         (Llama3 via Ollama)        │
                  └────────────────────────────────────┘
                                    │
                                    ▼
                  ┌────────────────────────────────────┐
                  │     Answer Evaluator (Tier 1)      │
                  │     Faithfulness / Relevancy       │
                  └────────────────────────────────────┘
                                    │
                                    ▼
                  ┌────────────────────────────────────┐
                  │     Answer Evaluator (Tier 2)      │
                  │     Grammar / Fluency / Bias       │
                  └────────────────────────────────────┘
                                    │
                                    ▼
                  ┌────────────────────────────────────┐
                  │        Final Verified Answer       │
                  └────────────────────────────────────┘
                                                                    
-----------------------------------------------------------------------------------------------

## File Structure

    project_root/
    ├── requirements.txt
    ├── rag_graph.ipynb # Jupyter Notebook – step-by-step learning
    ├── rag_graph.py # Integrated LangGraph pipeline script
    └── docs/ # Data & text corpus for RAG
        ├── resume.txt
        ├── projects.md
        └── usc_report.pdf

## Future Project Expansion
1. **Front-end Integration:** Integrate an Interactive Chat Interface powered by the Reasoning-RAG pipeline for seamless real-time Q&A on the personal website.
2. **Evaluation Enhancement:** Add multi-aspect evaluation (e.g., Ragas, GPT-Judge, or pairwise comparative scoring)
3. **Memory Augmentation:** Incorporate Long-term Memory Node in LangGraph
4. **Deployment Scaling:** Migrate to LangServe + FastAPI for scalable backend deployment
5. **Model Adaptation:** Fine-tune or LoRA-adapt Llama3-8B on domain-specific documents
6. **Analytics & Monitoring:** Add metrics dashboard (response latency, Faithfulness trend, retry rates)

## References

[1] S. Thakur *et al.*, “BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models,” *Advances in Neural Information Processing Systems (NeurIPS)*, 2021. [Online]. Available: https://github.com/beir-cellar/beir

[2] Hugging Face, “Massive Text Embedding Benchmark (MTEB) Leaderboard,” *Hugging Face Spaces*, 2024. [Online]. Available: https://huggingface.co/spaces/mteb/leaderboard

[3] Microsoft Research, “MS MARCO: Microsoft Machine Reading Comprehension Dataset,” *Microsoft Research Project Page*, 2018. [Online]. Available: https://microsoft.github.io/msmarco/

[4] Elastic Team, “Improving Information Retrieval with Elastic Stack Benchmarking — Passage Retrieval,” *Elastic Search Labs Blog*, 2023. [Online]. Available: https://www.elastic.co/search-labs/blog/improving-information-retrieval-elastic-stack-benchmarking-passage-retrieval

[5] F. Diaz and A. Ferraro, “Offline Retrieval Evaluation Without Evaluation Metrics,” in *Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’22)*, Madrid, Spain, Jul. 2022. DOI: 10.1145/3477495.3532033

[6] A. J. Smola and V. Vapnik, “Support Vector Regression Machines,” in *Efficient Learning Machines: Theories, Concepts, and Applications for Engineers and System Designers*, Springer, 2014, pp. 93–123. DOI: 10.1007/978-3-319-03979-0_4

[7] Confident AI, “RAG Evaluation Metrics: Answer Relevancy, Faithfulness, and More,” *Confident AI Blog*, 2024. [Online]. Available: https://www.confident-ai.com/blog/rag-evaluation-metrics-answer-relevancy-faithfulness-and-more

[8] Confident AI, “DeepEval: An Open-Source Framework for LLM Evaluation,” *GitHub Repository*, 2024. [Online]. Available: https://github.com/confident-ai/deepeval

[9] OpenAI, “Evaluating Model Performance (Evals Beta),” *OpenAI Documentation*, 2024. [Online]. Available: https://platform.openai.com/docs/guides/evals/evaluating-model-performance-beta

[10] V. Alini, “Evaluating RAG Responses using Ragas & OpenAI Eval Framework,” *Medium Article*, 2024. [Online]. Available: https://medium.com/@vishaalini70/evaluating-rag-responses-using-ragas-openai-eval-framework-f3952ee75778

[11] Cohere AI, “Retrieval Evaluation — Pydantic AI Integration,” *Cohere Documentation*, 2024. [Online]. Available: https://docs.cohere.com/page/retrieval-eval-pydantic-ai
