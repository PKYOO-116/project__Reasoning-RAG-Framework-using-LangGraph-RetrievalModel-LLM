# project__Open-Source-AI-based-Reasoning-RAG-Framework-for-Personal-Web-Integration-using-LangGraph

**MTEB Model:** Qwen3-Embedding-0.6B (STS, Zero-shot, Retrieval, Reranking)
**RTEB Model:** bge-m3 (CUREv1 performance while other performance indexes are great as well)
**Vector DB:** Qdrant
**LLM:** Llama3 (8B-Instruct)
**LLM Engine:** Ollama
**Pipeline framwork:** LangGraph + LangChain
**Environment:** Python 3.12

----------------------------------------- Data Flow -------------------------------------------


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

project_root/
├── requirements.txt
├── rag_graph.ipynb # Jupyter Notebook – step-by-step learning
├── rag_graph.py # Integrated LangGraph pipeline script
└── docs/ # Data & text corpus for RAG
    ├── resume.txt
    ├── projects.md
    └── usc_report.pdf

### Future Project Expansion
1. Add chatBot to website using the pipeline


### Reference

Retrieval Evaluation -
BEIR-A_Heterogeneous_Benchmark_for_Zero-shot_Evaluation_of_Information_Retrieval_Models
https://huggingface.co/spaces/mteb/leaderboard
https://microsoft.github.io/msmarco/
nDCG: https://www.elastic.co/search-labs/blog/improving-information-retrieval-elastic-stack-benchmarking-passage-retrieval?utm_source=chatgpt.com

LLM Evaluation -
https://www.confident-ai.com/blog/rag-evaluation-metrics-answer-relevancy-faithfulness-and-more
deepeval-github: https://github.com/confident-ai/deepeval?tab=readme-ov-file
https://platform.openai.com/docs/guides/evals/evaluating-model-performance-beta?api-mode=responses
https://medium.com/@vishaalini70/evaluating-rag-responses-using-ragas-openai-eval-framework-f3952ee75778
https://docs.cohere.com/page/retrieval-eval-pydantic-ai