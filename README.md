# project__Open-Source-AI-based-Reasoning-RAG-Framework-for-Personal-Web-Integration-using-LangGraph

MTEB Model: Qwen3-Embedding-0.6B (STS, Zero-shot, Retrieval, Reranking)
RTEB Model: bge-m3 (CUREv1 performance while other performance indexes are great as well)
Vector DB: Qdrant
LLM: Llama3 (8B-Instruct)
LLM Engine: Ollama
Pipeline framwork: LangGraph + LangChain
Environment: Python 3.12



------ Data Flow ------------------------------------------------------------------------------
                 ┌───────────────────────────────────────┐
                 │          LangGraph Pipeline           │
                 └───────────────────────────────────────┘
                               │
                               ▼
        ┌──────────────┐   ┌──────────────┐   ┌────────────────┐    ┌────────────────────┐
        │  Ingest Node │ → │ Embedding    │ → │   Qdrant DB    │ →  │   Retrieval Node   │
        │ (load docs)  │   │ (Qwen3+bge)  │   │ (store vectors)│    │  (semantic search) │
        └──────────────┘   └──────────────┘   └────────────────┘    └────────┬───────────┘
                                                                             │
                                                                             ▼
                                                                    ┌──────────────────────┐
                                                                    │        LLM Node      │
                                                                    │ (Llama 3 via Ollama) │
                                                                    └────────┬─────────────┘
                                                                             │
                                                                             ▼
                                                                      ┌──────────────┐
                                                                      │  Response    │
                                                                      │ (Answer Gen) │
                                                                      └──────────────┘
                                                                    
-----------------------------------------------------------------------------------------------

project_root/
├── .env
├── requirements.txt
├── rag_graph.ipynb        # Jupyter Notebook to learn each level
├── rag_graph.py           # Integrated code running from .ipynb
├── docs/                  # data text files
│    ├── resume.txt
│    ├── projects.md
│    └── usc_report.pdf
├── qdrant_storage/        # Local Qdrant Database
└── webapp/                # Web Front-end
     ├── src/
     ├── pages/chatbot.jsx
     └── api/
         └── rag_api.py (Flask/FastAPI)

Front-end: React + Vite + SCSS (Already built)
Back-end: LangServe / Server: AWS EC2??? (Too slow for Llama 3 to run. Should consider.)

What to do next:
1. edit Profile: Add Education, Leadership & Community Involvement, Skills, Language and Culture, Contact Info, Hobbies