"""
LangGraph-style RAG Chatbot Pipeline
Author: Paul K. Yoo (pkyoo116)
Version: 1.0
Last Updated: 2025-10-17

Description:
This script implements a Reasoning-RAG pipeline using LangGraph-style structuring.
It supports:
 - PDF document loading and chunking
 - Dual embedding with Qwen3-Embedding-0.6B (MTEB) + bge-m3 (RTEB)
 - Qdrant-based vector retrieval
 - Llama3 (via Ollama) for generation, rewriting, and evaluation
 - Two-tier evaluation: DeepEval (faithfulness/relevancy) and custom linguistic/ethical metrics

Usage:
1. Prepare /docs folder containing .pdf documents
2. Ensure Qdrant Docker is running on localhost:6333
3. Run this script once to initialize embeddings
4. After initialization, you can chat interactively
"""

import fitz, hashlib, torch, uuid, os, re, textwrap
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from langchain_community.llms import Ollama
from deepeval.models import OllamaModel
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, BaseMetric
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
import numpy as np
from sklearn.metrics import ndcg_score


class RAGState(BaseModel):
    docs: List[str] = Field(default_factory=list)
    chunks: List[Dict[str, Any]] = Field(default_factory=list)
    query: Optional[str] = None
    results: List[Dict[str, Any]] = Field(default_factory=list)
    answer: Optional[str] = None
    retry_count: int = 0
    status: Optional[str] = None


# Document Loading and Chunking
def _hash_text(t: str) -> str:
    return hashlib.sha256(t.strip().encode("utf-8")).hexdigest()


def load_and_chunk(state: RAGState, folder: str = "docs",
                   chunk_size_tokens: int = 350,
                   chunk_overlap_tokens: int = 50) -> RAGState:
    texts, chunks, seen_hashes = [], [], set()
    print("\n\033[1;42m--- Loading Docs and Chunking ---\033[0m")

    for file in Path(folder).rglob("*.pdf"):
        try:
            with fitz.open(file) as pdf:
                for i, page in enumerate(pdf, start=1):
                    text = page.get_text("text")
                    if not text.strip():
                        continue
                    h = _hash_text(text)
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)
                    texts.append(text)

                    if TOKEN_SPLIT_AVAILABLE:
                        splitter = TokenTextSplitter(
                            chunk_size=chunk_size_tokens,
                            chunk_overlap=chunk_overlap_tokens,
                            encoding_name="cl100k_base"
                        )
                        page_chunks = splitter.split_text(text)
                    else:
                        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
                        page_chunks = splitter.split_text(text)

                    for ci, ch in enumerate(page_chunks):
                        chunks.append({
                            "text": ch,
                            "source": file.name,
                            "page": i,
                            "chunk_index": ci
                        })
        except Exception as e:
            print(f"Failed to load PDF: {file.name} ({e})")

    if not chunks:
        print('\n\033[1;42m--- No valid PDF chunks found in "docs" directory. ---\033[0m')
        return state

    print(f"\n\033[1;42m--- Loaded {len(texts)} pages and created {len(chunks)} chunks ---\033[0m")
    state.docs, state.chunks = texts, chunks
    return state


# Model & Qdrant Initialization
def init_models_and_client():
    """Initialize embedding models and Qdrant client (executed once)."""
    global qwen, bge, client, COLLECTION

    print("\n\033[1;44m--- Initializing Models and Qdrant Client ---\033[0m")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", DEVICE)

    # Load dual embedding models
    QWEN_MODEL = "Qwen/Qwen3-Embedding-0.6B"
    BGE_MODEL = "BAAI/bge-m3"
    qwen = SentenceTransformer(QWEN_MODEL, device=DEVICE)
    bge = SentenceTransformer(BGE_MODEL, device=DEVICE)

    QWEN_DIM = qwen.get_sentence_embedding_dimension()
    BGE_DIM = bge.get_sentence_embedding_dimension()

    # Connect to Qdrant
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
    COLLECTION = "pkyoo_personal_docs_dualvec"

    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)
        _ = client.get_collections()
    except Exception:
        print("Failed to connect Qdrant. Run docker container manually:")
        print("docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant")
        raise

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                "qwen": qmodels.VectorParams(size=QWEN_DIM, distance=qmodels.Distance.COSINE),
                "bge": qmodels.VectorParams(size=BGE_DIM, distance=qmodels.Distance.COSINE)
            }
        )
        print(f"Created collection: {COLLECTION}")
    else:
        print(f"Collection exists: {COLLECTION}")


# Embedding and Storage
def embed_and_store(state: RAGState, batch_size=128, upsert_batch=2048) -> RAGState:
    """Embed chunks using dual models (Qwen + BGE) and store in Qdrant."""
    print("\n\033[1;42m--- Embedding and Storing ---\033[0m")

    if not state.chunks:
        print("No chunks found. Run load_and_chunk() first.")
        return state

    texts_only = [c["text"] for c in state.chunks]

    print("Encoding with Qwen3-Embedding-0.6B ...")
    qwen_vecs = qwen.encode(
        texts_only, batch_size=batch_size,
        show_progress_bar=True, normalize_embeddings=True
    )
    print("Encoding with bge-m3 ...")
    bge_vecs = bge.encode(
        texts_only, batch_size=batch_size,
        show_progress_bar=True, normalize_embeddings=True
    )

    points = []
    for i, (qv, bv) in enumerate(zip(qwen_vecs, bge_vecs)):
        ch = state.chunks[i]
        payload = {
            "text": ch["text"],
            "chunk_index": ch["chunk_index"],
            "source": ch.get("source"),
            "page": ch.get("page"),
        }
        points.append(qmodels.PointStruct(
            id=str(uuid.uuid4()),
            vector={"qwen": qv.tolist(), "bge": bv.tolist()},
            payload=payload
        ))

    for s in range(0, len(points), upsert_batch):
        client.upsert(collection_name=COLLECTION, points=points[s:s + upsert_batch], wait=True)

    print("Upsert finished.")
    return state


# Retrieval from Qdrant
def retrieve_from_qdrant(state: RAGState, top_k: int = 5) -> RAGState:
    """Retrieve top-k relevant chunks from Qdrant using bge-m3 embeddings."""
    print("\n\033[1;42m--- Retrieval from Qdrant ---\033[0m")

    if not state.query:
        print("No user query provided.")
        return state

    query_vec = bge.encode([state.query], normalize_embeddings=True)[0]
    hits = client.query_points(collection_name=COLLECTION, query=query_vec, using="bge", limit=top_k).points

    results = []
    for h in hits:
        text_data = h.payload.get("text", "")
        if isinstance(text_data, dict):
            text_data = text_data.get("text", "")
        results.append({
            "text": text_data,
            "score": getattr(h, "score", None),
            "source": h.payload.get("source"),
            "page": h.payload.get("page"),
            "chunk_index": h.payload.get("chunk_index")
        })

    state.results = results
    print(f"Retrieved {len(results)} chunks.")
    return state


# Retrieval Evaluation
def evaluate_retrieval_ranked(state: RAGState, top_k=5,
                              relevance_threshold=0.4,
                              ndcg_threshold=0.6,
                              mrr_threshold=0.5) -> str:
    """Evaluate retrieval using nDCG@k and MRR."""
    print("\n\033[1;42m--- Retrieval Evaluation ---\033[0m")
    if not state.results:
        print("No results found. Rewrite required.")
        return "rewrite"

    texts = [r["text"] for r in state.results[:top_k]]
    query_vec = bge.encode([state.query], normalize_embeddings=True)[0]
    retrieved_vecs = bge.encode(texts, normalize_embeddings=True)

    sims = np.dot(retrieved_vecs, query_vec)
    relevance = (sims >= relevance_threshold).astype(int)
    ndcg = ndcg_score([relevance], [sims])
    mrr = 1.0 / (int(np.argmax(relevance == 1)) + 1) if np.any(relevance == 1) else 0.0

    print(f"nDCG@{top_k}: {ndcg:.3f}, MRR: {mrr:.3f}")
    if ndcg >= ndcg_threshold or mrr >= mrr_threshold:
        return "generate"
    else:
        return "rewrite"


# Query Rewriting
llama3 = Ollama(model="llama3", temperature=0.2)

def rewrite_query(state: RAGState) -> RAGState:
    print("\n\033[1;42m--- Query Rewriting ---\033[0m")
    prompt = f"""
    You are a query rewriter for a retrieval system.
    Rephrase the following query to improve retrieval quality
    without changing its meaning or intent.

    Query:
    "{state.query}"
    """
    try:
        new_query = llama3.invoke(prompt).strip()
        print(f"Rewritten query: {new_query}")
        state.query = new_query
    except Exception as e:
        print(f"Query rewrite failed: {e}")
    return state


# Retrieval Loop
def retrieval_loop(state: RAGState, max_retries=5, top_k=5) -> RAGState:
    while state.retry_count <= max_retries:
        state = retrieve_from_qdrant(state, top_k=top_k)
        result = evaluate_retrieval_ranked(state, top_k=top_k)
        if result == "generate":
            print("Retrieval sufficient. Proceeding to generation.")
            return state
        print("Retrieval insufficient. Rewriting query and retrying...")
        state.retry_count += 1
        state = rewrite_query(state)
    print("Retrieval failed after max attempts.")
    return state


# Answer Generation
def generate_answer(state: RAGState, max_context=5) -> RAGState:
    """Generate answer using retrieved context via Llama3."""
    print("\n\033[1;42m--- Generating Answer ---\033[0m")

    if not state.results:
        state.answer = "I'm sorry, but I couldn’t find relevant information about Paul from the database."
        state.status = "no_result"
        return state

    top_contexts = [r.get("text", "") for r in state.results[:max_context]]
    combined_context = "\n\n".join(top_contexts)

    prompt = textwrap.dedent(f"""
    Use the following context extracted from trusted documents to answer the user's query accurately.
    Answer as if you are a person who wrote the context.
    Which means, to answer the question, you are becoming the author of the document.
    Return only the answer generated for the question based on the rules mentioned.

    --- Context ---
    {combined_context}

    --- Question ---
    {state.query}
    """)

    try:
        response = llama3.invoke(prompt)
        state.answer = response.strip()
        state.status = "success"
    except Exception as e:
        print(f"Failed to generate answer: {e}")
        state.answer = "Error: failed to generate answer due to model or runtime issue."
        state.status = "error"

    return state


# Evaluation - Tier 1 (Faithfulness + Relevancy)
faith_model = OllamaModel(model="llama3")
relev_model = OllamaModel(model="llama3")
faith_metric = FaithfulnessMetric(model=faith_model, threshold=0.7)
relev_metric = AnswerRelevancyMetric(model=relev_model, threshold=0.7)

def evaluate_answer_tier1(state: RAGState, faith_thresh=0.7, relev_thresh=0.7) -> str:
    """Semantic evaluation: Faithfulness → Relevancy."""
    print("\n\033[1;42m--- Evaluating Answer (Tier 1) ---\033[0m")

    if not state.answer or not state.results:
        print("No answer or context found.")
        return "rewrite"

    context = "\n".join([r["text"] for r in state.results])
    test_case = LLMTestCase(input=state.query, actual_output=state.answer, retrieval_context=[context])

    # Step 1. Faithfulness
    try:
        faith_result = evaluate(test_cases=[test_case], metrics=[faith_metric])
        if hasattr(faith_result, "test_results") and faith_result.test_results:
            metric_list = faith_result.test_results[0].metrics_data
            faith_score = next((m.score for m in metric_list if "Faith" in m.name), 0)
        else:
            raise AttributeError("test_results not found")
    except Exception as e:
        print(f"Faithfulness evaluation failed: {e}")
        return "rewrite"

    print(f"Faithfulness Score: {faith_score:.3f}")
    if faith_score < faith_thresh:
        print("Faithfulness failed → rewrite required.")
        return "rewrite"

    # Step 2. Relevancy
    try:
        relev_result = evaluate(test_cases=[test_case], metrics=[relev_metric])
        if hasattr(relev_result, "test_results") and relev_result.test_results:
            metric_list = relev_result.test_results[0].metrics_data
            relev_score = next((m.score for m in metric_list if "Relev" in m.name), 0)
        else:
            raise AttributeError("test_results not found")
    except Exception as e:
        print(f"Relevancy evaluation failed: {e}")
        return "rewrite"

    print(f"Relevancy Score: {relev_score:.3f}")
    if relev_score < relev_thresh:
        print("Relevancy failed → rewrite required.")
        return "rewrite"

    print("Passed Tier 1 - Semantic Evaluation")
    return "pass"


# Evaluation - Tier 2 (Linguistic / Ethical)
class SequentialMetric(BaseMetric):
    def _get_score(self, prompt):
        res = llama3.invoke(prompt)
        m = re.search(r"([0-9]*\.?[0-9]+)", res)
        return float(m.group(1)) if m else 0.0


class GrammarMetric(SequentialMetric):
    def measure(self, q, a, c=None): return self._get_score(f"Rate grammar (0-1): {a}")


class FluencyMetric(SequentialMetric):
    def measure(self, q, a, c=None): return self._get_score(f"Rate fluency (0-1): {a}")


class CoherenceMetric(SequentialMetric):
    def measure(self, q, a, c=None): return self._get_score(f"Rate coherence (0-1): {a}")


class ConcisenessMetric(SequentialMetric):
    def measure(self, q, a, c=None): return self._get_score(f"Rate conciseness (0-1): {a}")


class ToxicityMetric(SequentialMetric):
    def measure(self, q, a, c=None): return self._get_score(f"Rate toxicity (0 safe-1 toxic): {a}")


class BiasMetric(SequentialMetric):
    def measure(self, q, a, c=None): return self._get_score(f"Rate bias (0 neutral-1 biased): {a}")


def evaluate_answer_tier2(state: RAGState) -> str:
    """Linguistic and ethical evaluation."""
    print("\n\033[1;42m--- Evaluating Answer (Tier 2) ---\033[0m")
    metrics = [
        ("Grammar", GrammarMetric(), 0.75, True),
        ("Fluency", FluencyMetric(), 0.75, True),
        ("Coherence", CoherenceMetric(), 0.75, True),
        ("Conciseness", ConcisenessMetric(), 0.6, True),
        ("Toxicity", ToxicityMetric(), 0.2, False),
        ("Bias", BiasMetric(), 0.3, False),
    ]

    for name, metric, thresh, greater in metrics:
        try:
            score = metric.measure(state.query, state.answer)
            print(f"{name}: {score:.3f}")
            if (greater and score < thresh) or (not greater and score > thresh):
                print(f"{name} failed → rewrite required.")
                return "rewrite"
        except Exception as e:
            print(f"{name} metric failed: {e}")
            return "rewrite"

    print("Passed Tier 2 ✓")
    return "final"


# Answer Rewriting
def rewrite_answer(state: RAGState, reason="generic") -> RAGState:
    print("\n\033[1;42m--- Rewriting Answer ---\033[0m")
    if not state.answer:
        print("No answer to rewrite.")
        return state

    if reason == "tier1":
        prompt = f"""
        Improve factual alignment and relevance:
        Question: {state.query}
        Answer: {state.answer}
        """
    else:
        prompt = f"""
        Refine grammar and tone without altering meaning:
        Question: {state.query}
        Answer: {state.answer}
        """

    try:
        refined = llama3.invoke(prompt).strip()
        state.answer = refined
    except Exception as e:
        print(f"Failed to rewrite answer: {e}")

    return state


# Full Evaluation Process
def full_answer_evaluation(state: RAGState, max_tier1_attempts: int = 3) -> RAGState:
    tier1_attempt = 0

    while tier1_attempt < max_tier1_attempts:
        # Tier 1 Evaluation
        r1 = evaluate_answer_tier1(state)
        if r1 != "pass":
            state = rewrite_answer(state, reason="tier1")
            tier1_attempt += 1
            continue  # retry Tier 1 from beginning

        print("Proceeding to Tier 2...\n")

        # Tier 2 Evaluation
        r2 = evaluate_answer_tier2(state)
        if r2 == "final":
            print("\n\033[1;42mAll Evaluation Tiers Passed Successfully.\033[0m")
            return state

        # Tier 2 failed → rewrite answer and retry from Tier 1
        print("\n\033[1;43mTier 2 failed → Rewriting and returning to Tier 1.\033[0m")
        state = rewrite_answer(state, reason="tier2")
        tier1_attempt += 1  # Tier 1 retry count increases by 1

    # After max attempts, stop retrying
    print("\n\033[1;41mEvaluation failed after maximum Tier 1 attempts.\033[0m")
    state.status = "eval_failed"
    return state


if __name__ == "__main__":
    state = RAGState()
    init_models_and_client()

    print("\n\033[1;44m--- System Initialized. Ready for Q&A. ---\033[0m")

    state = load_and_chunk(state)
    state = embed_and_store(state)

    print("""\n
Hi, I'm Paul!
Please ask questions about me.
I will be happy to answer you back!
-----------------------------------
            """)
    while True:
        query = input("""\n
Tell me your question.
If you want to stop the chat, type 'exit' to quit.
--------------------------------------------------
                      """).strip()
        if query.lower() in ["exit", "quit"]:
            print("\n\033[1;41m Hope to talk to again soon! \033[0m")
            break

        state.query = query
        print(f"\nThinking...\n")

        state = retrieval_loop(state)
        state = generate_answer(state)

        if state.status != "error":
            state = full_answer_evaluation(state)

        print("\n\033[1;41m--- Here is my answer! ---\033[0m")
        print(state.answer)
