import os
from dotenv import load_dotenv
from groq import Groq
from rank_bm25 import BM25Okapi
import chromadb
from rich.console import Console
 
load_dotenv()
console = Console()


# ── 1. HyDE ──────────────────────────────────────────────────────────────────
 
HYDE_PROMPT = """You are a statistics textbook author. Write a short paragraph \
(3-5 sentences) that directly answers the following question, in the style of \
an academic statistics textbook like ISLR. Use technical language.
 
Question: {question}
 
Textbook-style answer:"""
 
 
def hyde_query(question: str, groq_client: Groq) -> str:
    """
    Generate a hypothetical ISLR-style answer to use as the retrieval query.
 
    Why this works:
      The LLM generates text that *sounds like* the textbook, so its embedding
      will land closer to actual textbook chunks in vector space.
    """
    response = groq_client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        temperature=0.3,  # slight creativity to generate diverse phrasing
        messages=[{"role": "user", "content": HYDE_PROMPT.format(question=question)}],
    )
    hypothetical_answer = response.choices[0].message.content
    console.print(f"[dim]HyDE generated: {hypothetical_answer[:120]}...[/]")
    return hypothetical_answer

# ── 2. BM25 Keyword Search ────────────────────────────────────────────────────
 
class BM25Index:
    """
    Lightweight BM25 index over all chunks.
 
    BM25 scores a document based on how often query terms appear in it,
    adjusted for document length. It's the backbone of search engines like
    Elasticsearch. Great at catching exact technical terms.
    """
 
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        tokenized = [c["text"].lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        console.print(f"[green]✓[/] BM25 index built over {len(chunks)} chunks")
 
    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Return top_k chunks by BM25 score."""
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
 
        # Pair scores with chunks, sort descending
        scored = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]
 
        return [
            {**self.chunks[i], "bm25_score": round(score, 4)}
            for i, score in scored
            if score > 0  # ignore zero-score chunks
        ]
 
 
# ── 3. Hybrid Search (BM25 + Semantic fusion) ────────────────────────────────
 
def hybrid_search(query: str,
                  collection: chromadb.Collection,
                  bm25_index: BM25Index,
                  top_k: int = 5,
                  alpha: float = 0.5) -> list[dict]:
    """
    Fuse BM25 and semantic search results using Reciprocal Rank Fusion (RRF).
 
    alpha controls the blend:
      alpha=1.0 → pure semantic
      alpha=0.0 → pure BM25
      alpha=0.5 → equal blend (default)
 
    RRF formula: score(d) = Σ 1/(k + rank(d))  where k=60
    RRF is rank-based so it handles the different score scales of BM25 and
    cosine similarity naturally — no normalization needed.
    """
    # Semantic retrieval (2x top_k as candidates)
    semantic_results = collection.query(
        query_texts=[query],
        n_results=top_k * 2,
        include=["documents", "metadatas", "distances"],
    )
    semantic_chunks = [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(
            semantic_results["documents"][0],
            semantic_results["metadatas"][0],
            semantic_results["distances"][0],
        )
    ]
 
    # BM25 retrieval
    bm25_chunks = bm25_index.search(query, top_k=top_k * 2)
 
    # RRF fusion
    k = 60
    scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}
 
    for rank, chunk in enumerate(semantic_chunks):
        key = chunk["text"][:80]  # use text prefix as dedup key
        scores[key] = scores.get(key, 0) + alpha * (1 / (k + rank))
        chunk_map[key] = chunk
 
    for rank, chunk in enumerate(bm25_chunks):
        key = chunk["text"][:80]
        scores[key] = scores.get(key, 0) + (1 - alpha) * (1 / (k + rank))
        chunk_map[key] = chunk
 
    # Sort by fused score, return top_k
    sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)[:top_k]
    return [chunk_map[k] for k in sorted_keys]

# ── 4. Multi-query Retrieval ──────────────────────────────────────────────────
 
MULTI_QUERY_PROMPT = """Generate {n} different phrasings of this question to \
improve document retrieval. Return only the questions, one per line, no numbers.
 
Original question: {question}"""
 
 
def multi_query_retrieve(question: str,
                         collection: chromadb.Collection,
                         groq_client: Groq,
                         top_k: int = 5,
                         n_queries: int = 3) -> list[dict]:
    """
    Generate multiple query variants and union their results.
 
    Why: A single query might miss relevant chunks. Three paraphrased queries
    cast a wider net and deduplicate the results.
    """
    response = groq_client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        temperature=0.5,
        messages=[{
            "role": "user",
            "content": MULTI_QUERY_PROMPT.format(n=n_queries, question=question)
        }],
    )
    queries = [question] + response.choices[0].message.content.strip().split("\n")
    queries = [q.strip() for q in queries if q.strip()][:n_queries + 1]
 
    console.print(f"[dim]Multi-query variants: {queries}[/]")
 
    # Retrieve for each query, deduplicate by text
    seen_texts: set[str] = set()
    all_chunks: list[dict] = []
 
    for q in queries:
        results = collection.query(
            query_texts=[q],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            if doc not in seen_texts:
                seen_texts.add(doc)
                all_chunks.append({"text": doc, "metadata": meta, "distance": dist})
 
    return all_chunks[:top_k * 2]  # cap at 2x top_k before generation