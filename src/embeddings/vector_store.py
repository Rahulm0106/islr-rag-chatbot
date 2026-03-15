import os
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from rich.console import Console
from rich.progress import track
 
load_dotenv()
console = Console()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
COLLECTION_NAME = "islr_rag"

def get_embedding_function():
    """
    Returns a ChromaDB-compatible embedding function.
    SentenceTransformer downloads the model on first run (~90MB, one-time).
    """
    console.print(f"[dim]Loading embedding model: {EMBEDDING_MODEL}[/]")
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

def get_or_create_collection(reset: bool = False) -> chromadb.Collection:
    """
    Connect to (or create) a persistent ChromaDB collection.
 
    reset=True  → delete existing data and start fresh (use when re-ingesting)
    reset=False → load existing collection (use when querying)
    """
    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
 
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn = get_embedding_function()

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            console.print("[yellow]⚠ Existing collection deleted[/]")
        except Exception:
            pass  # didn't exist yet, that's fine
 
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},  # cosine similarity for search
    )
    console.print(f"[green]✓[/] Collection '{COLLECTION_NAME}' ready "
                  f"({collection.count()} docs stored)")
    return collection
 
def embed_and_store(chunks: list[dict],
                    collection: chromadb.Collection,
                    batch_size: int = 100) -> None:
    """
    Embed all chunks and store them in ChromaDB.
 
    Why batching?
      sentence-transformers is most efficient when processing multiple texts
      at once (GPU/CPU matrix ops). batch_size=100 is a safe default.
 
    ChromaDB needs three parallel lists:
      ids        → unique string ID per chunk (required)
      documents  → the raw text (what gets returned at query time)
      metadatas  → dicts with page, source, etc. (filterable later)
    The embeddings are computed automatically by the embedding_function.
    """
    total = len(chunks)
    console.print(f"\n[bold]Embedding {total} chunks → ChromaDB...[/]")

    for start in track(range(0, total, batch_size), description="Embedding batches..."):
        batch = chunks[start: start + batch_size]
 
        ids       = [f"chunk_{start + i}" for i in range(len(batch))]
        documents = [c["text"] for c in batch]
        metadatas = [c["metadata"] for c in batch]
 
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            # embeddings computed automatically by the embedding_function
        )
    
    console.print(f"[green]✓[/] Stored {collection.count()} chunks in ChromaDB")

def similarity_search(query: str,
                      collection: chromadb.Collection,
                      top_k: int = 5) -> list[dict]:
    """
    Given a natural language query, find the top_k most relevant chunks.
 
    What happens under the hood:
      1. Your query is embedded into a vector using the same model
      2. ChromaDB computes cosine similarity between query vector and all
         stored chunk vectors
      3. The top_k closest chunks are returned
 
    Returns list of dicts: {text, metadata, distance}
    """
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
 
    # Unpack ChromaDB's nested list format
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "metadata": meta,
            "distance": round(dist, 4),  # lower = more similar (cosine)
        })
 
    return chunks
 
def print_search_results(query: str, results: list[dict]) -> None:
    """Pretty-print retrieval results for inspection."""
    console.print(f"\n[bold]Query:[/] {query}")
    console.print(f"[bold]Top {len(results)} results:[/]\n")
    for i, r in enumerate(results):
        console.print(f"[bold cyan]#{i+1}[/] | page {r['metadata']['page']} "
                      f"| similarity distance: {r['distance']}")
        console.print(f"[dim]{r['text'][:250]}...[/]\n")