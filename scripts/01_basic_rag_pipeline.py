import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
import argparse
from rich.console import Console
from rich.rule import Rule
from dotenv import load_dotenv

from src.ingestion.pdf_loader import extract_pages, chunk_pages, inspect_chunks
from src.embeddings.vector_store import (
    get_or_create_collection, embed_and_store, similarity_search, print_search_results
)
from src.generation.llm import get_groq_client, generate_answer, print_answer
 
load_dotenv()
console = Console()
 
PDF_PATH   = "data/ISLRv2.pdf"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
TOP_K      = int(os.getenv("TOP_K", 5))

# ── Test questions about ISLR ─────────────────────────────────────────────────
TEST_QUESTIONS = [
    "What is the bias-variance tradeoff?",
    "How does k-fold cross-validation work?",
    "What is the difference between LDA and QDA?",
    "Explain the lasso penalty and how it differs from ridge regression.",
    "What are natural splines and why are they used?",
]

def main(reset: bool = False):
    console.print(Rule("[bold green]ISLR RAG — Basic Pipeline[/]"))
 
    # ── Step 1: Ingest PDF ────────────────────────────────────────────────────
    console.print(Rule("Step 1: PDF Ingestion", style="cyan"))
 
    collection = get_or_create_collection(reset=reset)
 
    # Only re-embed if collection is empty (or if --reset was passed)
    if collection.count() == 0:
        console.print("[yellow]Collection empty → ingesting PDF...[/]")
        pages  = extract_pages(PDF_PATH)
        chunks = chunk_pages(pages, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        inspect_chunks(chunks)              # <-- look at sample chunks
        embed_and_store(chunks, collection) # <-- this takes 2–5 min first time
    else:
        console.print(f"[green]Collection already has {collection.count()} chunks — skipping ingestion[/]")
        console.print("[dim](Run with --reset to re-ingest)[/]")

    # ── Step 2: Test similarity search ────────────────────────────────────────
    console.print(Rule("Step 2: Similarity Search Test", style="cyan"))
 
    test_query = "what is overfitting and how do we detect it?"
    results = similarity_search(test_query, collection, top_k=TOP_K)
    print_search_results(test_query, results)
 
    # ── Step 3: Full RAG Q&A ──────────────────────────────────────────────────
    console.print(Rule("Step 3: Full RAG Answers", style="cyan"))
 
    groq = get_groq_client()
 
    for question in TEST_QUESTIONS:
        chunks   = similarity_search(question, collection, top_k=TOP_K)
        answer   = generate_answer(question, chunks, groq)
        print_answer(question, answer, chunks)
        console.print()
 
    console.print(Rule("[bold green]Saturday Complete! 🎉[/]"))
    console.print("\n[bold]What to reflect on:[/]")
    console.print("  1. Look at the retrieved chunks — are they actually relevant?")
    console.print("  2. Look at the answers — are they grounded or adding hallucinated info?")
    console.print("  3. Try changing TOP_K in .env (try 2 vs 10) and re-run")
    console.print("  4. Tomorrow we fix the retrieval weaknesses with advanced techniques\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true",
                        help="Delete and re-ingest the ChromaDB collection")
    args = parser.parse_args()
    main(reset=args.reset)