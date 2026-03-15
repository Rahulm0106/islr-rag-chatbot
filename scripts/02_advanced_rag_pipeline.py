import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from rich.console import Console
from rich.rule import Rule
from dotenv import load_dotenv

from src.embeddings.vector_store import (
    get_or_create_collection, similarity_search
)
from src.retrieval.advanced import (
    BM25Index, hybrid_search, hyde_query, multi_query_retrieve
)
from src.generation.llm import get_groq_client, generate_answer, print_answer
from src.evaluation.ragas_eval import manual_eval, print_eval_table

load_dotenv()
console = Console()

TOP_K = int(os.getenv("TOP_K", 5))

EVAL_QUESTIONS = [
    {
        "question": "What is the bias-variance tradeoff?",
        "ground_truth": "As model flexibility increases, variance increases and bias decreases. The optimal model minimizes the sum of both."
    },
    {
        "question": "How does LOOCV differ from k-fold cross-validation?",
        "ground_truth": "LOOCV uses n-1 training observations per fold giving n folds total. k-fold uses k folds. LOOCV has lower bias but higher variance and computational cost."
    },
    {
        "question": "What penalty does the lasso add and what is its effect on coefficients?",
        "ground_truth": "Lasso adds an L1 penalty (sum of absolute values). Unlike ridge, it can shrink coefficients to exactly zero, performing variable selection."
    },
    {
        "question": "When should you use QDA instead of LDA?",
        "ground_truth": "QDA is preferred when the covariance matrices differ across classes or when there are many training samples. LDA is better when training data is limited."
    },
    {
        "question": "What are the advantages of using splines over polynomial regression?",
        "ground_truth": "Splines fit local regions separately so they don't suffer from the global instability of high-degree polynomials. They provide more flexibility with fewer parameters."
    },
]


def run_naive(collection, groq, questions):
    """Baseline: basic similarity search → generate."""
    results = []
    for qa in questions:
        chunks = similarity_search(qa["question"], collection, top_k=TOP_K)
        answer = generate_answer(qa["question"], chunks, groq)
        results.append({
            "question": qa["question"],
            "answer": answer,
            "contexts": [c["text"] for c in chunks],
            "chunk_objects": chunks,
            "ground_truth": qa.get("ground_truth", ""),
        })
    return results


def run_advanced(collection, groq, questions, bm25_index):
    """Advanced: HyDE + hybrid search + multi-query → generate."""
    results = []
    for qa in questions:
        q = qa["question"]

        # HyDE: generate hypothetical answer, use it as query
        hyde_q = hyde_query(q, groq)

        # Hybrid search on HyDE-augmented query
        chunks = hybrid_search(hyde_q, collection, bm25_index, top_k=TOP_K)

        # Multi-query as fallback to add coverage
        extra = multi_query_retrieve(q, collection, groq, top_k=3)
        seen = {c["text"] for c in chunks}
        for c in extra:
            if c["text"] not in seen:
                chunks.append(c)
                seen.add(c["text"])

        chunks = chunks[:TOP_K + 2]  # cap for generation context
        answer = generate_answer(q, chunks, groq)

        results.append({
            "question": q,
            "answer": answer,
            "contexts": [c["text"] for c in chunks],
            "chunk_objects": chunks,
            "ground_truth": qa.get("ground_truth", ""),
        })
    return results


def mode_compare(collection, groq):
    """Run both naive and advanced, show eval scores side by side."""
    console.print(Rule("Building BM25 index for hybrid search", style="cyan"))

    # To build BM25 we need all chunks — get them from ChromaDB
    all_docs = collection.get(include=["documents", "metadatas"])
    all_chunks = [
        {"text": doc, "metadata": meta}
        for doc, meta in zip(all_docs["documents"], all_docs["metadatas"])
    ]
    bm25 = BM25Index(all_chunks)

    console.print(Rule("Running naive RAG", style="yellow"))
    naive_results = run_naive(collection, groq, EVAL_QUESTIONS)

    console.print(Rule("Running advanced RAG", style="green"))
    adv_results   = run_advanced(collection, groq, EVAL_QUESTIONS, bm25)

    import pandas as pd
    console.print(Rule("Naive RAG Evaluation", style="yellow"))
    naive_df = manual_eval(naive_results)
    print_eval_table(naive_df)

    console.print(Rule("Advanced RAG Evaluation", style="green"))
    adv_df = manual_eval(adv_results)
    print_eval_table(adv_df)

    # Side-by-side delta
    console.print(Rule("Improvement Delta", style="bold"))
    delta_f = adv_df["faithfulness"].mean() - naive_df["faithfulness"].mean()
    delta_r = adv_df["answer_relevancy"].mean() - naive_df["answer_relevancy"].mean()
    color_f = "green" if delta_f >= 0 else "red"
    color_r = "green" if delta_r >= 0 else "red"
    console.print(f"  Faithfulness delta:    [{color_f}]{delta_f:+.3f}[/]")
    console.print(f"  Relevancy delta:       [{color_r}]{delta_r:+.3f}[/]")


def mode_pgvector(groq):
    """Migrate to pgvector and run similarity search."""
    from src.embeddings.pgvector_store import (
        get_connection, setup_pgvector, embed_and_store_pg, similarity_search_pg
    )
    from src.ingestion.pdf_loader import extract_pages, chunk_pages

    console.print(Rule("Connecting to pgvector", style="purple"))
    conn = get_connection()
    setup_pgvector(conn)

    # Check if already populated
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM islr_chunks;")
        count = cur.fetchone()[0]

    if count == 0:
        console.print("[yellow]pgvector empty → ingesting...[/]")
        pages  = extract_pages("data/ISLRv2.pdf")
        chunks = chunk_pages(pages)
        embed_and_store_pg(chunks, conn)
    else:
        console.print(f"[green]pgvector has {count} chunks — skipping ingestion[/]")

    console.print(Rule("pgvector similarity search", style="purple"))
    for qa in EVAL_QUESTIONS[:3]:
        results = similarity_search_pg(qa["question"], conn, top_k=TOP_K)
        answer  = generate_answer(qa["question"], results, groq)
        print_answer(qa["question"], answer, results)

    conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["chroma", "pgvector", "compare"],
                        default="compare")
    args = parser.parse_args()

    console.print(Rule(f"[bold green]ISLR RAG — Advanced ({args.mode})[/]"))

    groq = get_groq_client()

    if args.mode == "pgvector":
        mode_pgvector(groq)
    else:
        collection = get_or_create_collection(reset=False)
        if collection.count() == 0:
            console.print("[red]Run saturday_basic_rag.py first to populate ChromaDB.[/]")
            sys.exit(1)

        if args.mode == "compare":
            mode_compare(collection, groq)
        else:
            # chroma mode — just run advanced
            all_docs = collection.get(include=["documents", "metadatas"])
            all_chunks = [
                {"text": doc, "metadata": meta}
                for doc, meta in zip(all_docs["documents"], all_docs["metadatas"])
            ]
            bm25 = BM25Index(all_chunks)
            results = run_advanced(collection, groq, EVAL_QUESTIONS, bm25)
            for r in results:
                print_answer(r["question"], r["answer"], r["chunk_objects"])

    console.print(Rule("[bold green]Advanced RAG Complete! You've built a production-grade RAG 🚀[/]"))


if __name__ == "__main__":
    main()