import pandas as pd
from rich.console import Console
from rich.table import Table
 
console = Console()
 
 
def build_eval_dataset(qa_pairs: list[dict]) -> list[dict]:
    """
    qa_pairs: list of {question, answer, contexts, ground_truth (optional)}
    Returns RAGAS-formatted dataset.
 
    ground_truth: what the correct answer should be (for context recall).
    If you don't have ground truth, skip context_recall and only compute
    faithfulness + answer_relevancy.
    """
    dataset = []
    for qa in qa_pairs:
        dataset.append({
            "question":     qa["question"],
            "answer":       qa["answer"],
            "contexts":     qa["contexts"],   # list of retrieved chunk texts
            "ground_truth": qa.get("ground_truth", ""),
        })
    return dataset

def run_ragas_eval(dataset: list[dict]) -> pd.DataFrame:
    """
    Run RAGAS metrics on your QA dataset.
    Returns a DataFrame with scores per question + aggregate averages.
 
    Note: RAGAS makes its own LLM calls to evaluate. It defaults to OpenAI
    but can be configured to use any provider. For learning, even running
    it on 5-10 questions gives useful signal.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_recall
        from datasets import Dataset
 
        hf_dataset = Dataset.from_list(dataset)
 
        metrics = [faithfulness, answer_relevancy]
        if any(d["ground_truth"] for d in dataset):
            metrics.append(context_recall)
 
        result = evaluate(hf_dataset, metrics=metrics)
        df = result.to_pandas()
        return df
 
    except ImportError:
        console.print("[yellow]RAGAS not installed. Run: pip install ragas datasets[/]")
        return pd.DataFrame()

def manual_eval(qa_pairs: list[dict]) -> pd.DataFrame:
    """
    Lightweight manual evaluation — no extra LLM calls needed.
    Scores each answer on simple heuristics you can inspect.
 
    Use this if you want to avoid RAGAS's LLM calls while learning.
    Good enough to compare naive RAG vs advanced RAG side-by-side.
    """
    rows = []
    for qa in qa_pairs:
        answer   = qa["answer"].lower()
        contexts = " ".join(qa["contexts"]).lower()
        question = qa["question"].lower()
 
        # Heuristic faithfulness: key words in answer also appear in context
        answer_words  = set(answer.split())
        context_words = set(contexts.split())
        common = answer_words & context_words
        faithfulness_score = round(len(common) / max(len(answer_words), 1), 3)
 
        # Heuristic relevancy: key question words appear in answer
        q_words = set(question.replace("?", "").split())
        overlap = q_words & answer_words
        relevancy_score = round(len(overlap) / max(len(q_words), 1), 3)
 
        rows.append({
            "question":          qa["question"],
            "answer_preview":    qa["answer"][:100] + "...",
            "faithfulness":      faithfulness_score,
            "answer_relevancy":  relevancy_score,
            "sources":           [c["metadata"]["page"] for c in qa.get("chunk_objects", [])],
        })
 
    return pd.DataFrame(rows)
 
 
def print_eval_table(df: pd.DataFrame) -> None:
    """Pretty-print eval results as a Rich table."""
    table = Table(title="RAG Evaluation Results", show_lines=True)
    table.add_column("Question", style="cyan", max_width=40)
    table.add_column("Faithfulness", justify="center")
    table.add_column("Relevancy", justify="center")
 
    for _, row in df.iterrows():
        f_score = row.get("faithfulness", "-")
        r_score = row.get("answer_relevancy", row.get("relevancy", "-"))
 
        f_color = "green" if isinstance(f_score, float) and f_score > 0.7 else "yellow"
        r_color = "green" if isinstance(r_score, float) and r_score > 0.7 else "yellow"
 
        table.add_row(
            str(row["question"])[:40],
            f"[{f_color}]{f_score}[/]",
            f"[{r_color}]{r_score}[/]",
        )
 
    console.print(table)
    avg_f = df["faithfulness"].mean() if "faithfulness" in df else "-"
    avg_r = df.get("answer_relevancy", df.get("relevancy", pd.Series())).mean()
    console.print(f"\n[bold]Avg Faithfulness:[/] {round(avg_f, 3)}   "
                  f"[bold]Avg Relevancy:[/] {round(avg_r, 3)}")