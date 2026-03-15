import os
from dotenv import load_dotenv
from groq import Groq
from rich.console import Console
 
load_dotenv()
console = Console()
 
GROQ_MODEL       = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.0"))

def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not found. Copy .env.example → .env and add your key.\n"
            "Get a free key at: https://console.groq.com"
        )
    return Groq(api_key=api_key)

# ── Prompt templates ─────────────────────────────────────────────────────────
 
SYSTEM_PROMPT = """You are a knowledgeable statistics tutor with deep expertise \
in the textbook "An Introduction to Statistical Learning" (ISLR V2).
 
Your job is to answer questions using ONLY the context excerpts provided.
Rules:
- Base your answer strictly on the provided context
- If the context does not contain enough information, say so clearly
- Cite the page number when referencing specific content (e.g. "As stated on p.142...")
- Keep answers clear and educational — you are teaching a student
- Use LaTeX notation for math where appropriate (e.g. $\\hat{y} = \\beta_0 + \\beta_1 x$)
"""

def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Build the user message by injecting retrieved context.
 
    The context block contains the top-k most relevant ISLR excerpts.
    Each excerpt is labeled with its page number for citation.
    """
    context_block = "\n\n".join([
        f"[Page {c['metadata']['page']}]\n{c['text']}"
        for c in retrieved_chunks
    ])
 
    return f"""Use the following excerpts from ISLR V2 to answer the question.
 
─── CONTEXT ───────────────────────────────────────────────
{context_block}
───────────────────────────────────────────────────────────
 
Question: {question}
 
Answer:"""

def generate_answer(question: str,
                    retrieved_chunks: list[dict],
                    client: Groq = None) -> str:
    """
    Full RAG generation step:
      1. Build prompt with retrieved context
      2. Call Groq LLM
      3. Return the grounded answer
 
    temperature=0.0 → deterministic output (best for factual Q&A)
    """
    if client is None:
        client = get_groq_client()
 
    prompt = build_prompt(question, retrieved_chunks)
 
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=GROQ_TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    )
 
    return response.choices[0].message.content

def print_answer(question: str, answer: str, chunks: list[dict]) -> None:
    """Pretty-print the full RAG response."""
    console.print(f"\n[bold yellow]Q:[/] {question}")
    console.print(f"\n[bold green]A:[/] {answer}")
    console.print(f"\n[dim]─── Sources used ───[/]")
    for c in chunks:
        console.print(f"[dim]  p.{c['metadata']['page']} | {c['text'][:80]}...[/]")
 