import fitz  # PyMuPDF
import re
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rich.console import Console
from rich.progress import track

console = Console()

def _clean_text(text: str) -> str:
    """
    Light cleaning — remove artifacts common in textbook PDFs.
    """
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'^\d+\n', '', text, flags=re.MULTILINE)
    text = text.strip()
    return text

def extract_pages(pdf_path: str) -> list[dict]:
    """
    Extract raw text from every page of the PDF.
    Returns a list of dicts: {page_num, text}
 
    Why page-by-page?
      We preserve page numbers as metadata. Later, when the RAG system answers
      a question, we can cite "ISLR p.142" — useful for studying!
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found at: {pdf_path}\n"
                                f"Place ISLR_V2.pdf inside the data/ folder.")
    doc = fitz.open(str(path))
    pages = []
    
    console.print(f"[bold green]Opening:[/] {path.name} ({len(doc)} pages)")

    for page_num, page in enumerate(doc):
        text = page.get_text("text")  # "text" = reading order, not raw stream
        cleaned = _clean_text(text)
        if cleaned:  # skip blank/image-only pages
            pages.append({
                "page_num": page_num + 1,
                "text": cleaned
            })
 
    console.print(f"[green]✓[/] Extracted text from {len(pages)} pages")
    return pages

def chunk_pages(pages: list[dict],
                chunk_size: int = 500,
                chunk_overlap: int = 50) -> list[dict]:
    """
    Split each page's text into smaller chunks, with some overlap for embedding.

    RecursiveCharacterTextSplitter tries to split on paragraph breaks first,
    then newlines, then sentences, then words — in that order. This keeps
    semantically related text together as long as possible.
 
    Returns list of dicts: {text, metadata: {page, chunk_index, source}}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,   # character-based (swap for tiktoken if needed)
    )
    all_chunks = []
    for page in track(pages, description="Chunking pages..."):
        chunks = splitter.split_text(page["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "metadata": {
                    "page": page["page_num"],
                    "chunk_index": i,
                    "source": "ISLR_V2",
                }
            })
 
    console.print(f"[green]✓[/] Created {len(all_chunks)} chunks "
                  f"(avg {sum(len(c['text']) for c in all_chunks)//len(all_chunks)} chars each)")
    return all_chunks

def inspect_chunks(chunks: list[dict], n: int = 3) -> None:
    """Print a few sample chunks so you can sanity-check quality."""
    console.print(f"\n[bold]── Sample chunks (showing {n}) ──[/]")
    step = max(1, len(chunks) // n)
    for i in range(0, min(n * step, len(chunks)), step):
        c = chunks[i]
        console.print(f"\n[bold cyan]Chunk #{i}[/] | page {c['metadata']['page']}")
        console.print(f"[dim]{c['text'][:300]}{'...' if len(c['text']) > 300 else ''}[/]")
        console.print(f"[dim]Length: {len(c['text'])} chars[/]")