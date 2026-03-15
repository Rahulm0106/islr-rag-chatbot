# ISLR RAG Chatbot

A Retrieval-Augmented Generation (RAG) system built on the "Introduction to Statistical Learning" (ISLR) textbook. This project demonstrates basic and advanced RAG techniques using local embeddings, vector stores, and LLMs.

## Features

- **Basic RAG**: Simple similarity search with ChromaDB and Groq LLM.
- **Advanced RAG**: HyDE (Hypothetical Document Embeddings), hybrid search (BM25 + vector), multi-query retrieval, and evaluation with RAGAS.
- **Vector Stores**: Support for ChromaDB (local) and pgvector (PostgreSQL).
- **Evaluation**: Automated metrics for faithfulness and answer relevancy.

## Project Structure

- `scripts/`: Executable pipelines
  - `01_basic_rag_pipeline.py`: Basic RAG implementation
  - `02_advanced_rag_pipeline.py`: Advanced RAG with multiple techniques
- `src/`: Source code modules
  - `embeddings/`: Vector store implementations (ChromaDB, pgvector)
  - `ingestion/`: PDF loading and chunking
  - `retrieval/`: Advanced retrieval methods (BM25, hybrid, HyDE, multi-query)
  - `generation/`: LLM integration (Groq)
  - `evaluation/`: RAGAS-based evaluation
- `requirements.txt`: Python dependencies
- `data/`: (Ignored) PDF and vector store data

## Setup

1. Clone the repo and navigate to the directory.

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables. Create a `.env` file and add your API keys:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   TOP_K=5  # Optional, default 5
   ```

4. Download the ISLR PDF (or any PDF) to `data/ISLRv2.pdf` for ingestion.

## Usage

### Basic RAG Pipeline

Run the basic RAG pipeline:

```bash
python scripts/01_basic_rag_pipeline.py
```

With reset (to re-ingest data):

```bash
python scripts/01_basic_rag_pipeline.py --reset
```

Help:

```
/Users/rahulmandviya/Documents/agent_projects/islr-rag-chatbot/.venv/lib/python3
.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020                            
warnings.warn(
usage: 01_basic_rag_pipeline.py [-h] [--reset]

optional arguments:
  -h, --help  show this help message and exit
  --reset     Delete and re-ingest the ChromaDB collection
```

### Advanced RAG Pipeline

Run the advanced RAG pipeline in compare mode (default):

```bash
python scripts/02_advanced_rag_pipeline.py
```

Other modes:

- ChromaDB mode: `python scripts/02_advanced_rag_pipeline.py --mode chroma`
- pgvector mode: `python scripts/02_advanced_rag_pipeline.py --mode pgvector`

Help:

```
usage: 02_advanced_rag_pipeline.py [-h] [--mode {chroma,pgvector,compare}]

optional arguments:
  -h, --help            show this help message and exit
  --mode {chroma,pgvector,compare}
```

### Sample Output

Running the scripts requires API keys and data setup. Example output from the compare mode (truncated):

```
──────────────────────────────── ISLR RAG — Advanced (compare) ────────────────────────────────
──────────────────────────── Building BM25 index for hybrid search ────────────────────────────
──────────────────────────── Running naive RAG ────────────────────────────
[Evaluation table would appear here]
──────────────────────────── Running advanced RAG ────────────────────────────
[Evaluation table would appear here]
──────────────────────────── Improvement Delta ────────────────────────────
  Faithfulness delta:    [+0.123]
  Relevancy delta:       [+0.456]
──────────────────────────────── Advanced RAG Complete! You've built a production-grade RAG 🚀 ────────────────────────────────
```

## Notes

- The `data/` folder is gitignored to avoid pushing large PDFs or databases.
- For pgvector, ensure PostgreSQL is running with pgvector extension.
- Evaluation uses predefined questions; modify `EVAL_QUESTIONS` in the scripts for custom testing.