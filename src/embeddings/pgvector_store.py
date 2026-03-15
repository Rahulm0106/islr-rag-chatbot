import os
import psycopg2
import psycopg2.extras
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from rich.console import Console
 
load_dotenv()
console = Console()
 
POSTGRES_URL    = os.getenv("POSTGRES_URL", "postgresql://localhost:5432/islr_rag")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
VECTOR_DIM      = 384  # all-MiniLM-L6-v2 produces 384-dimensional vectors
 
 
def get_connection():
    return psycopg2.connect(POSTGRES_URL)
 
 
def setup_pgvector(conn) -> None:
    """
    One-time setup: enable pgvector extension and create the chunks table.
 
    The `embedding vector(384)` column stores our 384-dim vectors.
    The HNSW index on that column enables fast ANN search.
    """
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
 
        cur.execute("""
            CREATE TABLE IF NOT EXISTS islr_chunks (
                id          SERIAL PRIMARY KEY,
                text        TEXT NOT NULL,
                page        INTEGER,
                chunk_index INTEGER,
                source      TEXT DEFAULT 'ISLR_V2',
                embedding   vector(384)  -- the magic column
            );
        """)
 
        # HNSW index: m=16 (connections per node), ef_construction=64 (build quality)
        # Higher values = better recall but slower build. These are production defaults.
        cur.execute("""
            CREATE INDEX IF NOT EXISTS islr_chunks_embedding_hnsw
            ON islr_chunks USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        """)
 
        conn.commit()
    console.print("[green]✓[/] pgvector setup complete (table + HNSW index)")
 
 
def embed_and_store_pg(chunks: list[dict], conn) -> None:
    """
    Embed all chunks and insert them into Postgres.
 
    SentenceTransformer.encode() returns numpy arrays which psycopg2 serializes
    into Postgres vector format automatically via the register_vector() call.
    """
    import pgvector.psycopg2
    pgvector.psycopg2.register_vector(conn)
 
    model = SentenceTransformer(EMBEDDING_MODEL)
    console.print(f"[dim]Embedding {len(chunks)} chunks with {EMBEDDING_MODEL}...[/]")
 
    texts = [c["text"] for c in chunks]
    # batch encode is much faster than one-by-one
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
 
    with conn.cursor() as cur:
        records = [
            (
                c["text"],
                c["metadata"]["page"],
                c["metadata"]["chunk_index"],
                c["metadata"].get("source", "ISLR_V2"),
                emb.tolist(),  # numpy → Python list → Postgres vector
            )
            for c, emb in zip(chunks, embeddings)
        ]
        psycopg2.extras.execute_values(
            cur,
            """INSERT INTO islr_chunks (text, page, chunk_index, source, embedding)
               VALUES %s""",
            records,
            template="(%s, %s, %s, %s, %s::vector)",
        )
        conn.commit()
 
    console.print(f"[green]✓[/] Stored {len(chunks)} chunks in pgvector")
 
 
def similarity_search_pg(query: str,
                          conn,
                          top_k: int = 5,
                          page_filter: int = None) -> list[dict]:
    """
    Vector similarity search in Postgres.
 
    The <=> operator computes cosine distance (lower = more similar).
 
    page_filter: optional — only search chunks from a specific page range.
    This shows off the power of pgvector: SQL WHERE + vector search in one query.
    """
    import pgvector.psycopg2
    pgvector.psycopg2.register_vector(conn)
 
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode([query])[0].tolist()
 
    # Build query — optionally add page filter (SQL + vector = pgvector superpower)
    where_clause = "WHERE page >= %s" if page_filter else ""
    params = (query_embedding, page_filter, top_k) if page_filter else (query_embedding, top_k)
 
    sql = f"""
        SELECT text, page, chunk_index, embedding <=> %s::vector AS distance
        FROM islr_chunks
        {where_clause}
        ORDER BY distance ASC
        LIMIT %s;
    """
 
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
 
    return [
        {
            "text": row["text"],
            "metadata": {"page": row["page"], "chunk_index": row["chunk_index"]},
            "distance": round(float(row["distance"]), 4),
        }
        for row in rows
    ]