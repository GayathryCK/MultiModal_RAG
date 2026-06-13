import os
import json
import fitz
import pdfplumber
import spacy
import nltk
from nltk.corpus import stopwords
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm
from loguru import logger
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.suppress_warnings import *

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PDF_DIR, IMAGE_DIR, EMBEDDINGS_DIR, VECTOR_DB_DIR,
    TEXT_EMBED_MODEL, CHROMA_COLLECTION
)

# load models at module level
_nlp       = None
_embedder  = None
_stopwords = None

def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
    return _nlp

def get_embedder():
    global _embedder
    if _embedder is None:
        logger.info(f"Loading text embedder: {TEXT_EMBED_MODEL}")
        _embedder = SentenceTransformer(TEXT_EMBED_MODEL)
    return _embedder

def get_stopwords():
    global _stopwords
    if _stopwords is None:
        try:
            _stopwords = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords")
            _stopwords = set(stopwords.words("english"))
    return _stopwords

def extract_text_pymupdf(pdf_path: Path) -> str:
    text = ""
    with fitz.open(str(pdf_path)) as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_pdfplumber(pdf_path: Path) -> str:
    text = ""
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def clean_text(raw: str) -> str:
    text = raw.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    text = text.encode("ascii", "ignore").decode()
    return text.strip()

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return [c for c in chunks if len(c.strip()) > 50]

def extract_topics(text: str) -> list[str]:
    nlp = get_nlp()
    if nlp is None:
        return []

    stop = get_stopwords()
    doc  = nlp(text[:20_000])   # spaCy token limit safeguard
    topics = set()

    for ent in doc.ents:
        if ent.label_ in {"ORG", "PERSON", "GPE", "PRODUCT",
                          "EVENT", "WORK_OF_ART", "LAW"}:
            topics.add(ent.text.strip().lower())

    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip().lower()
        words  = [w for w in phrase.split() if w not in stop and len(w) > 2]
        if len(words) >= 1 and len(phrase) > 3:
            topics.add(phrase)

    return list(topics)[:50]

def extract_images_from_pdf(pdf_path: Path, doc_name: str) -> list[str]:
    saved = []
    with fitz.open(str(pdf_path)) as doc:
        for page_num in range(len(doc)):
            images = doc[page_num].get_images(full=True)
            for img_idx, img_info in enumerate(images):
                xref       = img_info[0]
                base_image = doc.extract_image(xref)
                img_bytes  = base_image["image"]
                if len(img_bytes) < 5000:    # skip tiny icons/bullets
                    continue
                ext   = base_image["ext"]
                fname = f"{doc_name}_page{page_num+1}_img{img_idx+1}.{ext}"
                fpath = IMAGE_DIR / fname
                with open(fpath, "wb") as f:
                    f.write(img_bytes)
                saved.append(str(fpath))
    return saved

def store_in_chromadb(doc_name: str, pdf_path: str,
                       chunks: list[str], embeddings: list,
                       topics_per_chunk: list):
    client     = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    collection = client.get_or_create_collection(
                     name=CHROMA_COLLECTION,
                     metadata={"hnsw:space": "cosine"}
                 )

    ids, docs, metas, embs = [], [], [], []
    for i, (chunk, emb, topics) in enumerate(
            zip(chunks, embeddings, topics_per_chunk)):
        ids.append(f"pdf_{doc_name}_chunk{i}")
        docs.append(chunk)
        embs.append(emb)
        metas.append({
            "source"     : str(pdf_path),
            "source_type": "pdf",
            "doc_name"   : doc_name,
            "chunk_idx"  : i,
            "topics"     : json.dumps(topics),
        })

    for b in range(0, len(ids), 100):
        collection.upsert(
            ids        = ids[b:b+100],
            embeddings = embs[b:b+100],
            documents  = docs[b:b+100],
            metadatas  = metas[b:b+100],
        )
    logger.success(f"Stored {len(ids)} chunks in ChromaDB for [{doc_name}]")

# MAIN PIPELINE

def process_pdf(pdf_path: Path) -> dict:
    pdf_path = Path(pdf_path)
    doc_name = pdf_path.stem
    logger.info(f"Processing PDF: {pdf_path.name}")
    cache_path = EMBEDDINGS_DIR / "text" / f"{doc_name}_pdf.json"
    if cache_path.exists():
        logger.info(f"  Skipping {pdf_path.name} — already processed")
        with open(cache_path) as f:
            data = json.load(f)
        return {
            "node_id": f"pdf_{doc_name}",
            "node_type": "PDF",
            "name": pdf_path.name,
            "path": str(pdf_path),
            "text": "",
            "topics": data.get("topics", []),
            "embedding": data.get("embedding", []),
            "num_chunks": data.get("num_chunks", 0),
            "images_saved": 0,
        }

    text = extract_text_pymupdf(pdf_path)
    if not text.strip():
        logger.warning("PyMuPDF returned empty text, trying pdfplumber...")
        text = extract_text_pdfplumber(pdf_path)
    text = clean_text(text)
    logger.info(f"  Total chars: {len(text)}")

    chunks = chunk_text(text)
    logger.info(f"  Chunks created: {len(chunks)}")

    embedder = get_embedder()
    embeddings = []
    batch_size = 8
    for i in tqdm(range(0, len(chunks), batch_size),
                  desc=f"Embedding {doc_name}", leave=False):
        batch = chunks[i:i + batch_size]
        vecs = embedder.encode(batch, batch_size=batch_size,
                               show_progress_bar=False)
        embeddings.extend([v.tolist() for v in vecs])

    topics_per_chunk = [extract_topics(c) for c in chunks]
    store_in_chromadb(doc_name, pdf_path, chunks, embeddings, topics_per_chunk)
    doc_topics  = extract_topics(text)
    snippet     = " ".join(text.split()[:512])
    doc_embed   = embedder.encode(snippet).tolist()
    cache_path  = EMBEDDINGS_DIR / "text" / f"{doc_name}_pdf.json"
    with open(cache_path, "w") as f:
        json.dump({
            "node_id"   : f"pdf_{doc_name}",
            "embedding" : doc_embed,
            "topics"    : doc_topics,
            "num_chunks": len(chunks),
        }, f)
    logger.success(f"Saved embedding cache → {cache_path}")
    saved_images = extract_images_from_pdf(pdf_path, doc_name)
    logger.info(f"  Images extracted: {len(saved_images)}")
    return {
        "node_id"     : f"pdf_{doc_name}",
        "node_type"   : "PDF",
        "name"        : pdf_path.name,
        "path"        : str(pdf_path),
        "text"        : text,
        "topics"      : doc_topics,
        "embedding"   : doc_embed,
        "num_chunks"  : len(chunks),
        "images_saved": len(saved_images),
    }
def ingest_all_pdfs() -> list[dict]:
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDFs found in {PDF_DIR}")
        return []
    logger.info(f"Found {len(pdf_files)} PDF files")
    results = []
    for pdf_path in tqdm(pdf_files, desc="Ingesting PDFs"):
        try:
            results.append(process_pdf(pdf_path))
        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")

    logger.success(f"PDF ingestion complete: {len(results)}/{len(pdf_files)} successful")
    return results

if __name__ == "__main__":
    results = ingest_all_pdfs()
    print("\n── PDF Ingestion Summary ──")
    for r in results:
        print(f"  {r['name']}: {r['num_chunks']} chunks, {r['images_saved']} images")
    print(f"\nTotal PDFs processed: {len(results)}")