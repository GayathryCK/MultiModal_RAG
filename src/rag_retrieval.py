import json
import numpy as np
import pandas as pd
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
from loguru import logger
import sys,os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.suppress_warnings import *
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    VECTOR_DB_DIR, GRAPH_DB_DIR,
    NODES_CSV, EDGES_CSV, GNN_EMB_NPY,
    TEXT_EMBED_MODEL, CHROMA_COLLECTION,
    TOP_K, RETRIEVAL_WEIGHT_VECTOR, RETRIEVAL_WEIGHT_GNN
)
_embedder   = None
_collection = None
_nodes_df   = None
_gnn_emb    = None
_node_index = None

def get_embedder():
    global _embedder
    if _embedder is None:
        logger.info(f"Loading embedder: {TEXT_EMBED_MODEL}")
        _embedder = SentenceTransformer(TEXT_EMBED_MODEL)
    return _embedder

def get_collection():
    global _collection
    if _collection is None:
        client      = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
        _collection = client.get_or_create_collection(
                          name=CHROMA_COLLECTION,
                          metadata={"hnsw:space": "cosine"}                      )
        logger.info(f"ChromaDB collection loaded: "
                    f"{_collection.count()} entries")
    return _collection

def get_gnn_data():
    global _nodes_df, _gnn_emb, _node_index
    if _gnn_emb is None:
        _nodes_df   = pd.read_csv(NODES_CSV)
        _gnn_emb    = np.load(GNN_EMB_NPY)
        norms       = np.linalg.norm(_gnn_emb, axis=1, keepdims=True) + 1e-8
        _gnn_emb    = _gnn_emb / norms
        _node_index = {
            row["node_id"]: row["node_idx"]
            for _, row in _nodes_df.iterrows()
        }
        logger.info(f"GNN embeddings loaded: {_gnn_emb.shape}")
    return _nodes_df, _gnn_emb, _node_index

def vector_search(query: str, top_k: int = None) -> list[dict]:
    if top_k is None:
        top_k = TOP_K * 3
    embedder   = get_embedder()
    collection = get_collection()
    query_emb  = embedder.encode(query).tolist()
    results    = collection.query(
        query_embeddings = [query_emb],
        n_results        = min(top_k, collection.count()),
        include          = ["documents", "metadatas", "distances"]
    )
    hits = []
    for i in range(len(results["ids"][0])):
        doc_id   = results["ids"][0][i]
        distance = results["distances"][0][i]
        score    = float(1.0 - distance)
        hits.append({
            "id"          : doc_id,
            "text"        : results["documents"][0][i],
            "metadata"    : results["metadatas"][0][i],
            "vector_score": score,
            "gnn_score"   : 0.0,
            "final_score" : score,
        })
    return hits

def get_gnn_score(query_embedding: np.ndarray,
                   doc_id: str) -> float:
    _, gnn_emb, node_index = get_gnn_data()
    node_id = map_doc_id_to_node_id(doc_id)
    if node_id not in node_index:
        return 0.0
    idx      = node_index[node_id]
    node_emb = gnn_emb[idx]
    sim = float(np.dot(query_embedding, node_emb))
    return max(0.0, sim)

def map_doc_id_to_node_id(doc_id: str) -> str:
    if "_transcript_chunk" in doc_id:
        return "video_" + doc_id.split("video_")[1].split("_transcript_chunk")[0]
    if "_frame" in doc_id and doc_id.startswith("video_"):
        return "video_" + doc_id.split("video_")[1].split("_frame")[0]
    if doc_id.startswith("pdf_") and "_chunk" in doc_id:
        return "pdf_" + doc_id.split("pdf_")[1].rsplit("_chunk", 1)[0]
    if doc_id.startswith("ppt_") and "_slide" in doc_id:
        return "ppt_" + doc_id.split("ppt_")[1].rsplit("_slide", 1)[0]
    if doc_id.startswith("audio_") and "_chunk" in doc_id:
        return "audio_" + doc_id.split("audio_")[1].rsplit("_chunk", 1)[0]
    if doc_id.startswith("img_"):
        return "image_" + doc_id[4:]
    return doc_id

def gnn_rerank(query: str, hits: list[dict]) -> list[dict]:
    _, gnn_emb, node_index = get_gnn_data()
    top_node_ids  = []
    for hit in hits[:3]:
        nid = map_doc_id_to_node_id(hit["id"])
        if nid in node_index:
            top_node_ids.append(node_index[nid])
    if top_node_ids:
        query_gnn_emb = gnn_emb[top_node_ids].mean(axis=0)
        query_gnn_emb = query_gnn_emb / (np.linalg.norm(query_gnn_emb) + 1e-8)
    else:
        query_gnn_emb = np.zeros(gnn_emb.shape[1])
    alpha = RETRIEVAL_WEIGHT_VECTOR
    beta  = RETRIEVAL_WEIGHT_GNN
    for hit in hits:
        node_id = map_doc_id_to_node_id(hit["id"])
        if node_id in node_index:
            idx       = node_index[node_id]
            node_emb  = gnn_emb[idx]
            gnn_score = float(np.dot(query_gnn_emb, node_emb))
            gnn_score = max(0.0, gnn_score)
        else:
            gnn_score = 0.0
        hit["gnn_score"]   = gnn_score
        hit["final_score"] = alpha * hit["vector_score"] + beta * gnn_score
    hits.sort(key=lambda x: x["final_score"], reverse=True)
    return hits

def assemble_context(hits: list[dict], top_k: int = None) -> str:
    if top_k is None:
        top_k = TOP_K
    context_parts = []
    seen_texts    = set()
    for i, hit in enumerate(hits[:top_k]):
        text = hit["text"].strip()
        text_key = text[:100]
        if text_key in seen_texts:
            continue
        seen_texts.add(text_key)
        meta        = hit["metadata"]
        source_type = meta.get("source_type", "unknown")
        doc_name    = meta.get("doc_name",
                     meta.get("audio_name",
                     meta.get("video_name",
                     meta.get("image_name", "unknown"))))
        if source_type == "pdf":
            label = f"[PDF: {doc_name}]"
        elif source_type == "ppt":
            slide = meta.get("slide_title", f"slide {meta.get('slide_num','?')}")
            label = f"[PPT: {doc_name} — {slide}]"
        elif source_type == "audio":
            start = meta.get("start_sec", 0)
            label = f"[Audio: {doc_name} @ {start:.0f}s]"
        elif source_type == "video_transcript":
            start = meta.get("start_sec", 0)
            label = f"[Video: {doc_name} @ {start:.0f}s]"
        elif source_type == "video_keyframe":
            ts    = meta.get("timestamp_sec", 0)
            label = f"[Keyframe: {doc_name} @ {ts:.0f}s]"
        elif source_type == "image":
            label = f"[Image: {doc_name}]"
        else:
            label = f"[{source_type}: {doc_name}]"

        context_parts.append(f"{label}\n{text}")
    return "\n\n".join(context_parts)

def retrieve(query: str,
             top_k: int = None,
             use_gnn: bool = True) -> dict:
    if top_k is None:
        top_k = TOP_K
    logger.info(f"Query: {query[:80]}")
    hits = vector_search(query, top_k=top_k * 3)
    logger.info(f"  Vector search: {len(hits)} candidates")
    if use_gnn and len(hits) > 0:
        hits = gnn_rerank(query, hits)
        logger.info(f"  GNN re-ranking: done")
    context = assemble_context(hits, top_k=top_k)
    logger.info(f"  Context assembled: {len(context)} chars")
    return {
        "query"  : query,
        "hits"   : hits[:top_k],
        "context": context,
        "top_k"  : top_k,
    }

if __name__ == "__main__":
    test_queries = [
        "What is gradient descent and how does it work?",
        "Explain the perceptron algorithm",
        "How do convolutional neural networks work?",
        "What is the difference between supervised and unsupervised learning?",
    ]
    for query in test_queries:
        print(f"Query: {query}")
        result = retrieve(query, top_k=3)
        for i, hit in enumerate(result["hits"]):
            meta  = hit["metadata"]
            stype = meta.get("source_type", "?")
            name  = meta.get("doc_name",
                    meta.get("audio_name",
                    meta.get("video_name", "?")))
            print(f"\n  [{i+1}] {stype:20s} | "
                  f"vector={hit['vector_score']:.3f} | "
                  f"gnn={hit['gnn_score']:.3f} | "
                  f"final={hit['final_score']:.3f}")
            print(f"       {name}")
            print(f"       {hit['text'][:120]}...")
        print(f"\n Context Preview")
        print(result["context"][:400])
        print("...")