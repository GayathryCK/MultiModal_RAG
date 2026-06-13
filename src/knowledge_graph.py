import os
import json
import csv
import numpy as np
import networkx as nx
from pathlib import Path
from sentence_transformers import SentenceTransformer
from loguru import logger
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.suppress_warnings import *

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PDF_DIR, PPT_DIR, VIDEO_DIR, AUDIO_DIR, IMAGE_DIR,
    TRANSCRIPT_DIR, EMBEDDINGS_DIR,
    NODES_CSV, EDGES_CSV, GNN_EMB_NPY,
    TEXT_EMBED_MODEL, GRAPH_DB_DIR
)

_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        logger.info(f"Loading embedder: {TEXT_EMBED_MODEL}")
        _embedder = SentenceTransformer(TEXT_EMBED_MODEL)
    return _embedder

def add_course_nodes(G: nx.DiGraph):
    G.add_node("course_mit6036", node_type="Course",
               name="MIT 6.036 Machine Learning",
               description="Introduction to Machine Learning — MIT Fall 2020")
    G.add_node("prof_broderick", node_type="Professor",
               name="Tamara Broderick",
               description="MIT Professor — Machine Learning")
    G.add_node("prof_sontag", node_type="Professor",
               name="David Sontag",
               description="MIT Professor — Machine Learning")
    G.add_edge("prof_broderick", "course_mit6036", relation="teaches")
    G.add_edge("prof_sontag",    "course_mit6036", relation="teaches")
    logger.info("  Added Course + Professor nodes")

def add_pdf_nodes(G: nx.DiGraph) -> list[str]:
    pdf_node_ids = []
    cache_files  = list((EMBEDDINGS_DIR / "text").glob("*_pdf.json"))
    for cache_path in tqdm(cache_files, desc="  PDF nodes", leave=False):
        with open(cache_path) as f:
            data = json.load(f)
        node_id = data["node_id"]
        G.add_node(node_id,
                   node_type   = "PDF",
                   name        = cache_path.stem.replace("_pdf", ""),
                   num_chunks  = data.get("num_chunks", 0),
                   embedding   = data["embedding"])

        G.add_edge("course_mit6036", node_id, relation="contains")
        pdf_node_ids.append(node_id)
        for topic in data.get("topics", [])[:15]:
            topic_id = f"topic_{topic.replace(' ', '_')[:50]}"
            if not G.has_node(topic_id):
                G.add_node(topic_id, node_type="Topic", name=topic)
            G.add_edge(node_id, topic_id, relation="mentions")
    logger.info(f"  PDF nodes: {len(pdf_node_ids)}")
    return pdf_node_ids

def add_ppt_nodes(G: nx.DiGraph) -> list[str]:
    ppt_node_ids = []
    cache_files  = list((EMBEDDINGS_DIR / "text").glob("*_ppt.json"))
    for cache_path in tqdm(cache_files, desc="  PPT nodes", leave=False):
        with open(cache_path) as f:
            data = json.load(f)
        node_id = data["node_id"]
        G.add_node(node_id,
                   node_type   = "PPT",
                   name        = cache_path.stem.replace("_ppt", ""),
                   num_slides  = data.get("num_slides", 0),
                   embedding   = data["embedding"])
        G.add_edge("course_mit6036", node_id, relation="contains")
        ppt_node_ids.append(node_id)
        for topic in data.get("topics", [])[:15]:
            topic_id = f"topic_{topic.replace(' ', '_')[:50]}"
            if not G.has_node(topic_id):
                G.add_node(topic_id, node_type="Topic", name=topic)
            G.add_edge(node_id, topic_id, relation="mentions")
    logger.info(f"  PPT nodes: {len(ppt_node_ids)}")
    return ppt_node_ids

def add_video_nodes(G: nx.DiGraph) -> list[str]:
    video_node_ids = []
    for video_path in VIDEO_DIR.iterdir():
        if video_path.suffix.lower() not in {".mp4", ".avi", ".mkv", ".mov"}:
            continue
        video_name = video_path.stem
        node_id    = f"video_{video_name}"
        transcript_path = TRANSCRIPT_DIR / f"{video_name}_video.json"
        if not transcript_path.exists():
            transcript_path = TRANSCRIPT_DIR / f"{video_name}.json"
        transcript_text = ""
        if transcript_path.exists():
            with open(transcript_path) as f:
                data = json.load(f)
            transcript_text = data.get("transcript", "")
        snippet   = " ".join(transcript_text.split()[:256]) if transcript_text else video_name
        embedding = get_embedder().encode(snippet).tolist()
        G.add_node(node_id,
                   node_type  = "Video",
                   name       = video_name,
                   embedding  = embedding,
                   has_transcript = transcript_path.exists())
        G.add_edge("course_mit6036", node_id, relation="contains")
        video_node_ids.append(node_id)
        if "broderick" in video_name.lower() or "lecture" in video_name.lower():
            G.add_edge("prof_broderick", node_id, relation="delivers")
        if "sontag" in video_name.lower():
            G.add_edge("prof_sontag", node_id, relation="delivers")
    logger.info(f"  Video nodes: {len(video_node_ids)}")
    return video_node_ids

def add_audio_nodes(G: nx.DiGraph) -> list[str]:
    audio_node_ids = []
    for transcript_path in TRANSCRIPT_DIR.glob("*.json"):
        if "_video" in transcript_path.stem:
            continue
        with open(transcript_path) as f:
            data = json.load(f)
        audio_name = data.get("audio_name", transcript_path.stem)
        node_id    = f"audio_{audio_name}"
        transcript = data.get("transcript", "")
        snippet   = " ".join(transcript.split()[:256]) if transcript else audio_name
        embedding = get_embedder().encode(snippet).tolist()
        G.add_node(node_id,
                   node_type  = "Audio",
                   name       = audio_name,
                   embedding  = embedding,
                   has_transcript = bool(transcript))

        G.add_edge("course_mit6036", node_id, relation="contains")
        audio_node_ids.append(node_id)
        video_node_id = f"video_{audio_name}"
        if G.has_node(video_node_id):
            G.add_edge(video_node_id, node_id, relation="references")
    logger.info(f"  Audio nodes: {len(audio_node_ids)}")
    return audio_node_ids

def add_image_nodes(G: nx.DiGraph) -> list[str]:
    image_node_ids = []
    cache_files    = list((EMBEDDINGS_DIR / "image").glob("*.json"))
    for cache_path in tqdm(cache_files[:200], desc="  Image nodes", leave=False):
        with open(cache_path) as f:
            raw = json.load(f)
        if isinstance(raw, list):
            for kf in raw[:5]:
                frame_name = kf.get("frame_name", cache_path.stem)
                node_id = f"image_{Path(frame_name).stem}"
                caption = kf.get("caption", frame_name)
                text_emb = kf.get("clip_emb", [])
                if not text_emb:
                    text_emb = get_embedder().encode(caption).tolist()
                if len(text_emb) != 384:
                    text_emb = (text_emb + [0.0] * 384)[:384]
                G.add_node(node_id,
                           node_type="Image",
                           name=frame_name,
                           caption=caption,
                           embedding=text_emb)
                video_name = cache_path.stem.replace("_keyframes", "")
                video_node = f"video_{video_name}"
                if G.has_node(video_node):
                    G.add_edge(video_node, node_id, relation="contains")
                else:
                    G.add_edge("course_mit6036", node_id, relation="contains")

                image_node_ids.append(node_id)
            continue
        data = raw
        image_name = data.get("image_name", cache_path.stem)
        node_id = f"image_{cache_path.stem}"
        text_emb = data.get("text_emb", [])
        if not text_emb:
            caption = data.get("caption", image_name)
            text_emb = get_embedder().encode(caption).tolist()
        if len(text_emb) != 384:
            text_emb = (text_emb + [0.0] * 384)[:384]
        G.add_node(node_id,
                   node_type="Image",
                   name=image_name,
                   caption=data.get("caption", ""),
                   embedding=text_emb)
        image_name_lower = image_name.lower()
        linked = False
        for node_id2 in G.nodes():
            ntype = G.nodes[node_id2].get("node_type", "")
            nname = G.nodes[node_id2].get("name", "").lower()
            if ntype in {"PDF", "PPT"} and nname and nname in image_name_lower:
                G.add_edge(node_id2, node_id, relation="contains")
                linked = True
                break
        if not linked:
            G.add_edge("course_mit6036", node_id, relation="contains")
        image_node_ids.append(node_id)
    logger.info(f"  Image nodes: {len(image_node_ids)}")
    return image_node_ids

def add_cross_document_edges(G: nx.DiGraph):
    topic_to_docs = {}
    for node_id, data in G.nodes(data=True):
        if data.get("node_type") == "Topic":
            docs = [src for src, dst, d in G.edges(data=True)
                    if dst == node_id and d.get("relation") == "mentions"]
            topic_to_docs[node_id] = docs
    edges_added = 0
    for topic_id, docs in topic_to_docs.items():
        if len(docs) < 2:
            continue
        for i in range(len(docs)):
            for j in range(i+1, len(docs)):
                if not G.has_edge(docs[i], docs[j]):
                    G.add_edge(docs[i], docs[j],
                               relation="covers",
                               via_topic=topic_id)
                    edges_added += 1
    logger.info(f"  Cross-document edges added: {edges_added}")

def export_graph(G: nx.DiGraph):
    node_list      = list(G.nodes(data=True))
    node_index     = {nid: i for i, (nid, _) in enumerate(node_list)}
    embeddings     = []
    default_embed  = [0.0] * 384
    with open(NODES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node_idx", "node_id", "node_type", "name"])
        for i, (node_id, data) in enumerate(node_list):
            writer.writerow([
                i,
                node_id,
                data.get("node_type", "Unknown"),
                data.get("name", node_id)[:100],
            ])
            emb = data.get("embedding", default_embed)
            if len(emb) < 384:
                emb = emb + [0.0] * (384 - len(emb))
            elif len(emb) > 384:
                emb = emb[:384]
            embeddings.append(emb)
    with open(EDGES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["src_idx", "dst_idx", "src_id", "dst_id", "relation"])
        for src, dst, data in G.edges(data=True):
            if src in node_index and dst in node_index:
                writer.writerow([
                    node_index[src],
                    node_index[dst],
                    src,
                    dst,
                    data.get("relation", "related"),
                ])
    emb_matrix = np.array(embeddings, dtype=np.float32)
    np.save(GNN_EMB_NPY, emb_matrix)
    logger.success(f"Exported {len(node_list)} nodes → {NODES_CSV}")
    logger.success(f"Exported {G.number_of_edges()} edges → {EDGES_CSV}")
    logger.success(f"Saved embeddings matrix {emb_matrix.shape} → {GNN_EMB_NPY}")
    return node_index, emb_matrix

def build_knowledge_graph() -> nx.DiGraph:
    logger.info("Building Knowledge Graph...")
    G = nx.DiGraph()
    logger.info("Adding nodes...")
    add_course_nodes(G)
    add_pdf_nodes(G)
    add_ppt_nodes(G)
    add_video_nodes(G)
    add_audio_nodes(G)
    add_image_nodes(G)
    logger.info("Adding cross-document topic edges...")
    add_cross_document_edges(G)
    logger.info(f"\nGraph Statistics:")
    logger.info(f"  Total nodes : {G.number_of_nodes()}")
    logger.info(f"  Total edges : {G.number_of_edges()}")
    type_counts = {}
    for _, data in G.nodes(data=True):
        t = data.get("node_type", "Unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, count in sorted(type_counts.items()):
        logger.info(f"    {t:12s}: {count}")
    edge_counts = {}
    for _, _, data in G.edges(data=True):
        r = data.get("relation", "unknown")
        edge_counts[r] = edge_counts.get(r, 0) + 1
    for r, count in sorted(edge_counts.items()):
        logger.info(f"    {r:15s}: {count}")
    logger.info("\nExporting graph to CSV + NPY...")
    export_graph(G)
    logger.success("Knowledge graph built successfully!")
    return G

if __name__ == "__main__":
    G = build_knowledge_graph()

    print("\nKnowledge Graph Summary")
    print(f"  Nodes : {G.number_of_nodes()}")
    print(f"  Edges : {G.number_of_edges()}")
    print(f"\n  Node types:")
    type_counts = {}
    for _, d in G.nodes(data=True):
        t = d.get("node_type", "?")
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, c in sorted(type_counts.items()):
        print(f"    {t:15s}: {c}")
    print(f"\n  Files saved:")
    print(f"    {NODES_CSV}")
    print(f"    {EDGES_CSV}")
    print(f"    {GNN_EMB_NPY}")