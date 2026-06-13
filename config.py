import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"]     = "False"
from pathlib import Path

ROOT_DIR       = Path(r"D:\MultiModal_RAG")
DATA_DIR       = ROOT_DIR / "data"
EMBEDDINGS_DIR = ROOT_DIR / "embeddings"
VECTOR_DB_DIR  = ROOT_DIR / "vector_db" / "chroma"
GRAPH_DB_DIR   = ROOT_DIR / "graph_db"
MODELS_DIR     = ROOT_DIR / "models"
PDF_DIR        = DATA_DIR / "pdfs"
PPT_DIR        = DATA_DIR / "ppts"
VIDEO_DIR      = DATA_DIR / "videos"
AUDIO_DIR      = DATA_DIR / "audio"
IMAGE_DIR      = DATA_DIR / "images"
KEYFRAME_DIR   = DATA_DIR / "images" / "keyframes"
TRANSCRIPT_DIR = DATA_DIR / "transcripts"
TEXT_EMB_DIR   = EMBEDDINGS_DIR / "text"
IMAGE_EMB_DIR  = EMBEDDINGS_DIR / "image"
NODES_CSV      = GRAPH_DB_DIR / "nodes.csv"
EDGES_CSV      = GRAPH_DB_DIR / "edges.csv"
GNN_EMB_NPY    = GRAPH_DB_DIR / "gnn_embeddings.npy"
TEXT_EMBED_MODEL   = "all-MiniLM-L6-v2"
SENTENCE_MODEL     = TEXT_EMBED_MODEL        # alias
CLIP_MODEL         = "ViT-B-32"
CLIP_PRETRAINED    = "openai"
WHISPER_MODEL      = "base"
LLM_MODEL          = "llama3"
OLLAMA_URL         = "http://localhost:11434/api/generate"
GNN_INPUT_DIM      = 384
GNN_HIDDEN_DIM     = 128
GNN_OUTPUT_DIM     = 64
GNN_EPOCHS         = 50
GNN_LR             = 0.01
TOP_K                   = 5
RETRIEVAL_WEIGHT_VECTOR = 0.5
RETRIEVAL_WEIGHT_GNN    = 0.5
KEYFRAME_INTERVAL_SEC   = 30
CHROMA_COLLECTION       = "multimodal_rag"
for _d in [
    PDF_DIR, PPT_DIR, VIDEO_DIR, AUDIO_DIR,
    IMAGE_DIR, KEYFRAME_DIR, TRANSCRIPT_DIR,
    TEXT_EMB_DIR, IMAGE_EMB_DIR,
    VECTOR_DB_DIR, GRAPH_DB_DIR,
    MODELS_DIR / "whisper",
    MODELS_DIR / "llm",
]:
    _d.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print("config.py loaded successfully")
    print(f"ROOT : {ROOT_DIR}")
    print(f"DATA : {DATA_DIR}")
    print(f"VECT : {VECTOR_DB_DIR}")
    print(f"GRAPH: {GRAPH_DB_DIR}")
    print("All directories verified.")