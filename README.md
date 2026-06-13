# 🧠 Multimodal RAG + GNN Knowledge System

A fully **local, privacy-preserving** academic knowledge system combining:
- **Multimodal ingestion**: PDFs, PPTs, Videos, Audio, Images
- **Knowledge Graph**: Heterogeneous graph with GNN reasoning
- **RAG Retrieval**: ChromaDB vector store + GNN re-ranking
- **Local LLM**: LLaMA 3 / Mistral via Ollama (no cloud, no API keys)
- **Streamlit UI**: Interactive Q&A interface

---

## 📦 Installation

### 1. Prerequisites
```bash
# Python 3.10+
python --version

# Install ffmpeg (for video processing)
# Ubuntu/Debian:
sudo apt install ffmpeg tesseract-ocr

# macOS:
brew install ffmpeg tesseract
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt

# Install spaCy language model
python -m spacy download en_core_web_sm

# Install PyTorch Geometric (match your PyTorch version)
pip install torch-geometric
```

### 3. Install & configure Ollama (local LLM)
```bash
# Install Ollama: https://ollama.com
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model (pick one):
ollama pull llama3       # Recommended (8B, ~5GB)
ollama pull mistral      # Alternative (7B, ~4GB)
ollama pull phi3         # Lightweight (3.8B, ~2GB)
```

---

## 🚀 Quick Start

### Step 1: Add your files
```
knowledge_system/
├── data/
│   ├── pdfs/        ← Put lecture PDFs here
│   ├── ppts/        ← Put PPTX slides here
│   ├── videos/      ← Put MP4 lecture recordings here
│   ├── audio/       ← Put WAV/MP3 audio files here
│   └── images/      ← Put PNG/JPG diagrams here
```

### Step 2: Build the pipeline
```bash
cd knowledge_system
python build_pipeline.py --course "Machine Learning" --professor "Dr. Smith"

# Skip GNN training (faster, for testing):
python build_pipeline.py --skip-gnn
```

### Step 3: Launch the UI
```bash
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

---

## 🗂️ Project Structure
```
knowledge_system/
├── app.py                    # Streamlit UI
├── build_pipeline.py         # Master build script
├── config.py                 # All settings
├── requirements.txt
├── src/
│   ├── ingest_pdf.py         # PDF ingestion (PyMuPDF + pdfplumber)
│   ├── ingest_ppt.py         # PPT ingestion (python-pptx + CLIP)
│   ├── ingest_video.py       # Video ingestion (ffmpeg + Whisper + OpenCV)
│   ├── ingest_image_audio.py # Image (BLIP/CLIP) + Audio (Whisper)
│   ├── knowledge_graph.py    # Heterogeneous graph builder
│   ├── gnn_layer.py          # GCN / GAT training + re-ranking
│   ├── rag_retrieval.py      # ChromaDB + RAG pipeline
│   └── llm_generation.py     # Local LLM (Ollama) Q&A
└── data/, embeddings/, graph_db/, vector_db/, models/
```

---

## ⚙️ Configuration (config.py)

| Setting | Default | Description |
|---|---|---|
| `TEXT_EMBED_MODEL` | all-MiniLM-L6-v2 | SentenceTransformers model |
| `WHISPER_MODEL` | base | Whisper size (tiny/base/small/medium) |
| `LLM_MODEL` | llama3 | Ollama model name |
| `TOP_K_RETRIEVAL` | 5 | Documents retrieved per query |
| `FRAME_INTERVAL_SEC` | 10 | Keyframe extraction interval |
| `GNN_EPOCHS` | 50 | GNN training epochs |

---

## 🧪 Run Individual Modules
```bash
# Test PDF ingestion only
python -m src.ingest_pdf

# Test PPT ingestion only
python -m src.ingest_ppt

# Test video ingestion
python -m src.ingest_video

# Test RAG retrieval
python -m src.rag_retrieval

# Train GNN manually
python -m src.gnn_layer

# Test Q&A
python -m src.llm_generation
```

---

## 🔧 Troubleshooting

**"No documents found"** → Make sure files are in the correct `data/` subdirectory  
**"CUDA out of memory"** → Use smaller Whisper model (`tiny` or `base`)  
**"Ollama not found"** → Install Ollama and run `ollama pull llama3`  
**"PyTorch Geometric missing"** → `pip install torch-geometric` (GNN will be skipped otherwise)  
**"spaCy model missing"** → `python -m spacy download en_core_web_sm`

---

## 📚 Tech Stack

| Component | Technology |
|---|---|
| Text embeddings | SentenceTransformers (all-MiniLM-L6-v2) |
| Image embeddings | CLIP (ViT-B/32) |
| Image captioning | BLIP |
| Speech-to-text | OpenAI Whisper (local) |
| OCR | Tesseract |
| Graph learning | PyTorch Geometric (GCN / GAT) |
| Vector DB | ChromaDB |
| LLM inference | Ollama (LLaMA 3 / Mistral / Phi-3) |
| UI | Streamlit |
