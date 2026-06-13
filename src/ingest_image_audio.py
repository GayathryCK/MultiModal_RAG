import os
import json
import torch
import whisper
import open_clip
import numpy as np
from pathlib import Path
from PIL import Image
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm
from loguru import logger
from transformers import BlipProcessor, BlipForConditionalGeneration

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.suppress_warnings import *

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    IMAGE_DIR, AUDIO_DIR, TRANSCRIPT_DIR,
    EMBEDDINGS_DIR, VECTOR_DB_DIR,
    TEXT_EMBED_MODEL, CLIP_MODEL, CLIP_PRETRAINED,
    CHROMA_COLLECTION
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}

_clip_model     = None
_clip_preprocess = None
_clip_tokenizer  = None
_blip_processor  = None
_blip_model      = None
_embedder        = None
_whisper_model   = None
_device          = None

def get_device():
    global _device
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {_device}")
    return _device

def get_clip():
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if _clip_model is None:
        logger.info(f"Loading CLIP model: {CLIP_MODEL}")
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAINED
        )
        _clip_tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
        _clip_model = _clip_model.to(get_device())
        _clip_model.eval()
    return _clip_model, _clip_preprocess, _clip_tokenizer

def get_blip():
    global _blip_processor, _blip_model
    if _blip_model is None:
        logger.info("Loading BLIP captioning model...")
        _blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        _blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(get_device())
        _blip_model.eval()
    return _blip_processor, _blip_model

def get_embedder():
    global _embedder
    if _embedder is None:
        logger.info(f"Loading text embedder: {TEXT_EMBED_MODEL}")
        _embedder = SentenceTransformer(TEXT_EMBED_MODEL)
    return _embedder

def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        logger.info("Loading Whisper model (base)...")
        _whisper_model = whisper.load_model("base", device=get_device())
    return _whisper_model

def generate_caption(image: Image.Image) -> str:
    try:
        processor, model = get_blip()
        inputs = processor(image.convert("RGB"), return_tensors="pt").to(get_device())
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption.strip()
    except Exception as e:
        logger.warning(f"BLIP caption failed: {e}")
        return ""

def get_clip_embedding(image: Image.Image) -> list[float]:
    model, preprocess, _ = get_clip()
    img_tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(get_device())
    with torch.no_grad():
        features = model.encode_image(img_tensor)
        features = features / features.norm(dim=-1, keepdim=True)  # normalize
    return features.cpu().numpy()[0].tolist()

def process_image(image_path: Path) -> dict | None:
    try:
        image   = Image.open(image_path)
        if image.width < 64 or image.height < 64:
            return None
        caption   = generate_caption(image)
        clip_emb  = get_clip_embedding(image)
        text_emb  = get_embedder().encode(caption).tolist() if caption else [0.0] * 384
        return {
            "image_path" : str(image_path),
            "image_name" : image_path.name,
            "caption"    : caption,
            "clip_emb"   : clip_emb,
            "text_emb"   : text_emb,
            "width"      : image.width,
            "height"     : image.height,
        }
    except Exception as e:
        logger.warning(f"Failed to process image {image_path.name}: {e}")
        return None

def ingest_all_images() -> list[dict]:
    image_files = [
        p for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not image_files:
        logger.warning(f"No image files found in {IMAGE_DIR}")
        return []
    logger.info(f"Found {len(image_files)} images to process")
    client     = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    collection = client.get_or_create_collection(
                     name=CHROMA_COLLECTION,
                     metadata={"hnsw:space": "cosine"}
                 )
    results    = []
    ids, docs, metas, embs = [], [], [], []
    for image_path in tqdm(image_files, desc="Processing images"):
        result = process_image(image_path)
        if result is None:
            continue
        img_id = f"img_{image_path.stem}"
        ids.append(img_id)
        docs.append(result["caption"] or image_path.name)
        embs.append(result["text_emb"])
        metas.append({
            "source"      : str(image_path),
            "source_type" : "image",
            "image_name"  : image_path.name,
            "caption"     : result["caption"],
            "width"       : result["width"],
            "height"      : result["height"],
        })
        cache_path = EMBEDDINGS_DIR / "image" / f"{image_path.stem}.json"
        with open(cache_path, "w") as f:
            json.dump({
                "image_name" : image_path.name,
                "image_path" : str(image_path),
                "caption"    : result["caption"],
                "clip_emb"   : result["clip_emb"],
                "text_emb"   : result["text_emb"],
            }, f)
        results.append(result)
    for b in range(0, len(ids), 100):
        collection.upsert(
            ids        = ids[b:b+100],
            embeddings = embs[b:b+100],
            documents  = docs[b:b+100],
            metadatas  = metas[b:b+100],
        )
    logger.success(f"Stored {len(ids)} image embeddings in ChromaDB")
    logger.success(f"Saved {len(ids)} CLIP embedding caches to embeddings/image/")
    return results

def transcribe_audio(audio_path: Path) -> dict:
    model  = get_whisper()
    logger.info(f"  Transcribing: {audio_path.name}")
    result = model.transcribe(
        str(audio_path),
        language    = "en",
        verbose     = False,
        fp16        = torch.cuda.is_available(),
    )
    transcript = result["text"].strip()
    segments   = [
        {
            "start" : seg["start"],
            "end"   : seg["end"],
            "text"  : seg["text"].strip(),
        }
        for seg in result.get("segments", [])
    ]
    return {
        "transcript" : transcript,
        "segments"   : segments,
        "language"   : result.get("language", "en"),
    }

def chunk_transcript(segments: list[dict],
                     chunk_duration: float = 60.0) -> list[dict]:
    chunks     = []
    current    = {"text": "", "start": 0.0, "end": 0.0}
    for seg in segments:
        if (seg["end"] - current["start"]) > chunk_duration and current["text"]:
            chunks.append(current)
            current = {"text": seg["text"], "start": seg["start"], "end": seg["end"]}
        else:
            current["text"] += " " + seg["text"]
            current["end"]   = seg["end"]
    if current["text"].strip():
        chunks.append(current)
    return chunks

def process_audio(audio_path: Path) -> dict:
    audio_name = audio_path.stem
    logger.info(f"Processing audio: {audio_path.name}")
    result     = transcribe_audio(audio_path)
    transcript = result["transcript"]
    segments   = result["segments"]
    if not transcript:
        logger.warning(f"  Empty transcript for {audio_path.name}")
        return {"audio_name": audio_name, "num_chunks": 0}
    logger.info(f"  Transcript chars: {len(transcript)}")
    transcript_path = TRANSCRIPT_DIR / f"{audio_name}.json"
    with open(transcript_path, "w") as f:
        json.dump({
            "audio_name" : audio_name,
            "audio_path" : str(audio_path),
            "transcript" : transcript,
            "segments"   : segments,
            "language"   : result["language"],
        }, f, indent=2)
    logger.success(f"  Saved transcript → {transcript_path}")
    chunks = chunk_transcript(segments, chunk_duration=60.0)
    if not chunks:
        chunks = [{"text": transcript, "start": 0.0, "end": 0.0}]
    logger.info(f"  Transcript chunks: {len(chunks)}")
    embedder   = get_embedder()
    texts      = [c["text"] for c in chunks]
    embeddings = []
    for i in range(0, len(texts), 8):
        batch = texts[i:i+8]
        vecs  = embedder.encode(batch, batch_size=8, show_progress_bar=False)
        embeddings.extend([v.tolist() for v in vecs])
    client     = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    collection = client.get_or_create_collection(
                     name=CHROMA_COLLECTION,
                     metadata={"hnsw:space": "cosine"}
                 )
    ids, docs, metas, embs = [], [], [], []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        ids.append(f"audio_{audio_name}_chunk{i}")
        docs.append(chunk["text"])
        embs.append(emb)
        metas.append({
            "source"      : str(audio_path),
            "source_type" : "audio",
            "audio_name"  : audio_name,
            "chunk_idx"   : i,
            "start_sec"   : chunk["start"],
            "end_sec"     : chunk["end"],
        })
    for b in range(0, len(ids), 100):
        collection.upsert(
            ids=ids[b:b+100], embeddings=embs[b:b+100],
            documents=docs[b:b+100], metadatas=metas[b:b+100]
        )
    logger.success(f"Stored {len(ids)} audio chunks in ChromaDB for [{audio_name}]")
    return {
        "audio_name"  : audio_name,
        "transcript"  : transcript,
        "num_chunks"  : len(chunks),
        "num_segments": len(segments),
    }

def ingest_all_audio() -> list[dict]:
    audio_files = [
        p for p in AUDIO_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    ]
    if not audio_files:
        logger.warning(f"No audio files found in {AUDIO_DIR}")
        return []
    logger.info(f"Found {len(audio_files)} audio files")
    results = []
    for audio_path in tqdm(audio_files, desc="Processing audio"):
        try:
            results.append(process_audio(audio_path))
        except Exception as e:
            logger.error(f"Failed to process {audio_path.name}: {e}")
    logger.success(f"Audio ingestion complete: {len(results)}/{len(audio_files)} successful")
    return results

if __name__ == "__main__":
    print("=" * 50)
    print("STAGE 1 Image Ingestion (CLIP + BLIP)")
    print("=" * 50)
    image_results = ingest_all_images()
    print("\n" + "=" * 50)
    print("STAGE 2 Audio Ingestion (Whisper)")
    print("=" * 50)
    audio_results = ingest_all_audio()
    print("\n Image + Audio Ingestion Summary ")
    print(f"  Images processed : {len(image_results)}")
    for r in image_results[:5]:
        print(f"    {r['image_name']}: \"{r['caption'][:60]}\"")
    if len(image_results) > 5:
        print(f"     and {len(image_results)-5} more")
    print(f"\n  Audio files processed: {len(audio_results)}")
    for r in audio_results:
        print(f"    {r['audio_name']}: {r['num_chunks']} chunks")
    print("\nDone!")