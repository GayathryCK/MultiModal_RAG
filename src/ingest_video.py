import os
import json
import torch
import whisper
import cv2
import open_clip
import numpy as np
from pathlib import Path
from PIL import Image
from sentence_transformers import SentenceTransformer
import chromadb
import ffmpeg
from tqdm import tqdm
from loguru import logger
from transformers import BlipProcessor, BlipForConditionalGeneration
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.suppress_warnings import *

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    VIDEO_DIR, KEYFRAME_DIR, TRANSCRIPT_DIR,
    EMBEDDINGS_DIR, VECTOR_DB_DIR,
    TEXT_EMBED_MODEL, CLIP_MODEL, CLIP_PRETRAINED,
    CHROMA_COLLECTION, KEYFRAME_INTERVAL_SEC
)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}

_clip_model      = None
_clip_preprocess = None
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
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        logger.info(f"Loading CLIP: {CLIP_MODEL}")
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAINED
        )
        _clip_model = _clip_model.to(get_device())
        _clip_model.eval()
    return _clip_model, _clip_preprocess

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
        logger.info(f"Loading embedder: {TEXT_EMBED_MODEL}")
        _embedder = SentenceTransformer(TEXT_EMBED_MODEL)
    return _embedder

def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        logger.info("Loading Whisper model...")
        _whisper_model = whisper.load_model("base", device=get_device())
    return _whisper_model

def extract_audio_from_video(video_path: Path) -> Path:
    audio_path = VIDEO_DIR / f"{video_path.stem}_audio.wav"
    if audio_path.exists():
        logger.info(f"  Audio already extracted: {audio_path.name}")
        return audio_path
    try:
        (
            ffmpeg
            .input(str(video_path))
            .output(str(audio_path),
                    vn=None,
                    acodec="pcm_s16le",
                    ar=16000,
                    ac=1)
            .overwrite_output()
            .run(quiet=True)
        )
        logger.info(f"  Audio extracted → {audio_path.name}")
    except Exception as e:
        logger.error(f"  ffmpeg failed for {video_path.name}: {e}")
        return None
    return audio_path

def transcribe_audio(audio_path: Path, video_name: str) -> dict:
    model  = get_whisper()
    result = model.transcribe(
        str(audio_path),
        language = "en",
        verbose  = False,
        fp16     = torch.cuda.is_available(),
    )
    transcript = result["text"].strip()
    segments   = [
        {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
        for s in result.get("segments", [])
    ]
    transcript_path = TRANSCRIPT_DIR / f"{video_name}_video.json"
    with open(transcript_path, "w") as f:
        json.dump({
            "video_name" : video_name,
            "transcript" : transcript,
            "segments"   : segments,
            "language"   : result.get("language", "en"),
        }, f, indent=2)
    logger.success(f"  Saved transcript → {transcript_path.name}")
    return {"transcript": transcript, "segments": segments}

def chunk_transcript(segments: list[dict],
                     chunk_duration: float = 60.0) -> list[dict]:
    chunks  = []
    current = {"text": "", "start": 0.0, "end": 0.0}
    for seg in segments:
        if (seg["end"] - current["start"]) > chunk_duration and current["text"]:
            chunks.append(current)
            current = {"text": seg["text"],
                       "start": seg["start"], "end": seg["end"]}
        else:
            current["text"] += " " + seg["text"]
            current["end"]   = seg["end"]
    if current["text"].strip():
        chunks.append(current)
    return chunks or [{"text": "", "start": 0.0, "end": 0.0}]

def extract_keyframes(video_path: Path, video_name: str,
                      interval_sec: int = KEYFRAME_INTERVAL_SEC) -> list[dict]:
    cap       = cv2.VideoCapture(str(video_path))
    fps       = cap.get(cv2.CAP_PROP_FPS)
    interval  = int(fps * interval_sec)
    keyframes = []
    frame_idx = 0
    saved     = 0
    if fps == 0:
        logger.warning(f"  Could not read FPS for {video_path.name}")
        cap.release()
        return []
    logger.info(f"  FPS: {fps:.1f} | Interval: every {interval_sec}s ({interval} frames)")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            timestamp  = frame_idx / fps
            fname      = f"{video_name}_frame{frame_idx:06d}_{int(timestamp)}s.jpg"
            fpath      = KEYFRAME_DIR / fname
            cv2.imwrite(str(fpath), frame)
            keyframes.append({
                "frame_path"   : str(fpath),
                "frame_name"   : fname,
                "timestamp_sec": timestamp,
                "frame_idx"    : frame_idx,
            })
            saved += 1

        frame_idx += 1
    cap.release()
    logger.info(f"  Keyframes saved: {saved}")
    return keyframes

def generate_caption(image: Image.Image) -> str:
    try:
        processor, model = get_blip()
        inputs = processor(image.convert("RGB"), return_tensors="pt").to(get_device())
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50)
        return processor.decode(out[0], skip_special_tokens=True).strip()
    except Exception as e:
        logger.warning(f"Caption failed: {e}")
        return ""

def get_clip_embedding(image: Image.Image) -> list[float]:
    model, preprocess = get_clip()
    tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(get_device())
    with torch.no_grad():
        feat = model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy()[0].tolist()

def embed_keyframes(keyframes: list[dict], video_name: str) -> list[dict]:
    enriched = []
    for kf in tqdm(keyframes, desc=f"  Embedding keyframes", leave=False):
        try:
            image   = Image.open(kf["frame_path"])
            caption = generate_caption(image)
            clip_emb = get_clip_embedding(image)
            text_emb = get_embedder().encode(
                caption or kf["frame_name"],
                show_progress_bar=False
            ).tolist()
            enriched.append({**kf,
                "caption"  : caption,
                "clip_emb" : clip_emb,
                "text_emb" : text_emb,
            })
        except Exception as e:
            logger.warning(f"Failed to embed keyframe {kf['frame_name']}: {e}")
    return enriched

def store_transcript_chunks(video_name: str, video_path: str,
                             chunks: list[dict], embeddings: list):
    client     = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    collection = client.get_or_create_collection(
                     name=CHROMA_COLLECTION,
                     metadata={"hnsw:space": "cosine"}
                 )
    ids, docs, metas, embs = [], [], [], []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        ids.append(f"video_{video_name}_transcript_chunk{i}")
        docs.append(chunk["text"])
        embs.append(emb)
        metas.append({
            "source"      : str(video_path),
            "source_type" : "video_transcript",
            "video_name"  : video_name,
            "chunk_idx"   : i,
            "start_sec"   : chunk["start"],
            "end_sec"     : chunk["end"],
        })
    for b in range(0, len(ids), 100):
        collection.upsert(
            ids=ids[b:b+100], embeddings=embs[b:b+100],
            documents=docs[b:b+100], metadatas=metas[b:b+100]
        )
    logger.success(f"  Stored {len(ids)} transcript chunks in ChromaDB")

def store_keyframe_embeddings(video_name: str, video_path: str,
                               keyframes: list[dict]):
    client     = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    collection = client.get_or_create_collection(
                     name=CHROMA_COLLECTION,
                     metadata={"hnsw:space": "cosine"}
                 )
    ids, docs, metas, embs = [], [], [], []
    for kf in keyframes:
        ids.append(f"video_{video_name}_frame{kf['frame_idx']}")
        docs.append(kf["caption"] or kf["frame_name"])
        embs.append(kf["text_emb"])
        metas.append({
            "source"        : str(video_path),
            "source_type"   : "video_keyframe",
            "video_name"    : video_name,
            "frame_path"    : kf["frame_path"],
            "timestamp_sec" : kf["timestamp_sec"],
            "caption"       : kf["caption"],
        })
    for b in range(0, len(ids), 100):
        collection.upsert(
            ids=ids[b:b+100], embeddings=embs[b:b+100],
            documents=docs[b:b+100], metadatas=metas[b:b+100]
        )
    logger.success(f"  Stored {len(ids)} keyframe embeddings in ChromaDB")

def process_video(video_path: Path) -> dict:
    video_name = video_path.stem
    logger.info(f"\nProcessing video: {video_path.name}")
    transcript_path = TRANSCRIPT_DIR / f"{video_name}_video.json"
    if transcript_path.exists():
        logger.info(f"  Already processed — skipping {video_path.name}")
        with open(transcript_path) as f:
            data = json.load(f)
        return {
            "video_name"  : video_name,
            "num_chunks"  : 0,
            "num_keyframes": 0,
            "skipped"     : True,
        }
    result = {
        "video_name"   : video_name,
        "num_chunks"   : 0,
        "num_keyframes": 0,
        "skipped"      : False,
    }
    audio_path = extract_audio_from_video(video_path)
    if audio_path and audio_path.exists():
        transcript_data = transcribe_audio(audio_path, video_name)
        transcript      = transcript_data["transcript"]
        segments        = transcript_data["segments"]
        chunks     = chunk_transcript(segments)
        embedder   = get_embedder()
        texts      = [c["text"] for c in chunks if c["text"].strip()]
        embeddings = []
        for i in range(0, len(texts), 8):
            batch = texts[i:i+8]
            vecs  = embedder.encode(batch, batch_size=8,
                                    show_progress_bar=False)
            embeddings.extend([v.tolist() for v in vecs])
        valid_chunks = [c for c in chunks if c["text"].strip()]
        if valid_chunks and embeddings:
            store_transcript_chunks(video_name, video_path,
                                    valid_chunks, embeddings)
            result["num_chunks"] = len(valid_chunks)
    else:
        logger.warning(f"  No audio extracted for {video_path.name}")
    keyframes = extract_keyframes(video_path, video_name)
    if keyframes:
        enriched = embed_keyframes(keyframes, video_name)
        if enriched:
            store_keyframe_embeddings(video_name, video_path, enriched)
            result["num_keyframes"] = len(enriched)
        cache_path = EMBEDDINGS_DIR / "image" / f"{video_name}_keyframes.json"
        with open(cache_path, "w") as f:
            json.dump([{
                "frame_name"   : kf["frame_name"],
                "timestamp_sec": kf["timestamp_sec"],
                "caption"      : kf.get("caption", ""),
                "clip_emb"     : kf.get("clip_emb", []),
            } for kf in enriched], f)
        logger.success(f"  Saved keyframe cache → {cache_path.name}")
    return result

def ingest_all_videos() -> list[dict]:
    video_files = [
        p for p in VIDEO_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    if not video_files:
        logger.warning(f"No video files found in {VIDEO_DIR}")
        return []
    logger.info(f"Found {len(video_files)} video files")
    results = []
    for video_path in tqdm(video_files, desc="Ingesting videos"):
        try:
            results.append(process_video(video_path))
        except Exception as e:
            logger.error(f"Failed to process {video_path.name}: {e}")
    logger.success(f"Video ingestion complete: {len(results)}/{len(video_files)} successful")
    return results

if __name__ == "__main__":
    results = ingest_all_videos()
    print("\nVideo Ingestion Summary")
    for r in results:
        status = "skipped" if r.get("skipped") else "processed"
        print(f"  {r['video_name']}: "
              f"{r['num_chunks']} transcript chunks, "
              f"{r['num_keyframes']} keyframes  [{status}]")
    print(f"\nTotal videos: {len(results)}")