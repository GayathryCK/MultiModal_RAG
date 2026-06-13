import sys
import time
import json
import hashlib
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).parent.parent))
import suppress_warnings
from config import *
from watchdog.observers import Observer
from watchdog.events    import FileSystemEventHandler
import chromadb
from loguru             import logger

HASH_STORE_PATH = GRAPH_DB_DIR / "file_hashes.json"
EXT_TO_SOURCE = {
    ".pdf":  "pdf",
    ".pptx": "ppt",
    ".mp4":  ["video_transcript", "video_keyframe"],
    ".wav":  "audio",
    ".png":  "image",
    ".jpg":  "image",
    ".jpeg": "image",
}
def get_cache_path(doc_name, ext):
    cache_map = {
        ".pdf":  EMBEDDINGS_DIR / "text" / f"{doc_name}_pdf.json",
        ".pptx": EMBEDDINGS_DIR / "text" / f"{doc_name}_ppt.json",
        ".mp4":  TRANSCRIPT_DIR / f"{doc_name}_video.json",
        ".wav":  TRANSCRIPT_DIR / f"{doc_name}_audio.json",
    }
    return cache_map.get(ext)

def load_hash_store():
    """Load hash store from disk. Returns empty dict if not found."""
    if HASH_STORE_PATH.exists():
        with open(HASH_STORE_PATH, "r") as f:
            return json.load(f)
    return {}

def save_hash_store(store):
    GRAPH_DB_DIR.mkdir(parents=True, exist_ok=True)
    with open(HASH_STORE_PATH, "w") as f:
        json.dump(store, f, indent=2)

def compute_hash(file_path):
    md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5.update(chunk)
        return md5.hexdigest()
    except Exception as e:
        logger.error(f"Could not hash {file_path.name}: {e}")
        return None

def initialise_hash_store():
    existing_store = load_hash_store()
    if existing_store:
        logger.info(
            f"Hash store already exists — "
            f"{len(existing_store)} files tracked"
        )
        return existing_store
    logger.info("Hash store not found — building for the first time...")
    logger.info("(This only happens once — no ChromaDB changes will be made)")
    logger.info("")
    new_store  = {}
    all_dirs   = [PDF_DIR, PPT_DIR, VIDEO_DIR, AUDIO_DIR, IMAGE_DIR]
    count      = 0
    skipped    = 0
    for folder in all_dirs:
        folder = Path(folder)
        if not folder.exists():
            logger.warning(f"  Folder not found, skipping: {folder}")
            continue
        folder_files = [
            f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in EXT_TO_SOURCE
        ]
        if folder_files:
            logger.info(f"  Scanning: {folder.name}/ ({len(folder_files)} files)")
        for file_path in folder_files:
            doc_name     = file_path.stem
            current_hash = compute_hash(file_path)
            if current_hash:
                new_store[doc_name] = {
                    "hash":     current_hash,
                    "file":     file_path.name,
                    "folder":   str(folder),
                    "hashed_at": datetime.now().isoformat()
                }
                logger.info(f"     {file_path.name} → {current_hash[:12]}...")
                count += 1
            else:
                logger.warning(f"      Could not hash: {file_path.name}")
                skipped += 1
    save_hash_store(new_store)
    logger.info("")
    logger.success(f"Hash store built successfully!")
    logger.success(f"  Files hashed:  {count}")
    logger.success(f"  Files skipped: {skipped}")
    logger.success(f"  Saved to: {HASH_STORE_PATH}")
    logger.info("")
    return new_store

def has_content_changed(file_path, hash_store):
    doc_name     = Path(file_path).stem
    current_hash = compute_hash(file_path)
    if current_hash is None:
        return False
    stored_entry = hash_store.get(doc_name)
    if stored_entry is None:
        logger.info(f"New file detected: {Path(file_path).name}")
        hash_store[doc_name] = {
            "hash":      current_hash,
            "file":      Path(file_path).name,
            "folder":    str(Path(file_path).parent),
            "hashed_at": datetime.now().isoformat()
        }
        save_hash_store(hash_store)
        return True
    stored_hash = stored_entry.get("hash", stored_entry) \
                  if isinstance(stored_entry, dict) else stored_entry
    if current_hash != stored_hash:
        logger.info(f"Content change detected: {Path(file_path).name}")
        logger.info(f"  Old hash: {stored_hash[:16]}...")
        logger.info(f"  New hash: {current_hash[:16]}...")
        hash_store[doc_name] = {
            "hash":       current_hash,
            "file":       Path(file_path).name,
            "folder":     str(Path(file_path).parent),
            "hashed_at":  datetime.now().isoformat(),
            "prev_hash":  stored_hash   # keep previous for audit trail
        }
        save_hash_store(hash_store)
        return True
    logger.debug(f"False trigger ignored: {Path(file_path).name} (hash unchanged)")
    return False

def delete_chromadb_entries(collection, doc_name, source_type):
    try:
        if isinstance(source_type, list):
            # Video has two source types
            for st in source_type:
                collection.delete(where={"$and": [
                    {"doc_name":    {"$eq": doc_name}},
                    {"source_type": {"$eq": st}}
                ]})
        else:
            collection.delete(where={"$and": [
                {"doc_name":    {"$eq": doc_name}},
                {"source_type": {"$eq": source_type}}
            ]})
        logger.success(f"ChromaDB entries deleted for: {doc_name}")
    except Exception as e:
        logger.error(f"Failed to delete ChromaDB entries for {doc_name}: {e}")

def delete_cache(doc_name, ext):
    cache = get_cache_path(doc_name, ext)
    if cache and cache.exists():
        cache.unlink()
        logger.success(f"Cache deleted: {cache.name}")

class DataFolderWatcher(FileSystemEventHandler):
    def __init__(self):
        self.client     = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
        self.collection = self.client.get_collection(CHROMA_COLLECTION)
        self.hash_store = initialise_hash_store()
        logger.info(
            f"Watcher ready — tracking {len(self.hash_store)} files"
        )

    def on_deleted(self, event):
        if event.is_directory:
            return
        file_path   = Path(event.src_path)
        doc_name    = file_path.stem
        ext         = file_path.suffix.lower()
        source_type = EXT_TO_SOURCE.get(ext)
        if not source_type:
            return
        logger.warning(f"[DELETED] {file_path.name}")
        delete_chromadb_entries(self.collection, doc_name, source_type)
        delete_cache(doc_name, ext)
        if doc_name in self.hash_store:
            del self.hash_store[doc_name]
            save_hash_store(self.hash_store)
            logger.info(f"Hash entry removed for: {doc_name}")
        logger.success(
            f"[DELETED] Full cleanup done for: {file_path.name}\n"
        )
    def on_modified(self, event):
        if event.is_directory:
            return
        file_path   = Path(event.src_path)
        doc_name    = file_path.stem
        ext         = file_path.suffix.lower()
        source_type = EXT_TO_SOURCE.get(ext)
        if not source_type:
            return

        if not has_content_changed(file_path, self.hash_store):
            return

        logger.warning(f"[MODIFIED] {file_path.name} — content changed")
        delete_chromadb_entries(self.collection, doc_name, source_type)
        delete_cache(doc_name, ext)
        logger.success(f"[MODIFIED] Cleanup done for: {file_path.name}")
        logger.info(
            f"           Old vectors removed from ChromaDB \n"
            f"           Run build_pipeline.py to add new vectors \n"
        )

    def on_created(self, event):
        if event.is_directory:
            return
        file_path   = Path(event.src_path)
        doc_name    = file_path.stem
        ext         = file_path.suffix.lower()
        if ext not in EXT_TO_SOURCE:
            return

        current_hash = compute_hash(file_path)
        if current_hash:
            self.hash_store[doc_name] = {
                "hash":      current_hash,
                "file":      file_path.name,
                "folder":    str(file_path.parent),
                "hashed_at": datetime.now().isoformat()
            }
            save_hash_store(self.hash_store)
            logger.info(f"[NEW FILE] {file_path.name} — hash stored")
            logger.info(
                f"           Run build_pipeline.py to process and add to ChromaDB\n"
            )

if __name__ == "__main__":
    logger.info("  Multimodal RAG + GNN — File Watcher")
    observer = Observer()
    watcher  = DataFolderWatcher()
    watch_dirs = [PDF_DIR, PPT_DIR, VIDEO_DIR, AUDIO_DIR, IMAGE_DIR]
    logger.info("\nMonitoring folders:")
    for watch_dir in watch_dirs:
        watch_dir = Path(watch_dir)
        watch_dir.mkdir(parents=True, exist_ok=True)
        observer.schedule(watcher, str(watch_dir), recursive=False)
        logger.info(f"   {watch_dir}")
    observer.start()
    logger.info("")
    logger.success("File watcher is ACTIVE")
    logger.info("   File deleted  : ChromaDB + cache auto-cleaned")
    logger.info("   File replaced : hash checked - ChromaDB + cache cleaned if changed")
    logger.info("   File added    : hash stored - run build_pipeline.py to process")
    logger.info("")
    logger.info("Press Ctrl+C to stop\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logger.info("\nFile watcher stopped cleanly")
    observer.join()