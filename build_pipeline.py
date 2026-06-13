import time
import json
from pathlib import Path
from loguru import logger
import sys
sys.path.insert(0, str(Path(__file__).parent))
from src.ingest_pdf          import ingest_all_pdfs
from src.ingest_ppt          import ingest_all_pptx
from src.ingest_image_audio  import ingest_all_images, ingest_all_audio
from src.ingest_video        import ingest_all_videos
from src.knowledge_graph     import build_knowledge_graph
from src.gnn_layer           import train_gnn
from config import GRAPH_DB_DIR

def run_step(name: str, fn, *args, **kwargs):
    print(f"\n{'='*60}")
    print(f"  STEP: {name}")
    print(f"{'='*60}")
    start = time.time()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.time() - start
        logger.success(f"{name} completed in {elapsed:.1f}s")
        return result, elapsed, None
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"{name} FAILED after {elapsed:.1f}s: {e}")
        return None, elapsed, str(e)

def run_full_pipeline(
    skip_pdfs   : bool = False,
    skip_ppts   : bool = False,
    skip_images : bool = False,
    skip_audio  : bool = False,
    skip_videos : bool = False,
    skip_graph  : bool = False,
    skip_gnn    : bool = False,
    gnn_model   : str  = "gcn",
) -> dict:
    pipeline_start = time.time()
    summary        = {}
    logger.info("Starting MultiModal RAG Pipeline...")
    logger.info(f"  GNN model: {gnn_model}")
    if not skip_pdfs:
        results, elapsed, error = run_step("PDF Ingestion", ingest_all_pdfs)
        summary["pdfs"] = {
            "count"  : len(results) if results else 0,
            "elapsed": elapsed,
            "error"  : error,
        }
    else:
        logger.info("Skipping PDF ingestion")
        summary["pdfs"] = {"skipped": True}
    if not skip_ppts:
        results, elapsed, error = run_step("PPTX Ingestion", ingest_all_pptx)
        summary["ppts"] = {
            "count"  : len(results) if results else 0,
            "elapsed": elapsed,
            "error"  : error,
        }
    else:
        logger.info("Skipping PPTX ingestion")
        summary["ppts"] = {"skipped": True}
    if not skip_images:
        results, elapsed, error = run_step("Image Ingestion", ingest_all_images)
        summary["images"] = {
            "count"  : len(results) if results else 0,
            "elapsed": elapsed,
            "error"  : error,
        }
    else:
        logger.info("Skipping image ingestion")
        summary["images"] = {"skipped": True}
    if not skip_audio:
        results, elapsed, error = run_step("Audio Ingestion", ingest_all_audio)
        summary["audio"] = {
            "count"  : len(results) if results else 0,
            "elapsed": elapsed,
            "error"  : error,
        }
    else:
        logger.info("Skipping audio ingestion")
        summary["audio"] = {"skipped": True}
    if not skip_videos:
        results, elapsed, error = run_step("Video Ingestion", ingest_all_videos)
        summary["videos"] = {
            "count"  : len(results) if results else 0,
            "elapsed": elapsed,
            "error"  : error,
        }
    else:
        logger.info("Skipping video ingestion")
        summary["videos"] = {"skipped": True}
    if not skip_graph:
        G, elapsed, error = run_step(
            "Knowledge Graph", build_knowledge_graph
        )
        summary["graph"] = {
            "nodes"  : G.number_of_nodes() if G else 0,
            "edges"  : G.number_of_edges() if G else 0,
            "elapsed": elapsed,
            "error"  : error,
        }
    else:
        logger.info("Skipping knowledge graph")
        summary["graph"] = {"skipped": True}
    if not skip_gnn:
        embeddings, elapsed, error = run_step(
            "GNN Training", train_gnn, model_type=gnn_model
        )
        summary["gnn"] = {
            "shape"  : list(embeddings.shape) if embeddings is not None else None,
            "elapsed": elapsed,
            "error"  : error,
        }
    else:
        logger.info("Skipping GNN training")
        summary["gnn"] = {"skipped": True}
    total_elapsed = time.time() - pipeline_start
    summary["total_elapsed"] = total_elapsed
    summary_path = GRAPH_DB_DIR / "pipeline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.success(f"Pipeline summary saved → {summary_path}")
    return summary

def print_summary(summary: dict):
    print("  PIPELINE COMPLETE — SUMMARY")
    steps = [
        ("PDFs",           summary.get("pdfs",   {})),
        ("PPTXs",          summary.get("ppts",   {})),
        ("Images",         summary.get("images", {})),
        ("Audio",          summary.get("audio",  {})),
        ("Videos",         summary.get("videos", {})),
        ("Knowledge Graph",summary.get("graph",  {})),
        ("GNN Training",   summary.get("gnn",    {})),
    ]
    for name, data in steps:
        if data.get("skipped"):
            status = "⏭  SKIPPED"
            detail = ""
        elif data.get("error"):
            status = " FAILED"
            detail = f"  → {data['error'][:60]}"
        else:
            status = " DONE"
            elapsed = data.get("elapsed", 0)
            if "count" in data:
                detail = f"  → {data['count']} items in {elapsed:.1f}s"
            elif "nodes" in data:
                detail = f"  → {data['nodes']} nodes, {data['edges']} edges in {elapsed:.1f}s"
            elif "shape" in data:
                detail = f"  → embeddings {data['shape']} in {elapsed:.1f}s"
            else:
                detail = f"  → {elapsed:.1f}s"
        print(f"  {status}  {name:20s}{detail}")
    total = summary.get("total_elapsed", 0)
    mins  = int(total // 60)
    secs  = int(total % 60)
    print(f"\n  Total time: {mins}m {secs}s")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Build MultiModal RAG Pipeline"
    )
    parser.add_argument("--skip-pdfs",   action="store_true")
    parser.add_argument("--skip-ppts",   action="store_true")
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--skip-audio",  action="store_true")
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--skip-graph",  action="store_true")
    parser.add_argument("--skip-gnn",    action="store_true")
    parser.add_argument("--gnn-model",   default="gcn",
                        choices=["gcn", "gat", "sage"])

    args = parser.parse_args()
    summary = run_full_pipeline(
        skip_pdfs   = args.skip_pdfs,
        skip_ppts   = args.skip_ppts,
        skip_images = args.skip_images,
        skip_audio  = args.skip_audio,
        skip_videos = args.skip_videos,
        skip_graph  = args.skip_graph,
        skip_gnn    = args.skip_gnn,
        gnn_model   = args.gnn_model,
    )
    print_summary(summary)
