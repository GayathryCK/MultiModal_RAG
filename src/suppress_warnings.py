import os
import warnings
import logging
os.environ["ANONYMIZED_TELEMETRY"]        = "False"
os.environ["CHROMA_TELEMETRY"]            = "False"
os.environ["TOKENIZERS_PARALLELISM"]      = "False"
os.environ["TF_CPP_MIN_LOG_LEVEL"]        = "3"
os.environ["OMP_NUM_THREADS"]             = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)
logging.getLogger("timm").setLevel(logging.ERROR)
logging.getLogger("open_clip").setLevel(logging.ERROR)
logging.getLogger("posthog").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)

logging.getLogger("posthog").setLevel(logging.CRITICAL)
logging.getLogger("posthog").disabled = True
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

try:
    import posthog
    posthog.disabled = True
    posthog.capture  = lambda *args, **kwargs: None
    posthog.identify = lambda *args, **kwargs: None
except Exception:
    pass

try:
    from chromadb.telemetry.product import posthog as chroma_posthog
    chroma_posthog.capture  = lambda *args, **kwargs: None
except Exception:
    pass

try:
    import chromadb.telemetry.product.posthog as cp
    cp.Posthog.capture = lambda *args, **kwargs: None
except Exception:
    pass