import json
import requests
import time
from pathlib import Path
from loguru import logger
import sys,os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.suppress_warnings import *
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OLLAMA_URL, LLM_MODEL
from src.rag_retrieval import retrieve

SYSTEM_PROMPT = """You are an expert Machine Learning teaching assistant for MIT 6.036.
You have access to lecture notes, slides, video transcripts, and diagrams from the course.

Your job is to answer student questions clearly and accurately using ONLY the provided context.

Rules:
- Base your answer strictly on the provided context
- If the context does not contain enough information, say so honestly
- Use clear explanations with examples where possible
- Cite sources using the labels provided (e.g. [PDF: ...], [Video: ...])
- Keep answers concise but complete
- Use bullet points or numbered steps for complex explanations
"""

def build_prompt(query: str, context: str) -> str:
    return f"""### Context from course materials:
{context}
### Student Question:
{query}
### Answer:"""

def check_ollama_running() -> bool:
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False

def check_model_available(model: str = LLM_MODEL) -> bool:
    try:
        resp   = requests.get("http://localhost:11434/api/tags", timeout=3)
        models = [m["name"] for m in resp.json().get("models", [])]
        return any(model in m for m in models)
    except Exception:
        return False

def generate_streaming(prompt: str,
                        model: str = LLM_MODEL,
                        temperature: float = 0.3,
                        max_tokens: int = 1024) -> str:
    payload = {
        "model"  : model,
        "prompt" : prompt,
        "system" : SYSTEM_PROMPT,
        "stream" : True,
        "options": {
            "temperature"  : temperature,
            "num_predict"  : max_tokens,
            "top_p"        : 0.9,
            "repeat_penalty": 1.1,
        }
    }
    full_response = ""
    try:
        with requests.post(OLLAMA_URL, json=payload,
                           stream=True, timeout=120) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                data  = json.loads(line.decode("utf-8"))
                token = data.get("response", "")
                full_response += token
                print(token, end="", flush=True)   # stream to console
                if data.get("done", False):
                    break
    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to Ollama. "
                     "Make sure Ollama is running: ollama serve")
        return "Error: Ollama is not running."
    except requests.exceptions.Timeout:
        logger.error("Ollama request timed out.")
        return "Error: Request timed out."
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        return f"Error: {e}"
    print()
    return full_response

def generate(prompt: str,
             model: str = LLM_MODEL,
             temperature: float = 0.3,
             max_tokens: int = 1024) -> str:
    payload = {
        "model"  : model,
        "prompt" : prompt,
        "system" : SYSTEM_PROMPT,
        "stream" : False,
        "options": {
            "temperature"  : temperature,
            "num_predict"  : max_tokens,
            "top_p"        : 0.9,
            "repeat_penalty": 1.1,
        }
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to Ollama. Run: ollama serve")
        return "Error: Ollama is not running."
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        return f"Error: {e}"

def answer_question(query: str,
                    top_k: int = 5,
                    stream: bool = True,
                    use_gnn: bool = True) -> dict:
    start_time = time.time()
    logger.info(f"Retrieving context for: {query[:60]}")
    retrieval  = retrieve(query, top_k=top_k, use_gnn=use_gnn)
    context    = retrieval["context"]
    hits       = retrieval["hits"]
    if not context.strip():
        return {
            "query"  : query,
            "answer" : "I could not find relevant information in the course materials.",
            "context": "",
            "hits"   : [],
            "latency": time.time() - start_time,
        }
    prompt = build_prompt(query, context)
    logger.info(f"Prompt length: {len(prompt)} chars")
    logger.info("Generating answer...")
    if stream:
        answer = generate_streaming(prompt)
    else:
        answer = generate(prompt)
    latency = time.time() - start_time
    logger.success(f"Answer generated in {latency:.2f}s")
    return {
        "query"  : query,
        "answer" : answer,
        "context": context,
        "hits"   : hits,
        "latency": latency,
    }

def answer_question_no_context(query: str) -> str:
    prompt = f"Answer this machine learning question: {query}"
    return generate(prompt)

if __name__ == "__main__":
    print("Checking Ollama...")
    if not check_ollama_running():
        print(" Ollama is NOT running!")
        print("   Open a new terminal and run: ollama serve")
        print("   Then run this file again.")
        exit(1)
    if not check_model_available(LLM_MODEL):
        print(f" Model '{LLM_MODEL}' not found!")
        print(f"   Run: ollama pull {LLM_MODEL}")
        exit(1)
    print(f" Ollama running | Model: {LLM_MODEL}")
    print()
    test_questions = [
        "What is gradient descent and how does it work?",
        "Explain the perceptron learning algorithm with an example.",
    ]
    for question in test_questions:
        print(f"Question: {question}")
        result = answer_question(question, top_k=5, stream=True)
        print(f"\n Sources Used ")
        for i, hit in enumerate(result["hits"]):
            meta  = hit["metadata"]
            stype = meta.get("source_type", "?")
            name  = meta.get("doc_name",
                    meta.get("audio_name",
                    meta.get("video_name", "?")))
            print(f"  [{i+1}] {stype:20s} | "
                  f"score={hit['final_score']:.3f} | {name}")
        print(f"\n  Latency: {result['latency']:.2f}s")