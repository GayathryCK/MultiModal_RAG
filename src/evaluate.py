import json
import time
import math
import numpy as np
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
from rank_bm25 import BM25Okapi
import sys,os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.suppress_warnings import *
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.rag_retrieval  import retrieve, vector_search
from src.llm_generation import answer_question, generate
from config import TOP_K, GRAPH_DB_DIR

TEST_QUESTIONS = [
    {
        "question": "What is gradient descent and how does it work?",
        "reference_answer": (
            "Gradient descent is an iterative optimization algorithm used to "
            "minimize a loss function by updating model parameters in the "
            "direction opposite to the gradient. At each step, the parameters "
            "are updated by subtracting the gradient multiplied by the learning "
            "rate. The process repeats until convergence."
        ),
        "relevant_sources": ["pdf", "ppt", "audio", "video_transcript"],
        "relevant_keywords": [
            "gradient", "descent", "optimization", "learning rate",
            "loss function", "parameters", "update"
        ],
    },
    {
        "question": "Explain the perceptron learning algorithm.",
        "reference_answer": (
            "The perceptron is a linear binary classifier. It initializes "
            "weights to zero, then iterates through training examples. For each "
            "misclassified example, it updates the weights by adding the product "
            "of the learning rate, the true label, and the input features. "
            "The algorithm converges if the data is linearly separable."
        ),
        "relevant_sources": ["pdf", "ppt", "audio", "video_transcript"],
        "relevant_keywords": [
            "perceptron", "weights", "linear", "classifier",
            "misclassified", "update", "convergence"
        ],
    },
    {
        "question": "How do convolutional neural networks work?",
        "reference_answer": (
            "Convolutional neural networks use convolutional layers to extract "
            "spatial features from input data. Each layer applies learned filters "
            "to produce feature maps. Pooling layers reduce spatial dimensions. "
            "CNNs are especially effective for image recognition tasks due to "
            "their ability to capture local patterns and hierarchical features."
        ),
        "relevant_sources": ["pdf", "ppt", "audio", "video_transcript"],
        "relevant_keywords": [
            "convolutional", "filters", "feature maps", "pooling",
            "image", "spatial", "layers"
        ],
    },
    {
        "question": "What is the difference between supervised and unsupervised learning?",
        "reference_answer": (
            "Supervised learning uses labeled training data where each example "
            "has an associated target output. The model learns to map inputs to "
            "outputs. Unsupervised learning finds patterns in unlabeled data "
            "without predefined targets, using techniques like clustering and "
            "dimensionality reduction."
        ),
        "relevant_sources": ["pdf", "ppt", "audio"],
        "relevant_keywords": [
            "supervised", "unsupervised", "labeled", "unlabeled",
            "clustering", "classification", "targets"
        ],
    },
    {
        "question": "What is regularization and why is it important?",
        "reference_answer": (
            "Regularization is a technique to prevent overfitting by adding a "
            "penalty term to the loss function. L1 regularization adds the sum "
            "of absolute weights, promoting sparsity. L2 regularization adds the "
            "sum of squared weights, shrinking all weights. Regularization "
            "improves model generalization to unseen data."
        ),
        "relevant_sources": ["pdf", "ppt", "audio"],
        "relevant_keywords": [
            "regularization", "overfitting", "L1", "L2",
            "penalty", "generalization", "weights"
        ],
    },
    {
        "question": "Explain backpropagation in neural networks.",
        "reference_answer": (
            "Backpropagation computes gradients of the loss function with respect "
            "to network weights using the chain rule of calculus. Starting from "
            "the output layer, gradients are propagated backward through the "
            "network. These gradients are then used by gradient descent to update "
            "the weights and minimize the loss."
        ),
        "relevant_sources": ["pdf", "ppt", "audio", "video_transcript"],
        "relevant_keywords": [
            "backpropagation", "chain rule", "gradients",
            "weights", "loss", "neural network", "layers"
        ],
    },
    {
        "question": "What is logistic regression used for?",
        "reference_answer": (
            "Logistic regression is used for binary and multiclass classification "
            "problems. It applies the sigmoid function to a linear combination of "
            "features to output a probability between 0 and 1. The decision "
            "boundary separates classes. It is trained using maximum likelihood "
            "estimation and cross-entropy loss."
        ),
        "relevant_sources": ["pdf", "ppt", "audio"],
        "relevant_keywords": [
            "logistic regression", "sigmoid", "classification",
            "probability", "binary", "cross-entropy", "decision boundary"
        ],
    },
    {
        "question": "How does the k-nearest neighbors algorithm work?",
        "reference_answer": (
            "K-nearest neighbors classifies a new point by finding the K closest "
            "training examples using a distance metric like Euclidean distance. "
            "The majority class among the K neighbors determines the prediction. "
            "KNN is a non-parametric lazy learner that requires no training phase "
            "but is slow at prediction time."
        ),
        "relevant_sources": ["pdf", "ppt"],
        "relevant_keywords": [
            "k-nearest", "neighbors", "distance", "euclidean",
            "classification", "non-parametric", "lazy"
        ],
    },
    {
        "question": "What is the bias-variance tradeoff?",
        "reference_answer": (
            "The bias-variance tradeoff describes the tension between underfitting "
            "and overfitting. High bias means the model is too simple and misses "
            "patterns. High variance means the model is too complex and fits "
            "noise. The total error is bias squared plus variance plus irreducible "
            "noise. Regularization and ensemble methods help balance this tradeoff."
        ),
        "relevant_sources": ["pdf", "ppt", "audio"],
        "relevant_keywords": [
            "bias", "variance", "overfitting", "underfitting",
            "tradeoff", "generalization", "error"
        ],
    },
    {
        "question": "What are recurrent neural networks used for?",
        "reference_answer": (
            "Recurrent neural networks process sequential data by maintaining a "
            "hidden state that captures information from previous time steps. "
            "They are used for tasks like language modeling, speech recognition, "
            "and time series prediction. LSTMs and GRUs are popular variants that "
            "address the vanishing gradient problem."
        ),
        "relevant_sources": ["pdf", "ppt", "audio", "video_transcript"],
        "relevant_keywords": [
            "recurrent", "RNN", "sequential", "hidden state",
            "LSTM", "GRU", "vanishing gradient", "time series"
        ],
    },
]

def recall_at_k(hits: list[dict], relevant_sources: list[str],
                k: int) -> float:
    top_k_types = set(
        h["metadata"].get("source_type", "")
        for h in hits[:k]
    )
    relevant_set = set(relevant_sources)
    found        = len(top_k_types & relevant_set)
    return found / len(relevant_set) if relevant_set else 0.0

def mean_reciprocal_rank(hits: list[dict],
                          relevant_sources: list[str]) -> float:
    for i, hit in enumerate(hits, start=1):
        stype = hit["metadata"].get("source_type", "")
        if stype in relevant_sources:
            return 1.0 / i
    return 0.0

def ndcg_at_k(hits: list[dict], relevant_sources: list[str],
               k: int) -> float:
    def dcg(scores):
        return sum(
            s / math.log2(i + 2)
            for i, s in enumerate(scores)
        )
    actual_scores = [
        1 if hits[i]["metadata"].get("source_type", "") in relevant_sources
        else 0
        for i in range(min(k, len(hits)))
    ]
    ideal_scores = sorted(actual_scores, reverse=True)
    dcg_val  = dcg(actual_scores)
    idcg_val = dcg(ideal_scores)
    return dcg_val / idcg_val if idcg_val > 0 else 0.0

def keyword_recall(text: str, keywords: list[str]) -> float:
    text_lower = text.lower()
    found = sum(1 for kw in keywords if kw.lower() in text_lower)
    return found / len(keywords) if keywords else 0.0

def compute_bleu(hypothesis: str, reference: str) -> float:
    try:
        hyp_tokens = hypothesis.lower().split()
        ref_tokens = reference.lower().split()
        smoother   = SmoothingFunction().method1
        return sentence_bleu(
            [ref_tokens], hyp_tokens,
            smoothing_function=smoother
        )
    except Exception:
        return 0.0

def compute_rouge_l(hypothesis: str, reference: str) -> float:
    try:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        result = scorer.score(reference, hypothesis)
        return result["rougeL"].fmeasure
    except Exception:
        return 0.0

def compute_faithfulness(answer: str, context: str) -> float:
    try:
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were",
            "in", "on", "at", "to", "for", "of", "and",
            "or", "but", "it", "its", "this", "that", "be",
            "by", "with", "as", "from", "have", "has", "had"
        }
        answer_words  = [
            w.lower().strip(".,!?;:")
            for w in answer.split()
            if w.lower() not in stop_words and len(w) > 3
        ]
        context_lower = context.lower()
        found         = sum(1 for w in answer_words if w in context_lower)
        return found / len(answer_words) if answer_words else 0.0
    except Exception:
        return 0.0

def build_bm25_index() -> tuple:
    import chromadb
    from config import VECTOR_DB_DIR, CHROMA_COLLECTION
    client     = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    collection = client.get_or_create_collection(CHROMA_COLLECTION)
    total  = collection.count()
    limit  = 1000
    docs, metadatas, ids = [], [], []
    for offset in range(0, total, limit):
        batch = collection.get(
            limit=limit, offset=offset,
            include=["documents", "metadatas"]
        )
        docs.extend(batch["documents"])
        metadatas.extend(batch["metadatas"])
        ids.extend(batch["ids"])
    tokenized = [d.lower().split() for d in docs]
    bm25      = BM25Okapi(tokenized)
    logger.info(f"BM25 index built: {len(docs)} documents")
    return bm25, docs, metadatas, ids

def bm25_search(query: str, bm25, docs: list,
                metadatas: list, ids: list,
                top_k: int = 5) -> list[dict]:
    tokens  = query.lower().split()
    scores  = bm25.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [{
        "id"          : ids[i],
        "text"        : docs[i],
        "metadata"    : metadatas[i],
        "vector_score": float(scores[i]),
        "gnn_score"   : 0.0,
        "final_score" : float(scores[i]),
    } for i in top_idx]

def evaluate_system(
    system_name : str,
    search_fn,
    generate_fn,
    test_questions: list,
    top_k: int = TOP_K,
    compute_bert: bool = True,
) -> dict:
    logger.info(f"\nEvaluating: {system_name}")
    all_metrics = []
    for item in tqdm(test_questions,
                     desc=f"Evaluating {system_name}"):
        query      = item["question"]
        reference  = item["reference_answer"]
        rel_src    = item["relevant_sources"]
        keywords   = item["relevant_keywords"]
        t0   = time.time()
        hits = search_fn(query, top_k)
        retrieval_latency = time.time() - t0
        t0     = time.time()
        answer = generate_fn(query, hits)
        generation_latency = time.time() - t0
        total_latency = retrieval_latency + generation_latency
        r1    = recall_at_k(hits, rel_src, k=1)
        r3    = recall_at_k(hits, rel_src, k=3)
        r5    = recall_at_k(hits, rel_src, k=5)
        mrr   = mean_reciprocal_rank(hits, rel_src)
        ndcg3 = ndcg_at_k(hits, rel_src, k=3)
        ndcg5 = ndcg_at_k(hits, rel_src, k=5)
        bleu        = compute_bleu(answer, reference)
        rouge_l     = compute_rouge_l(answer, reference)
        faith       = compute_faithfulness(
                          answer,
                          "\n".join(h["text"] for h in hits)
                      )
        kw_recall   = keyword_recall(answer, keywords)

        metrics = {
            "question"           : query,
            "recall_at_1"        : r1,
            "recall_at_3"        : r3,
            "recall_at_5"        : r5,
            "mrr"                : mrr,
            "ndcg_at_3"          : ndcg3,
            "ndcg_at_5"          : ndcg5,
            "bleu"               : bleu,
            "rouge_l"            : rouge_l,
            "faithfulness"       : faith,
            "keyword_recall"     : kw_recall,
            "retrieval_latency"  : retrieval_latency,
            "generation_latency" : generation_latency,
            "total_latency"      : total_latency,
            "answer"             : answer,
        }
        all_metrics.append(metrics)
    if compute_bert:
        logger.info("Computing BERTScore...")
        hypotheses = [m["answer"]    for m in all_metrics]
        references = [item["reference_answer"]
                      for item in test_questions]
        try:
            P, R, F1 = bert_score_fn(
                hypotheses, references,
                lang="en", verbose=False
            )
            for i, m in enumerate(all_metrics):
                m["bert_score_f1"] = float(F1[i])
        except Exception as e:
            logger.warning(f"BERTScore failed: {e}")
            for m in all_metrics:
                m["bert_score_f1"] = 0.0
    else:
        for m in all_metrics:
            m["bert_score_f1"] = 0.0

    def mean(key):
        vals = [m[key] for m in all_metrics]
        return float(np.mean(vals))

    aggregated = {
        "system"                  : system_name,
        "num_questions"           : len(all_metrics),
        "avg_recall_at_1"         : mean("recall_at_1"),
        "avg_recall_at_3"         : mean("recall_at_3"),
        "avg_recall_at_5"         : mean("recall_at_5"),
        "avg_mrr"                 : mean("mrr"),
        "avg_ndcg_at_3"           : mean("ndcg_at_3"),
        "avg_ndcg_at_5"           : mean("ndcg_at_5"),
        "avg_bleu"                : mean("bleu"),
        "avg_rouge_l"             : mean("rouge_l"),
        "avg_bert_score_f1"       : mean("bert_score_f1"),
        "avg_faithfulness"        : mean("faithfulness"),
        "avg_keyword_recall"      : mean("keyword_recall"),
        "avg_retrieval_latency"   : mean("retrieval_latency"),
        "avg_generation_latency"  : mean("generation_latency"),
        "avg_total_latency"       : mean("total_latency"),
        "per_question"            : all_metrics,
    }
    return aggregated

def run_full_evaluation(
    test_questions : list = None,
    top_k          : int  = TOP_K,
    run_bm25       : bool = True,
    run_vector_only: bool = True,
    run_full_system: bool = True,
    compute_bert   : bool = True,
) -> dict:
    if test_questions is None:
        test_questions = TEST_QUESTIONS
    results = {}
    if run_bm25:
        logger.info("Building BM25 index...")
        bm25, docs, metadatas, ids = build_bm25_index()
        def bm25_search_fn(query, top_k):
            return bm25_search(query, bm25, docs,
                               metadatas, ids, top_k)
        def bm25_generate_fn(query, hits):
            context = "\n".join(h["text"] for h in hits[:3])
            prompt  = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
            return generate(prompt)
        results["bm25"] = evaluate_system(
            "BM25 Baseline",
            bm25_search_fn,
            bm25_generate_fn,
            test_questions,
            top_k        = top_k,
            compute_bert = compute_bert,
        )
    if run_vector_only:
        def vector_only_search_fn(query, top_k):
            return vector_search(query, top_k=top_k)
        def vector_generate_fn(query, hits):
            from src.rag_retrieval  import assemble_context
            from src.llm_generation import build_prompt
            context = assemble_context(hits, top_k=top_k)
            prompt  = build_prompt(query, context)
            return generate(prompt)
        results["vector_only"] = evaluate_system(
            "Text-only RAG (no GNN)",
            vector_only_search_fn,
            vector_generate_fn,
            test_questions,
            top_k        = top_k,
            compute_bert = compute_bert,
        )

    if run_full_system:
        def full_search_fn(query, top_k):
            result = retrieve(query, top_k=top_k, use_gnn=True)
            return result["hits"]
        def full_generate_fn(query, hits):
            from src.rag_retrieval  import assemble_context
            from src.llm_generation import build_prompt
            context = assemble_context(hits, top_k=top_k)
            prompt  = build_prompt(query, context)
            return generate(prompt)
        results["full_system"] = evaluate_system(
            "Multimodal RAG + GNN",
            full_search_fn,
            full_generate_fn,
            test_questions,
            top_k        = top_k,
            compute_bert = compute_bert,
        )
    return results

def print_results(results: dict):
    systems = list(results.keys())
    metrics = [
        ("Recall@1",       "avg_recall_at_1"),
        ("Recall@3",       "avg_recall_at_3"),
        ("Recall@5",       "avg_recall_at_5"),
        ("MRR",            "avg_mrr"),
        ("NDCG@3",         "avg_ndcg_at_3"),
        ("NDCG@5",         "avg_ndcg_at_5"),
        ("BLEU",           "avg_bleu"),
        ("ROUGE-L",        "avg_rouge_l"),
        ("BERTScore F1",   "avg_bert_score_f1"),
        ("Faithfulness",   "avg_faithfulness"),
        ("Keyword Recall", "avg_keyword_recall"),
        ("Latency (s)",    "avg_total_latency"),
    ]
    print(f"  EVALUATION RESULTS — MultiModal RAG + GNN")
    print(f"  {'Metric':<20}", end="")
    for s in systems:
        name = results[s]["system"][:18]
        print(f"  {name:>18}", end="")
    print()
    for label, key in metrics:
        print(f"  {label:<20}", end="")
        row_vals = [results[s][key] for s in systems]
        best_val = max(row_vals) if "Latency" not in label \
                   else min(row_vals)

        for i, (s, val) in enumerate(zip(systems, row_vals)):
            marker = " ←best" if val == best_val else ""
            print(f"  {val:>12.4f}{marker:>6}", end="")
        print()

    print(f"\n  SUCCESS THRESHOLDS CHECK (Full System):")
    if "full_system" in results:
        fs = results["full_system"]
        checks = [
            ("Recall@5 > 0.80",    fs["avg_recall_at_5"]     > 0.80),
            ("MRR > 0.70",         fs["avg_mrr"]              > 0.70),
            ("BERTScore F1 > 0.85",fs["avg_bert_score_f1"]   > 0.85),
            ("Faithfulness > 0.85",fs["avg_faithfulness"]     > 0.85),
            ("Latency < 60s",      fs["avg_total_latency"]    < 60.0),
        ]
        for label, passed in checks:
            icon = "correct" if passed else "wrong"
            print(f"    {icon} {label}")

def save_results(results: dict):
    summary = {}
    for system, data in results.items():
        summary[system] = {
            k: v for k, v in data.items()
            if k != "per_question"
        }
    summary_path = GRAPH_DB_DIR / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.success(f"Saved evaluation summary → {summary_path}")
    full_path = GRAPH_DB_DIR / "evaluation_full.json"
    with open(full_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.success(f"Saved full results → {full_path}")
    return summary_path

if __name__ == "__main__":
    print("  MultiModal RAG + GNN — Evaluation Suite")
    print(f"  Test questions : {len(TEST_QUESTIONS)}")
    print(f"  Top-K          : {TOP_K}")
    print()
    results = run_full_evaluation(
        test_questions  = TEST_QUESTIONS,
        top_k           = TOP_K,
        run_bm25        = True,
        run_vector_only = True,
        run_full_system = True,
        compute_bert    = True,
    )
    print_results(results)
    summary_path = save_results(results)
    print(f"  Results saved to: {summary_path}")
    print("\nEvaluation complete!")