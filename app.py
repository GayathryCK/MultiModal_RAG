import time
import streamlit as st
from pathlib import Path
from PIL import Image
import sys,os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.suppress_warnings import *
sys.path.insert(0, str(Path(__file__).parent))
from src.rag_retrieval  import retrieve
from src.llm_generation import answer_question, check_ollama_running, check_model_available
from config import LLM_MODEL, TOP_K

st.set_page_config(
    page_title = "MIT 6.036 ML Assistant",
    page_icon  = "🎓",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .source-card {
        background: #F0F4F8;
        border-left: 4px solid #0D9488;
        padding: 0.6rem 1rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
    }
    .score-badge {
        background: #0D9488;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .metric-card {
        background: #1E3A5F;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .answer-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.5rem;
        margin-top: 1rem;
        line-height: 1.7;
    }
    .stTextInput > div > div > input {
        border: 2px solid #0D9488 !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

SOURCE_ICONS = {
    "pdf"              : "📄",
    "ppt"              : "📊",
    "audio"            : "🎙️",
    "video_transcript" : "🎬",
    "video_keyframe"   : "🖼️",
    "image"            : "🖼️",
}
def get_icon(source_type: str) -> str:
    return SOURCE_ICONS.get(source_type, "📎")

def format_source_label(hit: dict) -> str:
    meta  = hit["metadata"]
    stype = meta.get("source_type", "unknown")
    icon  = get_icon(stype)
    if stype == "pdf":
        name = meta.get("doc_name", "?")
        return f"{icon} PDF: {name}"
    elif stype == "ppt":
        name  = meta.get("doc_name", "?")
        slide = meta.get("slide_title", f"slide {meta.get('slide_num','?')}")
        return f"{icon} PPT: {name} — {slide}"
    elif stype == "audio":
        name  = meta.get("audio_name", "?")
        start = meta.get("start_sec", 0)
        mins  = int(start // 60)
        secs  = int(start % 60)
        return f"{icon} Audio: {name[:40]} @ {mins}m{secs:02d}s"
    elif stype == "video_transcript":
        name  = meta.get("video_name", "?")
        start = meta.get("start_sec", 0)
        mins  = int(start // 60)
        secs  = int(start % 60)
        return f"{icon} Video: {name[:40]} @ {mins}m{secs:02d}s"
    elif stype == "video_keyframe":
        name = meta.get("video_name", "?")
        ts   = meta.get("timestamp_sec", 0)
        mins = int(ts // 60)
        secs = int(ts % 60)
        return f"{icon} Keyframe: {name[:40]} @ {mins}m{secs:02d}s"
    elif stype == "image":
        name = meta.get("image_name", "?")
        return f"{icon} Image: {name[:50]}"
    return f"{icon} {stype}: {meta.get('doc_name', '?')}"

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    top_k = st.slider(
        "Number of sources (Top-K)",
        min_value = 1,
        max_value = 10,
        value     = TOP_K,
        help      = "How many source chunks to retrieve"
    )
    use_gnn = st.toggle(
        "Use GNN Re-ranking",
        value = True,
        help  = "Enable graph-aware re-ranking of results"
    )
    stream_output = st.toggle(
        "Stream answer",
        value = True,
        help  = "Show answer token by token as it generates"
    )
    temperature = st.slider(
        "Temperature",
        min_value = 0.0,
        max_value = 1.0,
        value     = 0.3,
        step      = 0.1,
        help      = "Higher = more creative, Lower = more factual"
    )
    st.markdown("---")
    st.markdown("### 📊 System Status")
    if check_ollama_running():
        st.success(f"✅ Ollama running")
        if check_model_available(LLM_MODEL):
            st.success(f"✅ {LLM_MODEL} ready")
        else:
            st.error(f"❌ {LLM_MODEL} not found")
            st.code(f"ollama pull {LLM_MODEL}")
    else:
        st.error("❌ Ollama not running")
        st.code("ollama serve")
    st.markdown("---")
    st.markdown("### 💡 Sample Questions")
    sample_questions = [
        "What is gradient descent?",
        "Explain the perceptron algorithm",
        "How do CNNs work?",
        "What is regularization?",
        "Explain bias-variance tradeoff",
        "What is backpropagation?",
        "How does logistic regression work?",
        "What are decision trees?",
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True, key=f"sample_{q}"):
            st.session_state["prefill_query"] = q

    st.markdown("---")
    st.markdown("### 🗑️ Chat")
    if st.button("Clear History", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.last_result  = None
        st.rerun()

st.markdown('<div class="main-header">🎓 MIT 6.036 ML Assistant</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multimodal RAG + GNN — Ask anything about Machine Learning</div>',
            unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("📄 PDFs",    "23")
with col2:
    st.metric("📊 PPTs",    "14")
with col3:
    st.metric("🎬 Videos",  "14")
with col4:
    st.metric("🎙️ Audio",   "14")
with col5:
    st.metric("🗄️ Indexed", "7,669")

st.markdown("---")

prefill = st.session_state.pop("prefill_query", "")
query = st.text_input(
    "Ask a question about Machine Learning:",
    value       = prefill,
    placeholder = "e.g. What is gradient descent and how does it work?",
    key         = "query_input",
)
col_ask, col_clear = st.columns([1, 5])
with col_ask:
    ask_clicked = st.button("🔍 Ask", type="primary",
                             use_container_width=True)


if ask_clicked and query.strip():

    if not check_ollama_running():
        st.error("❌ Ollama is not running. Open a terminal and run: `ollama serve`")
        st.stop()
    with st.spinner("🔍 Searching course materials..."):
        retrieval = retrieve(query, top_k=top_k, use_gnn=use_gnn)

    col_answer, col_sources = st.columns([3, 2])
    with col_sources:
        st.markdown("#### 📚 Sources Retrieved")
        for i, hit in enumerate(retrieval["hits"]):
            label = format_source_label(hit)
            score = hit["final_score"]
            meta  = hit["metadata"]
            with st.expander(f"{label}", expanded=(i == 0)):
                st.markdown(
                    f'<span class="score-badge">score: {score:.3f}</span>',
                    unsafe_allow_html=True
                )
                st.markdown(f"**Vector:** {hit['vector_score']:.3f} | "
                           f"**GNN:** {hit['gnn_score']:.3f}")
                st.markdown("---")
                text = hit["text"]
                st.markdown(text[:500] + ("..." if len(text) > 500 else ""))
                if meta.get("source_type") == "video_keyframe":
                    frame_path = meta.get("frame_path", "")
                    if frame_path and Path(frame_path).exists():
                        try:
                            img = Image.open(frame_path)
                            st.image(img, caption=meta.get("caption", ""),
                                    use_column_width=True)
                        except Exception:
                            pass

                elif meta.get("source_type") == "image":
                    img_path = meta.get("source", "")
                    if img_path and Path(img_path).exists():
                        try:
                            img = Image.open(img_path)
                            st.image(img, caption=meta.get("caption", ""),
                                    use_column_width=True)
                        except Exception:
                            pass

    with col_answer:
        st.markdown("#### 💬 Answer")
        from src.llm_generation import build_prompt, generate, generate_streaming
        prompt     = build_prompt(query, retrieval["context"])
        start_time = time.time()
        if stream_output:
            answer_placeholder = st.empty()
            full_answer        = ""
            payload = {
                "model"  : LLM_MODEL,
                "prompt" : prompt,
                "stream" : True,
                "options": {
                    "temperature": temperature,
                    "num_predict": 1024,
                }
            }
            import requests, json
            from config import OLLAMA_URL
            from src.llm_generation import SYSTEM_PROMPT
            payload["system"] = SYSTEM_PROMPT
            try:
                with requests.post(OLLAMA_URL, json=payload,
                                   stream=True, timeout=120) as resp:
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        data        = json.loads(line.decode("utf-8"))
                        token       = data.get("response", "")
                        full_answer += token
                        answer_placeholder.markdown(
                            f'<div class="answer-box">{full_answer}▌</div>',
                            unsafe_allow_html=True
                        )
                        if data.get("done", False):
                            break
            except Exception as e:
                full_answer = f"Error: {e}"
            answer_placeholder.markdown(
                f'<div class="answer-box">{full_answer}</div>',
                unsafe_allow_html=True
            )
        else:
            with st.spinner("Generating answer..."):
                full_answer = generate(prompt, temperature=temperature)
            st.markdown(
                f'<div class="answer-box">{full_answer}</div>',
                unsafe_allow_html=True
            )
        latency = time.time() - start_time
        st.markdown(f"⏱ **{latency:.1f}s** | "
                   f"🔍 {len(retrieval['hits'])} sources | "
                   f"{'🧠 GNN' if use_gnn else '📐 Vector only'}")
    st.session_state.chat_history.append({
        "query"  : query,
        "answer" : full_answer,
        "hits"   : len(retrieval["hits"]),
        "latency": latency,
    })
    st.session_state.last_result = {
        "query"    : query,
        "answer"   : full_answer,
        "retrieval": retrieval,
        "latency"  : latency,
    }

if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("#### 🕑 Chat History")

    for i, entry in enumerate(reversed(
            st.session_state.chat_history[-5:])):  # show last 5
        with st.expander(f"Q: {entry['query'][:80]}", expanded=False):
            st.markdown(entry["answer"])
            st.caption(f"⏱ {entry['latency']:.1f}s | "
                      f"📚 {entry['hits']} sources")


st.markdown("---")
st.markdown(
    "<center><small>MultiModal RAG + GNN | MIT 6.036 | "
    "Built with LLaMA3 · ChromaDB · PyTorch Geometric · Streamlit</small></center>",
    unsafe_allow_html=True
)