"""
app.py - Clinical Epilepsy Research Assistant (Interactive Streamlit Dashboard)
=============================================================================
Run using:
    streamlit run app.py
"""

import streamlit as st
import time
import os

# Page configuration
st.set_page_config(
    page_title="EpilepsyAI - Clinical RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Clinical Teal / Modern Dark & Light Accent)
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0E7490;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        background-color: #E0F2FE;
        color: #0369A1;
        margin-right: 0.5rem;
    }
    .source-card {
        border-left: 4px solid #0E7490;
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Initializing RAG Engine & Loading Neural Models (Ollama Qwen 2.5 + BGE Reranker)...")
def get_rag_engine():
    from rag_engine import EpilepsyRAGEngine
    return EpilepsyRAGEngine()


# Header
st.markdown('<div class="main-header">🧠 EpilepsyAI Clinical Research Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Advanced Retrieval-Augmented Generation with Multi-Query Expansion, BGE Cross-Encoder Reranking & Qwen 2.5:7B</div>',
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.header("⚙️ RAG Configuration")
    
    top_k = st.slider("Top Reranked Sources (K)", min_value=3, max_value=12, value=6, step=1)
    
    st.markdown("---")
    st.subheader("📚 Knowledge Base")
    st.info("📄 **Active Document:** `epilepsy.pdf` (MedComm 2026 Comprehensive Review)")
    
    st.markdown("---")
    st.subheader("🤖 Neural Models")
    st.markdown("""
    - **Generator:** `Qwen 2.5:7B (Ollama)`
    - **Reranker:** `BAAI/bge-reranker-base`
    - **Embedding:** `all-MiniLM-L6-v2`
    - **Vectorstore:** `FAISS (MMR Search)`
    """)
    
    st.markdown("---")
    if st.button("🔄 Rebuild FAISS Index", use_container_width=True):
        with st.spinner("Rebuilding FAISS vector database with optimal chunks..."):
            engine = get_rag_engine()
            engine.build_index(force_rebuild=True)
            st.success("FAISS Index rebuilt successfully!")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Quick example prompts
st.markdown("**💡 Example Clinical Questions / أسئلة إكلينيكية مقترحة:**")
cols_en = st.columns(3)
sample_q1 = "What are the three invasive neuromodulation options approved for adult drug-resistant epilepsy, and what are their typical responder rates?"
sample_q2 = "What are the efficacy rates and indications for Ketogenic Dietary Therapies (KD / MAD)?"
sample_q3 = "What is the clinical definition of Drug-Resistant Epilepsy (DRE) according to ILAE?"

clicked_q = None
if cols_en[0].button("⚡ Neuromodulation ", use_container_width=True):
    clicked_q = sample_q1
if cols_en[1].button("🥗 Ketogenic Diet ", use_container_width=True):
    clicked_q = sample_q2
if cols_en[2].button("📋 Definition of DRE ", use_container_width=True):
    clicked_q = sample_q3

cols_ar = st.columns(3)
sample_ar1 = "ما هي خيارات التحفيز العصبي الجراحية الثلاثة المعتمدة لعلاج الصرع المقاوم للأدوية وما هي نسب الاستجابة لكل منها؟"
sample_ar2 = "ما هي فعالية ونسب نجاح الحمية الكيتونية (Ketogenic Diet) في تقليل نوبات الصرع؟"
sample_ar3 = "ما هو التعريف الإكلينيكي للصرع المقاوم للأدوية (DRE) بحسب رابطة ILAE؟"

if cols_ar[0].button("⚡أجهزة التحفيز العصبي ", use_container_width=True):
    clicked_q = sample_ar1
if cols_ar[1].button("🥗 الحمية الكيتونية ", use_container_width=True):
    clicked_q = sample_ar2
if cols_ar[2].button("📋 تعريف الصرع المقاوم ", use_container_width=True):
    clicked_q = sample_ar3

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🩺" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])
        if "details" in msg:
            details = msg["details"]
            with st.expander("🔍 Clinical Evidence & Citations"):
                for src in details.get("sources", []):
                    st.markdown(f"""
                    **📄 Source {src['source_id']} (Page {src['page']})** — *Cross-Encoder Relevance Score:* `{src['score']:.4f}`
                    > {src['text']}
                    """)
            with st.expander("🧬 Multi-Query Expansion Breakdown"):
                st.write("**Generated Sub-Queries for Retrieval:**")
                for q in details.get("queries", []):
                    st.markdown(f"- `{q}`")

# Handle user input
user_input = st.chat_input("Ask any clinical question regarding epilepsy diagnostics, pharmacotherapy, or surgery...") or clicked_q

if user_input:
    # Append user question
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # Generate answer
    with st.chat_message("assistant", avatar="🩺"):
        engine = get_rag_engine()
        with st.status("Analyzing document, expanding queries, and reranking evidence...", expanded=False) as status:
            t0 = time.time()
            st.write("🔎 Step 1: Generating multi-query expansions with Qwen 2.5:7B...")
            queries = engine.expand_queries(user_input)
            
            st.write(f"📊 Step 2: MMR vector retrieval ({len(queries)} queries executed)...")
            st.write("⚖️ Step 3: Deep Cross-Encoder reranking (BAAI/bge-reranker-base)...")
            top_docs = engine.retrieve_and_rerank(user_input, queries, top_k=top_k)
            
            st.write("✍️ Step 4: Synthesizing medical response with Qwen 2.5:7B...")
            rag_output = engine.generate_answer(user_input, top_docs)
            rag_output["queries"] = queries
            latency = time.time() - t0
            status.update(label=f"Completed in {latency:.2f}s!", state="complete")

        st.markdown(rag_output["answer"])

        # Display sources
        with st.expander("🔍 Clinical Evidence & Citations", expanded=False):
            for src in rag_output.get("sources", []):
                st.markdown(f"""
                **📄 Source {src['source_id']} (Page {src['page']})** — *Cross-Encoder Relevance Score:* `{src['score']:.4f}`
                > {src['text']}
                """)
                
        with st.expander("🧬 Multi-Query Expansion Breakdown", expanded=False):
            st.write("**Generated Sub-Queries for Retrieval:**")
            for q in rag_output.get("queries", []):
                st.markdown(f"- `{q}`")

    # Append assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": rag_output["answer"],
        "details": rag_output
    })
