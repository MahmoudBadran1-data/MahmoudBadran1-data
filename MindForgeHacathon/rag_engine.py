"""
rag_engine.py - Advanced Medical RAG Engine for Epilepsy Research
================================================================
Pipeline:
  1. PDF Parsing & Structure-Aware Chunking (PyPDFLoader + RecursiveCharacterTextSplitter)
  2. Dense Vector Indexing (FAISS + all-MiniLM-L6-v2)
  3. Multi-Query Reformulation (Qwen 2.5:7B via Ollama)
  4. Diverse MMR Retrieval
  5. Cross-Encoder Deep Reranking (BAAI/bge-reranker-base)
  6. Evidence-Grounded Medical Synthesis (Qwen 2.5:7B)
"""

import os
import re
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

# HuggingFace Cache setup
os.environ["HF_HOME"] = r"F:\HuggingFace"
os.environ["HF_HUB_CACHE"] = r"F:\HuggingFace\hub"
os.environ["TRANSFORMERS_CACHE"] = r"F:\HuggingFace\transformers"

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from langchain_ollama import ChatOllama


class EpilepsyRAGEngine:
    def __init__(
        self,
        pdf_path: str = r"F:\Lectures\AI Hacathon\epilepsy.pdf",
        faiss_dir: str = r"F:\Lectures\AI Hacathon\FAISS_DB_V2",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model_name: str = "BAAI/bge-reranker-base",
        llm_model_name: str = "qwen2.5:7b",
        chunk_size: int = 1200,
        chunk_overlap: int = 250,
    ):
        self.pdf_path = pdf_path
        self.faiss_dir = faiss_dir
        self.embedding_model_name = embedding_model_name
        self.reranker_model_name = reranker_model_name
        self.llm_model_name = llm_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize Embedding Model
        print(f"[1/4] Loading Embedding Model: {embedding_model_name}...")
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

        # Initialize Cross-Encoder Reranker
        print(f"[2/4] Loading Cross-Encoder Reranker: {reranker_model_name}...")
        self.reranker = CrossEncoder(self.reranker_model_name)

        # Initialize LLM
        print(f"[3/4] Initializing Ollama LLM: {llm_model_name}...")
        self.llm = ChatOllama(model=self.llm_model_name, temperature=0.1)

        # Build or Load Vectorstore
        print(f"[4/4] Setting up FAISS Vector Database...")
        self.vectorstore = self._get_or_create_vectorstore()
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8, "fetch_k": 30, "lambda_mult": 0.6}
        )
        print(">>> Medical RAG Engine is fully ready!\n")

    def _get_or_create_vectorstore(self) -> FAISS:
        """Loads existing FAISS index or creates a fresh one with optimal chunking."""
        if os.path.exists(self.faiss_dir) and os.path.exists(os.path.join(self.faiss_dir, "index.faiss")):
            print(f"  -> Found existing FAISS index at '{self.faiss_dir}'. Loading...")
            try:
                return FAISS.load_local(
                    self.faiss_dir,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"  -> Warning: Could not load index ({e}). Rebuilding...")

        return self.build_index(force_rebuild=True)

    def build_index(self, force_rebuild: bool = False) -> FAISS:
        """Parses the PDF and builds a high-quality FAISS index."""
        print(f"  -> Parsing PDF from '{self.pdf_path}' with PyPDFLoader...")
        loader = PyPDFLoader(self.pdf_path)
        raw_docs = loader.load()
        print(f"  -> Loaded {len(raw_docs)} pages.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(raw_docs)
        print(f"  -> Created {len(chunks)} text chunks with chunk_size={self.chunk_size}.")

        print(f"  -> Creating FAISS embeddings and building vector index...")
        vs = FAISS.from_documents(documents=chunks, embedding=self.embeddings)
        os.makedirs(self.faiss_dir, exist_ok=True)
        vs.save_local(self.faiss_dir)
        print(f"  -> FAISS index saved successfully at '{self.faiss_dir}'.")
        return vs

    def expand_queries(self, query: str) -> List[str]:
        """Generates 3 focused medical search sub-queries in English using Qwen 2.5:7B."""
        expansion_prompt = f"""You are a specialized medical query expansion assistant.
Given the clinical research question below, generate exactly 3 short, specific search queries in ENGLISH covering different aspects, clinical terms, synonyms, and specific therapies mentioned.

Note: If the user question is in Arabic or any language other than English, translate its meaning and output all 3 search queries in ENGLISH (because the medical documents in the database are in English).

User Question: {query}

Instructions:
- Provide exactly 3 search queries in English, one per line.
- Do NOT output preamble, numbering, bullets, or explanations.
- Keep each query direct and focused on medical entities, device names, or outcome measures.
"""
        try:
            response = self.llm.invoke(expansion_prompt)
            lines = response.content.strip().split("\n")
            cleaned_queries = []
            for line in lines:
                cleaned = re.sub(r"^[\d\.\-\*\)\s]+", "", line.strip())
                if cleaned and len(cleaned) > 5 and not cleaned.lower().startswith("here are"):
                    cleaned_queries.append(cleaned)
            queries = [query] + cleaned_queries[:3]
        except Exception as e:
            print(f"Query expansion fallback: {e}")
            queries = [query]

        return queries

    def retrieve_and_rerank(
        self,
        query: str,
        retrieval_queries: List[str],
        top_k: int = 8
    ) -> List[Tuple[Any, float]]:
        """Retrieves documents across all expanded queries and applies Cross-Encoder reranking."""
        # 1. Multi-Query Retrieval
        candidate_docs = []
        for q in retrieval_queries:
            docs = self.retriever.invoke(q)
            candidate_docs.extend(docs)

        # 2. Global Deduplication by exact text
        unique_docs = {}
        for doc in candidate_docs:
            txt = doc.page_content.strip()
            if txt not in unique_docs:
                unique_docs[txt] = doc
        unique_doc_list = list(unique_docs.values())

        if not unique_doc_list:
            return []

        # 3. Cross-Encoder Reranking (Using English context for BGE model)
        has_arabic = bool(re.search(r'[\u0600-\u06FF]', query))
        if has_arabic:
            # Use the English queries for the Cross-Encoder matching
            en_queries = [q for q in retrieval_queries if not re.search(r'[\u0600-\u06FF]', q)]
            rerank_query = " ".join(en_queries[:2]) if en_queries else query
        else:
            rerank_query = query

        pairs = [[rerank_query, doc.page_content] for doc in unique_doc_list]
        scores = self.reranker.predict(pairs)

        scored_docs = list(zip(unique_doc_list, [float(s) for s in scores]))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return scored_docs[:top_k]

    def generate_answer(
        self,
        query: str,
        top_docs: List[Tuple[Any, float]]
    ) -> Dict[str, Any]:
        """Generates evidence-backed response with citations using Qwen 2.5:7B."""
        if not top_docs:
            return {
                "answer": "No relevant context could be retrieved from the document.",
                "sources": []
            }

        # Build structured context with page citations
        context_parts = []
        for i, (doc, score) in enumerate(top_docs, start=1):
            page_num = doc.metadata.get("page", "Unknown")
            if isinstance(page_num, int):
                page_num += 1  # 1-indexed for human readability
            context_parts.append(
                f"[Source {i} | Page {page_num} | Relevance Score: {score:.4f}]\n{doc.page_content.strip()}"
            )
        context_text = "\n\n" + ("=" * 50) + "\n\n".join(context_parts)

        # Grounded Medical Prompt
        system_and_user_prompt = f"""You are an expert clinical neurology research assistant specializing in epilepsy and neurotherapeutics.

Your task is to provide a complete, accurate, and rigorous answer to the user's clinical question based EXCLUSIVELY on the provided document excerpts.

CRITICAL GUIDELINES:
1. Ground every claim directly in the provided context.
2. If multiple therapies, devices, or percentages are requested, enumerate each one clearly with bullet points.
3. Preserve all numerical percentages, trial names, and responder rates exactly as reported.
4. Attribute each responder rate or outcome specifically to its corresponding device/treatment.
5. If some details are not mentioned in the context, explicitly clarify what the text states rather than giving up.
6. Provide a concise synthesis followed by structured details.
7. LANGUAGE REQUIREMENT: Respond in the EXACT SAME LANGUAGE as the user's question. If the user asks in Arabic, provide the entire clinical answer in fluent, professional, and clear Arabic (اللغة العربية الطبية السليمة) while preserving key device names and scientific terms in English/Arabic (e.g. VNS / تحفيز العصب الحائر). If the user asks in English, respond in English.

DOCUMENT CONTEXT:
{context_text}

USER CLINICAL QUESTION:
{query}

CLINICAL ANSWER:"""

        response = self.llm.invoke(system_and_user_prompt)
        answer = response.content.strip()

        # Build clean source items for UI / evaluation
        formatted_sources = []
        for i, (doc, score) in enumerate(top_docs, start=1):
            page_num = doc.metadata.get("page", "Unknown")
            if isinstance(page_num, int):
                page_num += 1
            formatted_sources.append({
                "source_id": i,
                "page": page_num,
                "score": round(score, 4),
                "text": doc.page_content.strip(),
                "metadata": doc.metadata
            })

        return {
            "answer": answer,
            "sources": formatted_sources,
            "raw_context": context_text
        }

    def ask(self, query: str, top_k: int = 8) -> Dict[str, Any]:
        """End-to-end question answering entrypoint."""
        queries = self.expand_queries(query)
        top_docs = self.retrieve_and_rerank(query, queries, top_k=top_k)
        result = self.generate_answer(query, top_docs)
        result["queries"] = queries
        return result


if __name__ == "__main__":
    engine = EpilepsyRAGEngine()
    test_query = "What are the three invasive neuromodulation options approved for adult drug-resistant epilepsy, and what are their typical responder rates?"
    print(f"\nTesting Query:\n{test_query}\n")
    output = engine.ask(test_query)
    print("=" * 80)
    print("FINAL ANSWER:")
    print("=" * 80)
    print(output["answer"])
    print("\n" + "=" * 80)
    print("TOP SOURCES RETRIEVED:")
    print("=" * 80)
    for src in output["sources"][:3]:
        print(f"Source {src['source_id']} (Page {src['page']}, Score: {src['score']}):\n{src['text'][:200]}...\n")
