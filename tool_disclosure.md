# Tool Disclosure Document
**CSAI 422: Practical Data Mining — Course Project (Option A)**
**AuraTech Electronics AI Customer Support Agent**

---

## Frameworks & Libraries

| Tool / Library | Version | Role in Architecture | Justification |
|---|---|---|---|
| **LangGraph** | 1.2.4 | Core multi-agent orchestration — StateGraph, nodes, edges, conditional routing, checkpointing | Required by course specification. Provides typed state management and deterministic routing between agents. |
| **LangChain** | 1.3.x | Agent message types, document loaders, vector store integration | Ecosystem for LLM orchestration; integrates cleanly with LangGraph state schema. |
| **langchain-groq** | latest | Groq API integration for `llama-3.1-8b-instant` (main LLM) and `llama-3.3-70b-versatile` (evaluation judge) | Groq provides low-latency inference at no cost, essential for running a live demo with acceptable P95 latency. |
| **langchain-community** | latest | `FastEmbedEmbeddings`, `Chroma` vector store, `TextLoader` document loader | Community integrations reduce boilerplate for standard RAG components. |
| **ChromaDB** | 1.5.x | Vector store for policy and product catalog documents | In-process, no external server required; suitable for a course project scope. Persistent across restarts via `.chroma/` directory. |
| **fastembed** | latest | Local sentence embedding model (`BAAI/bge-small-en-v1.5`) for ChromaDB | No API key required; runs locally; competitive quality for retrieval tasks. |
| **flashrank** | latest | Cross-encoder reranking (`ms-marco-MiniLM-L-12-v2`) — Advanced RAG strategy | Lightweight reranker that runs locally. Chosen over Cohere Rerank (paid API) and `cross-encoder/ms-marco-MiniLM-L-6-v2` (heavier). Improves context precision by rescoring the top-k candidates. |
| **langgraph-checkpoint-sqlite** | 3.1.0 | `SqliteSaver` — long-term memory persistence via `memory.db` | Persists full LangGraph conversation checkpoints across server restarts without requiring a separate database server. |
| **FastAPI** | latest | REST API server — serves the chat UI and exposes `/chat`, `/chats`, `/chats/{id}` endpoints | Async-native, minimal overhead, integrates well with uvicorn for production-style serving. |
| **uvicorn** | latest | ASGI server for FastAPI | Standard ASGI runtime for FastAPI applications. |
| **pandas** | latest | Loading and preprocessing the Flipkart product catalog CSV for RAG indexing | Standard data manipulation library; used to select and format catalog rows as LangChain `Document` objects. |
| **python-dotenv** | latest | Loading `GROQ_API_KEY` from `.env` file | Keeps secrets out of source code. |

---

## Datasets

| Dataset | Source | How Used |
|---|---|---|
| **Flipkart Product Dataset** | Kaggle (`flipkart_products_dataset`) | Top 200 rows indexed as product knowledge base in ChromaDB. Columns used: Name, Brand, Selling Price, Details. Provides product-level RAG context for customer queries about specific items. |
| **AuraTech Store Policies** (`store_policies.md`) | LLM-generated (fictional store) | Primary policy knowledge base for RAG. Covers return windows, restocking fees, refund timelines, shipping costs, and exceptions. Ground truth was known, making RAGAS evaluation more meaningful. |
| **Bitext Customer Support LLM Dataset** | Hugging Face (`bitext/Bitext-customer-support-llm-chatbot-training-dataset`) | 26,872 real customer support intent/response pairs used as the happy-path evaluation set in `eval.py`. Covers order tracking, returns, billing, and delivery intents. |
| **Synthetic Order Records** (`mock_orders.json`) | Generated via Python script (`generate_orders.py`) | 200 synthetic orders covering delivered, in-transit, delayed, returned, and cancelled statuses. Used as the mock API backend for the Order Lookup Agent. |
| **Adversarial Test Cases** (`adversarial_test_cases.json`) | Hand-crafted | 12 domain-specific adversarial inputs covering data extraction attempts, policy bypass attempts, and prompt injection. Used to test all three guardrail classes. |

---

## AI Assistants

| Tool | How Used |
|---|---|
| **Claude (Anthropic)** | Used as a coding assistant throughout development: debugging LangGraph state schema issues, drafting the store policy document, generating synthetic order records, designing the evaluation rubric, and troubleshooting dependency conflicts. All architectural decisions, design choices, and code were reviewed and understood by the team. |
| **Groq / llama-3.1-8b-instant** | Main agent LLM for all customer-facing responses at runtime. |
| **Groq / llama-3.3-70b-versatile** | LLM-as-judge for the evaluation suite (`eval.py`, `ragas_baseline.py`). More capable model used for scoring to reduce judge bias. |

---

*All architectural decisions, routing logic, evaluation design, and system trade-offs are the responsibility of the project team.*
