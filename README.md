# AuraTech Electronics — AI Customer Support Agent
**CSAI 422: Practical Data Mining — Course Project (Option A)**

A production-grade multi-agent AI support system built with LangGraph that handles order tracking, return/refund policy queries, escalations, and adversarial inputs — all grounded in real product and policy data via an advanced RAG pipeline.

---

## Architecture

```
User Message
     │
     ▼
┌─────────────────────────────────────────────────────┐
│              Supervisor Router                       │
│  (keyword-based intent classification + guardrails) │
└──────┬────────────┬──────────────┬──────────────────┘
       │            │              │
  Guardrails    Specialists   Guardrails
       │            │              │
  ┌────┴────┐  ┌────┴──────────────┴────┐  
  │ Input   │  │ Order Lookup Agent     │  
  │Guardrail│  │ Policy RAG Agent       │  
  │Toxicity │  │ Escalation Agent       │  
  │Guardrail│  └────────────────────────┘  
  │ Policy  │
  │Guardrail│
  └─────────┘
       │
       ▼
   Response + SQLite Memory Checkpoint
```

### Core Pillars

| Pillar | Implementation |
|--------|---------------|
| **Advanced RAG** | ChromaDB + FastEmbedEmbeddings on `store_policies.md` + Flipkart product catalog. FlashrankRerank cross-encoder reranking (k=5 → top_n=3). Baseline vs. final RAGAS scores in `ragas_results.json`. |
| **Multi-Agent Design** | LangGraph `StateGraph` with 6 nodes: `order_lookup_agent`, `policy_rag_agent`, `escalation_agent`, `input_guardrail_agent`, `toxicity_agent`, `policy_guardrail_agent`. |
| **Memory** | **Short-term**: `MemorySaver`-equivalent — full in-session context via LangGraph state. **Long-term**: `SqliteSaver` (`memory.db`) persists conversation checkpoints across server restarts. |
| **Guardrails** | Three classes: (1) Input — blocks prompt injection/jailbreaks. (2) Toxicity — de-escalates hostile messages. (3) Policy — flags requests exceeding system authority. |
| **Observability & Eval** | `eval.py` — LLM-as-judge (llama-3.3-70b), P95 latency, resolution rate, policy compliance rate. `ragas_baseline.py` — context precision, context recall, faithfulness (baseline vs. advanced). |

---

## Project Structure

```
├── main.py                    # LangGraph multi-agent system
├── server.py                  # FastAPI server + session persistence
├── index.html                 # Custom chat UI (HTML/CSS/JS)
├── store_policies.md          # RAG knowledge base (returns, shipping, FAQ)
├── mock_orders.json           # Synthetic order database (mock API)
├── flipkart_catalog.csv       # Product catalog (indexed into RAG)
├── eval.py                    # Evaluation suite (LLM-as-judge + metrics)
├── ragas_baseline.py          # RAGAS baseline vs. advanced comparison
├── adversarial_test_cases.json # Guardrail test cases
├── evaluation_happy_paths.csv # Happy path test set (30+ cases)
├── generate_orders.py         # Synthetic order data generator
├── app.py                     # Original Gradio prototype
└── .env                       # API keys (not committed)
```

---

## Setup & Run

### Prerequisites
- Python 3.10+
- A [Groq API key](https://console.groq.com) (free)

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd Applied_Final_Project
```

### 2. Create a virtual environment and install dependencies
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

pip install fastapi uvicorn langchain langchain-groq langchain-community \
            langgraph langgraph-checkpoint-sqlite chromadb fastembed \
            python-dotenv flashrank pandas numpy
```

### 3. Configure your API key
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Start the web app
```bash
python -m uvicorn server:server --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000` in your browser.

---

## Running the Evaluation Suite

### Full evaluation (LLM-as-judge, P95 latency, resolution rate)
```bash
python eval.py
```

### RAGAS baseline vs. advanced retrieval comparison
```bash
python ragas_baseline.py
```
Results are printed to stdout and saved to `ragas_results.json`.

---

## Environment Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `langchain` | ≥0.2 | Agent orchestration framework |
| `langchain-groq` | latest | Groq LLM integration |
| `langchain-community` | ≥0.2 | Document loaders, vectorstore, reranker |
| `langgraph` | 1.2.4 | Multi-agent state graph |
| `langgraph-checkpoint-sqlite` | 3.1.0 | Long-term memory persistence |
| `chromadb` | latest | Vector store |
| `fastembed` | latest | Embedding model (no API key needed) |
| `flashrank` | latest | Cross-encoder reranking |
| `fastapi` + `uvicorn` | latest | Web server |
| `groq` (via langchain-groq) | latest | LLM: `llama-3.1-8b-instant` |

---

## LLM
- **Main agent LLM**: `llama-3.1-8b-instant` via Groq API
- **Evaluation judge LLM**: `llama-3.3-70b-versatile` via Groq API
- **Embeddings**: `FastEmbedEmbeddings` (local, no API key required)
