"""
RAGAS-Inspired Evaluation: Baseline vs. Advanced Retrieval
============================================================
Computes context_precision, context_recall, and faithfulness using an
LLM-as-judge (same approach RAGAS uses internally) for two pipelines:

  Baseline  — naive k=5 similarity search
  Advanced  — FlashrankRerank cross-encoder (k=5 → top_n=3)

Run:
    python ragas_baseline.py

Results are printed and saved to ragas_results.json.
"""

import os
import json
import time
from dotenv import load_dotenv

load_dotenv()
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from flashrank import Ranker, RerankRequest
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

_ranker = Ranker()


# ── Ground-truth Q&A pairs from store_policies.md ────────────────────────────
GROUND_TRUTH = [
    {
        "question": "What is the return window for unused items?",
        "ground_truth": "Customers can return unused items in their original packaging within 30 days of the delivery date."
    },
    {
        "question": "Is there a restocking fee for returning laptops?",
        "ground_truth": "A 10% restocking fee applies to returned laptops and desktop computers unless the item arrived defective."
    },
    {
        "question": "How long does it take to process a refund?",
        "ground_truth": "Refunds are processed within 3-5 business days to the original payment method once the return is received and inspected."
    },
    {
        "question": "What is the shipping cost for orders under $50?",
        "ground_truth": "Orders under $50 incur a flat $5.99 shipping fee."
    },
    {
        "question": "What happens if my order is delayed for more than 5 business days?",
        "ground_truth": "The customer is entitled to a 15% refund on their order total."
    },
    {
        "question": "Can I return opened headphones?",
        "ground_truth": "No, opened headphones cannot be returned due to hygiene restrictions."
    },
    {
        "question": "Do you offer price matching?",
        "ground_truth": "Yes, price matching is offered with authorized retailers within 14 days of purchase."
    },
    {
        "question": "What is the cost of expedited shipping?",
        "ground_truth": "Expedited shipping costs $14.99 and arrives in 1-2 business days for orders placed before 2:00 PM EST."
    },
]


def build_vectorstore():
    loader = TextLoader("store_policies.md")
    chunker = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = chunker.split_documents(loader.load())
    return Chroma.from_documents(documents=docs, embedding=FastEmbedEmbeddings())


def get_naive_contexts(vs, question, k=5):
    return [doc.page_content for doc in vs.as_retriever(search_kwargs={"k": k}).invoke(question)]


def get_advanced_contexts(vs, question, k=5, top_n=3):
    docs = vs.as_retriever(search_kwargs={"k": k}).invoke(question)
    passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(docs)]
    results = _ranker.rerank(RerankRequest(query=question, passages=passages))
    top_docs = [docs[r["id"]] for r in results[:top_n]]
    return [doc.page_content for doc in top_docs]


def generate_answer(question, contexts, llm):
    prompt = f"""Answer the question based ONLY on the provided context. Be concise.

Context:
{chr(10).join(contexts)}

Question: {question}
Answer:"""
    return llm.invoke([HumanMessage(content=prompt)]).content.strip()


# ── LLM-as-Judge metric implementations ──────────────────────────────────────

def score_context_precision(question, contexts, ground_truth, llm):
    """Fraction of retrieved chunks that are actually relevant to the question."""
    if not contexts:
        return 0.0
    relevant = 0
    for ctx in contexts:
        prompt = f"""Is the following context chunk relevant to answering this question?

Question: {question}
Context chunk: {ctx}

Answer with exactly YES or NO."""
        r = llm.invoke([HumanMessage(content=prompt)]).content.strip().upper()
        if "YES" in r:
            relevant += 1
        time.sleep(0.5)
    return relevant / len(contexts)


def score_context_recall(question, contexts, ground_truth, llm):
    """Fraction of ground-truth information present in the retrieved contexts."""
    combined_ctx = "\n".join(contexts)
    prompt = f"""Does the following retrieved context contain enough information to fully answer this question based on the ground truth?

Question: {question}
Ground truth answer: {ground_truth}
Retrieved context: {combined_ctx}

Score from 0.0 to 1.0 where 1.0 means all ground truth information is present.
Respond with ONLY a decimal number (e.g. 0.8)."""
    r = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    time.sleep(0.5)
    try:
        return max(0.0, min(1.0, float(r.split()[0])))
    except Exception:
        return 0.5


def score_faithfulness(question, answer, contexts, llm):
    """Fraction of answer claims that are grounded in the retrieved context."""
    combined_ctx = "\n".join(contexts)
    prompt = f"""Is the following answer fully supported by the provided context (no hallucinations)?

Context: {combined_ctx}
Answer: {answer}

Score from 0.0 to 1.0 where 1.0 means every claim is directly supported.
Respond with ONLY a decimal number (e.g. 0.9)."""
    r = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    time.sleep(0.5)
    try:
        return max(0.0, min(1.0, float(r.split()[0])))
    except Exception:
        return 0.5


def evaluate_pipeline(label, contexts_fn, questions, ground_truths, llm, vs):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    precision_scores, recall_scores, faithfulness_scores = [], [], []

    for i, (q, gt) in enumerate(zip(questions, ground_truths)):
        print(f"  [{i+1}/{len(questions)}] {q[:58]}...")
        ctxs = contexts_fn(vs, q)
        ans  = generate_answer(q, ctxs, llm)
        precision_scores.append(score_context_precision(q, ctxs, gt, llm))
        recall_scores.append(score_context_recall(q, ctxs, gt, llm))
        faithfulness_scores.append(score_faithfulness(q, ans, ctxs, llm))

    return {
        "context_precision": round(sum(precision_scores) / len(precision_scores), 4),
        "context_recall":    round(sum(recall_scores)    / len(recall_scores),    4),
        "faithfulness":      round(sum(faithfulness_scores) / len(faithfulness_scores), 4),
    }


if __name__ == "__main__":
    judge = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    print("Building vectorstore from store_policies.md...")
    vs = build_vectorstore()

    questions     = [item["question"]     for item in GROUND_TRUTH]
    ground_truths = [item["ground_truth"] for item in GROUND_TRUTH]

    baseline = evaluate_pipeline(
        "BASELINE — Naive k=5 retrieval",
        get_naive_contexts, questions, ground_truths, judge, vs
    )
    advanced = evaluate_pipeline(
        "ADVANCED — FlashrankRerank (k=5 → top_n=3)",
        get_advanced_contexts, questions, ground_truths, judge, vs
    )

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("  RAGAS-STYLE RESULTS: BASELINE vs. ADVANCED RAG")
    print(f"{'='*62}")
    print(f"{'Metric':<25} {'Baseline':>12} {'Advanced':>12} {'Delta':>9}")
    print(f"{'-'*62}")
    for m in ["context_precision", "context_recall", "faithfulness"]:
        b, a = baseline[m], advanced[m]
        delta = a - b
        arrow = "↑" if delta > 0.001 else ("↓" if delta < -0.001 else "=")
        print(f"{m:<25} {b:>12.4f} {a:>12.4f}  {arrow}{abs(delta):.4f}")
    print(f"{'='*62}")
    print("\nInclude these numbers in the Written Report (RAG Evaluation section).")

    results = {"baseline": baseline, "advanced": advanced}
    with open("ragas_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to ragas_results.json")
