"""
RAGAS Evaluation: Baseline vs. Advanced Retrieval
===================================================
Measures context_precision, context_recall, and faithfulness for:
  - Baseline: naive k=5 similarity search
  - Advanced: FlashrankRerank cross-encoder (k=5 → top_n=3)

Run:
    python ragas_baseline.py

Results are printed to stdout and saved to ragas_results.json.
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
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall, faithfulness


# ── Ground-truth Q&A pairs derived from store_policies.md ──────────────────
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
        "ground_truth": "Once a return is received and inspected, refunds are processed within 3-5 business days to the original payment method."
    },
    {
        "question": "What is the shipping cost for orders under $50?",
        "ground_truth": "Orders under $50 incur a flat $5.99 shipping fee."
    },
    {
        "question": "What happens if my order is delayed for more than 5 business days?",
        "ground_truth": "The customer is entitled to a 15% refund on their order total if the order is delayed more than 5 business days past the expected delivery date."
    },
    {
        "question": "Can I return opened headphones?",
        "ground_truth": "No, opened headphones cannot be returned due to hygiene restrictions."
    },
    {
        "question": "Do you offer price matching?",
        "ground_truth": "Yes, price matching is offered with authorized retailers within 14 days of purchase. The item must be identical and currently in stock."
    },
    {
        "question": "What is the cost of expedited shipping?",
        "ground_truth": "Expedited shipping is available for $14.99 and arrives in 1-2 business days for orders placed before 2:00 PM EST."
    },
]


def build_vectorstore():
    loader = TextLoader("store_policies.md")
    docs = loader.load()
    return Chroma.from_documents(documents=docs, embedding=FastEmbedEmbeddings())


def get_naive_contexts(vs, question, k=5):
    retriever = vs.as_retriever(search_kwargs={"k": k})
    return [doc.page_content for doc in retriever.invoke(question)]


def get_advanced_contexts(vs, question, k=5, top_n=3):
    retriever = vs.as_retriever(search_kwargs={"k": k})
    compressor = FlashrankRerank(top_n=top_n)
    advanced = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    return [doc.page_content for doc in advanced.invoke(question)]


def generate_answer(question, contexts, llm):
    prompt = f"""Answer this question based strictly on the provided context.

Context:
{chr(10).join(contexts)}

Question: {question}
Answer:"""
    return llm.invoke([HumanMessage(content=prompt)]).content.strip()


def ragas_score(questions, contexts_list, answers, ground_truths):
    dataset = Dataset.from_dict({
        "question": questions,
        "contexts": contexts_list,
        "answer": answers,
        "ground_truth": ground_truths,
    })
    return evaluate(dataset, metrics=[context_precision, context_recall, faithfulness])


if __name__ == "__main__":
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    print("Building vectorstore from store_policies.md...")
    vs = build_vectorstore()

    questions    = [item["question"]     for item in GROUND_TRUTH]
    ground_truths = [item["ground_truth"] for item in GROUND_TRUTH]

    # ── Baseline run ──────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  BASELINE — Naive k=5 retrieval")
    print(f"{'='*55}")
    naive_contexts, naive_answers = [], []
    for i, q in enumerate(questions):
        ctxs = get_naive_contexts(vs, q)
        ans  = generate_answer(q, ctxs, llm)
        naive_contexts.append(ctxs)
        naive_answers.append(ans)
        print(f"  [{i+1}/{len(questions)}] {q[:60]}...")
        time.sleep(1)  # avoid Groq rate limit

    print("\nComputing RAGAS scores for baseline...")
    baseline_scores = ragas_score(questions, naive_contexts, naive_answers, ground_truths)

    # ── Advanced run ─────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  ADVANCED — FlashrankRerank (k=5 → top_n=3)")
    print(f"{'='*55}")
    adv_contexts, adv_answers = [], []
    for i, q in enumerate(questions):
        ctxs = get_advanced_contexts(vs, q)
        ans  = generate_answer(q, ctxs, llm)
        adv_contexts.append(ctxs)
        adv_answers.append(ans)
        print(f"  [{i+1}/{len(questions)}] {q[:60]}...")
        time.sleep(1)

    print("\nComputing RAGAS scores for advanced pipeline...")
    advanced_scores = ragas_score(questions, adv_contexts, adv_answers, ground_truths)

    # ── Report ────────────────────────────────────────────────────────────────
    metrics = ["context_precision", "context_recall", "faithfulness"]
    print(f"\n{'='*60}")
    print("  RAGAS RESULTS: BASELINE vs. ADVANCED RAG")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'Baseline':>12} {'Advanced':>12} {'Delta':>8}")
    print(f"{'-'*60}")
    results = {}
    for m in metrics:
        b = float(baseline_scores[m])
        a = float(advanced_scores[m])
        delta = a - b
        arrow = "↑" if delta > 0.001 else ("↓" if delta < -0.001 else "=")
        print(f"{m:<25} {b:>12.4f} {a:>12.4f} {arrow}{abs(delta):>6.4f}")
        results[m] = {"baseline": round(b, 4), "advanced": round(a, 4), "delta": round(delta, 4)}
    print(f"{'='*60}")
    print("\nInclude these scores in the Written Report — RAG Evaluation section.")

    # Save to JSON for the report
    with open("ragas_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to ragas_results.json")
