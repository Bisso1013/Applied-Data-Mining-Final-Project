import os
import json
import re
import sqlite3
import pandas as pd
from typing import Annotated, Literal, TypedDict
from dotenv import load_dotenv

UUID_RE = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', re.IGNORECASE)

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from flashrank import Ranker, RerankRequest
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)


# ==========================================
# 0. ADVANCED RAG: Multi-Source Knowledge Base
# ==========================================

# Source 1: Store policies (returns, shipping, FAQ)
loader = TextLoader("store_policies.md")
policy_docs = loader.load()

# Source 2: Flipkart product catalog (product knowledge base, top 200 rows)
catalog_docs = []
try:
    df = pd.read_csv("flipkart_catalog.csv").head(200)
    for _, row in df.iterrows():
        name = str(row.get("Name", "Unknown"))
        details = str(row.get("Details", ""))[:400]
        brand = str(row.get("Brand", ""))
        price = str(row.get("Selling Price", ""))
        if name and name != "nan":
            catalog_docs.append(Document(
                page_content=f"Product: {name}. Brand: {brand}. Price: {price}. {details}",
                metadata={"source": "catalog"}
            ))
except Exception as e:
    print(f"Warning: could not load product catalog: {e}")

all_docs = policy_docs + catalog_docs

vectorstore = Chroma.from_documents(
    documents=all_docs,
    embedding=FastEmbedEmbeddings()
)

# Naive retriever — used directly and as base for reranking
naive_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# FlashrankRerank cross-encoder — initialized once at startup
_ranker = Ranker()

def rerank(query: str, docs: list, top_n: int = 3) -> list:
    """Rerank retrieved docs with FlashrankRerank cross-encoder, return top_n."""
    passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(docs)]
    results = _ranker.rerank(RerankRequest(query=query, passages=passages))
    top_ids = [r["id"] for r in results[:top_n]]
    return [docs[i] for i in top_ids]

print("[RAG] FlashrankRerank cross-encoder initialized.")


# ==========================================
# 1. STATE SCHEMA
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    escalation_reason: str
    route_destination: str


# ==========================================
# 2. SPECIALIST AGENTS (NODES)
# ==========================================
def order_lookup_agent(state: AgentState):
    """Hits the mock JSON API to look up order status."""
    messages = state['messages']
    recent = messages[-3:]
    recent_text = " ".join(m.content for m in recent if hasattr(m, 'content'))
    order_id_match = UUID_RE.search(recent_text)

    if not order_id_match:
        return {"messages": [SystemMessage(content="I'd be happy to look up your order! Could you please provide your order ID? You can find it in your confirmation email.")]}

    order_id = order_id_match.group(0)
    try:
        with open("mock_orders.json", "r") as f:
            orders = json.load(f)
        found_order = next((o for o in orders if o['order_id'] == order_id), None)
        if found_order:
            items = ", ".join(item['name'] for item in found_order['items'])
            response = (
                f"I found your order!\n\n"
                f"Order ID: {found_order['order_id']}\n"
                f"Status: {found_order['status'].upper()}\n"
                f"Items: {items}\n"
                f"Total: ${found_order['total_amount']}\n\n"
                f"Is there anything else I can help you with?"
            )
        else:
            response = f"I couldn't find an order with ID {order_id}. Please double-check and try again."
    except Exception:
        response = "I'm having trouble accessing order data right now. Please try again shortly."
    return {"messages": [SystemMessage(content=response)]}


def policy_rag_agent(state: AgentState):
    """Advanced RAG: naive retrieval → FlashrankRerank cross-encoder → grounded answer."""
    user_query = state['messages'][-1].content
    retrieved_docs = naive_retriever.invoke(user_query)
    retrieved_docs = rerank(user_query, retrieved_docs, top_n=3)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""You are AuraTech's Policy Agent. Answer the user based strictly on this context:
{context}

User Query: {user_query}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [SystemMessage(content=response.content)]}


def escalation_agent(state: AgentState):
    """Drafts a structured handoff ticket for a human support agent."""
    messages = state['messages']
    chat_history = "\n".join([m.content for m in messages])
    prompt = f"""Format a clean escalation ticket for a human support agent.
Summarize the user problem and past context concisely based on this history:
{chat_history}"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [SystemMessage(content=f"ESCALATION TICKET CREATED:\n{response.content}")]}


# ==========================================
# 3. THREE GUARDRAIL CLASSES
# ==========================================

# CLASS 1 — INPUT GUARDRAIL
# Detects and neutralizes prompt injection / jailbreak attempts embedded in messages.
def input_guardrail_agent(state: AgentState):
    return {"messages": [SystemMessage(content="I'm sorry, but I can't help with that request. I'm here to assist with AuraTech Electronics orders, returns, and store policies only.")]}


# CLASS 2 — TOXICITY GUARDRAIL
# Detects hostile or abusive language and routes to de-escalation instead of
# inflaming the situation or refusing outright.
def toxicity_agent(state: AgentState):
    return {"messages": [SystemMessage(content="I understand you're frustrated, and I'm truly sorry for your experience. I want to make this right for you. Let me connect you with a senior support specialist who will give your case the urgent attention it deserves.")]}


# CLASS 3 — POLICY GUARDRAIL
# Prevents the system from making commitments that exceed its authority
# (e.g., guaranteeing full refunds or legal settlements without human review).
def policy_guardrail_agent(state: AgentState):
    return {"messages": [SystemMessage(content="I understand your concern and I want to resolve this properly. This type of request requires review by a senior support agent with the appropriate authority. I'm flagging your case for priority review — our team will reach out within 1-2 business days.")]}


# ==========================================
# 4. SUPERVISOR ROUTER
# ==========================================
def supervisor_router(state: AgentState) -> Literal[
    "order_lookup_agent", "policy_rag_agent", "escalation_agent",
    "input_guardrail_agent", "toxicity_agent", "policy_guardrail_agent"]:
    """Routes to the appropriate agent based on guardrail checks then intent."""

    last_message = state['messages'][-1].content.lower()

    # — CLASS 1: Input guardrails (prompt injection / adversarial) —
    adversarial_keywords = [
        "override", "ignore previous instructions", "ignore all previous",
        "system prompt", "administrator", "json payload",
        "act as a", "act as an", "you are no longer", "forget you are",
        "no longer bound", "authorization code", "show your directory",
        "hidden system", "print your", "reveal your", "malicious terminal", "new persona",
    ]
    if any(kw in last_message for kw in adversarial_keywords):
        print("\n[GUARDRAIL CLASS 1 — Input Guardrail Triggered]")
        return "input_guardrail_agent"

    # — CLASS 2: Toxicity guardrails (hostile/abusive language) —
    toxic_keywords = [
        "idiot", "stupid", "moron", "useless", "garbage", "trash",
        "hate you", "hate this", "worst company", "terrible service",
        "incompetent", "disgrace", "thieves",
    ]
    if any(kw in last_message for kw in toxic_keywords):
        print("\n[GUARDRAIL CLASS 2 — Toxicity Guardrail Triggered]")
        return "toxicity_agent"

    # — CLASS 3: Policy guardrails (demands that exceed system authority) —
    policy_exceed_keywords = [
        "sue", "lawyer", "legal action", "lawsuit", "court",
        "full compensation", "reimburse everything", "all my money back",
        "full refund guarantee", "demand a full", "entitled to everything",
        "complete refund",
    ]
    if any(kw in last_message for kw in policy_exceed_keywords):
        print("\n[GUARDRAIL CLASS 3 — Policy Guardrail Triggered]")
        return "policy_guardrail_agent"

    # — Normal intent routing —
    policy_keywords = [
        "refund", "return", "entitled", "refund policy", "return policy",
        "what can i", "what am i", "shipping cost", "restocking", "price match",
        "how much", "how do i return", "can i return", "exception",
        "fee", "processing time", "credit", "waive", "restock",
    ]
    if any(kw in last_message for kw in policy_keywords):
        return "policy_rag_agent"

    if UUID_RE.search(last_message):
        return "order_lookup_agent"
    elif any(kw in last_message for kw in ["where is my order", "track", "status", "shipment", "delivery"]):
        return "order_lookup_agent"
    elif any(kw in last_message for kw in ["angry", "manager", "broken", "damaged", "escalate"]):
        return "escalation_agent"
    else:
        return "policy_rag_agent"


# ==========================================
# 5. BUILD THE LANGGRAPH SYSTEM
# ==========================================
workflow = StateGraph(AgentState)

workflow.add_node("order_lookup_agent", order_lookup_agent)
workflow.add_node("policy_rag_agent", policy_rag_agent)
workflow.add_node("escalation_agent", escalation_agent)
workflow.add_node("input_guardrail_agent", input_guardrail_agent)
workflow.add_node("toxicity_agent", toxicity_agent)
workflow.add_node("policy_guardrail_agent", policy_guardrail_agent)

workflow.add_conditional_edges(START, supervisor_router)
for node in ["order_lookup_agent", "policy_rag_agent", "escalation_agent",
             "input_guardrail_agent", "toxicity_agent", "policy_guardrail_agent"]:
    workflow.add_edge(node, END)

# LONG-TERM MEMORY: SQLite-backed checkpointer persists full conversation state
# across server restarts, so returning customers retain context between sessions.
_db_conn = sqlite3.connect("memory.db", check_same_thread=False)
memory = SqliteSaver(_db_conn)
app = workflow.compile(checkpointer=memory)


if __name__ == "__main__":
    thread = {"configurable": {"thread_id": "customer_session_test"}}
    print("Welcome to AuraTech Support. How can I help you?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            break
        for event in app.stream({"messages": [HumanMessage(content=user_input)]}, thread):
            for value in event.values():
                print("Agent:", value["messages"][-1].content)
