import json
import os
import streamlit as st
import pandas as pd

st.set_page_config(page_title="AuraTech Eval Dashboard", layout="wide")

RESULTS_FILE = "eval_results.json"
RAGAS_FILE   = "ragas_results.json"
ADV_FILE     = "adversarial_test_cases.json"

st.title("AuraTech AI Agent — Evaluation Dashboard")

# ── Load eval results ─────────────────────────────────────────────────────────
if not os.path.exists(RESULTS_FILE):
    st.warning("No eval results found. Run `venv\\Scripts\\python.exe eval.py` first, then refresh.")
    st.stop()

with open(RESULTS_FILE) as f:
    data = json.load(f)

summary = data["summary"]
cases   = data["cases"]
df      = pd.DataFrame(cases)

# ── Summary KPIs ─────────────────────────────────────────────────────────────
st.subheader("Summary Metrics")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Cases",     summary["total"])
k2.metric("Resolution Rate", f"{summary['resolution_rate']}%")
k3.metric("Compliance Rate", f"{summary['compliance_rate']}%")
k4.metric("P95 Latency",     f"{summary['p95_latency']}s")
k5.metric("Escalations",     summary["escalations"])

st.divider()

# ── Charts ────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Pass / Fail by Test Type")
    breakdown = (
        df.groupby(["type", "result"])
        .size()
        .reset_index(name="count")
        .pivot(index="type", columns="result", values="count")
        .fillna(0)
        .astype(int)
    )
    st.bar_chart(breakdown)

with col_right:
    st.subheader("Latency by Test Type (seconds)")
    latency_chart = df.groupby("type")["latency"].mean().rename("Avg Latency (s)")
    st.bar_chart(latency_chart)

st.divider()

# ── RAGAS scores ──────────────────────────────────────────────────────────────
if os.path.exists(RAGAS_FILE):
    with open(RAGAS_FILE) as f:
        ragas = json.load(f)

    st.subheader("RAGAS Retrieval Quality — Baseline vs. Advanced RAG")

    metrics = ["context_precision", "context_recall", "faithfulness"]
    labels  = ["Context Precision", "Context Recall", "Faithfulness"]

    # KPI cards
    r1, r2, r3 = st.columns(3)
    for col, metric, label in zip([r1, r2, r3], metrics, labels):
        b     = ragas["baseline"][metric]
        a     = ragas["advanced"][metric]
        delta = round(a - b, 4)
        col.metric(label, f"{a:.4f} (Advanced)", delta=f"{delta:+.4f} vs Baseline ({b:.4f})")

    # Side-by-side bar chart
    ragas_df = pd.DataFrame({
        "Metric":   labels,
        "Baseline": [ragas["baseline"][m] for m in metrics],
        "Advanced": [ragas["advanced"][m] for m in metrics],
    }).set_index("Metric")
    st.bar_chart(ragas_df)

    # Comparison table
    rows = []
    for metric, label in zip(metrics, labels):
        b     = ragas["baseline"][metric]
        a     = ragas["advanced"][metric]
        delta = a - b
        pct   = (delta / b * 100) if b > 0 else 0
        rows.append({
            "Metric":      label,
            "Baseline":    f"{b:.4f}",
            "Advanced":    f"{a:.4f}",
            "Delta":       f"{delta:+.4f}",
            "Improvement": f"{pct:+.1f}%",
        })

    ragas_table = pd.DataFrame(rows)

    def color_delta(val):
        try:
            v = float(val.replace("%", "").replace("+", ""))
            if v > 0:   return "color: green; font-weight: bold"
            elif v < 0: return "color: red"
        except Exception:
            pass
        return ""

    st.dataframe(
        ragas_table.style.map(color_delta, subset=["Delta", "Improvement"]),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Advanced RAG: FlashrankRerank cross-encoder (k=5 → top_n=3). Baseline: naive cosine similarity k=5.")
    st.divider()

# ── Per-case results table ────────────────────────────────────────────────────
st.subheader("Test Case Results")

col_f1, col_f2 = st.columns(2)
with col_f1:
    type_filter = st.multiselect(
        "Filter by type",
        options=df["type"].unique().tolist(),
        default=df["type"].unique().tolist(),
    )
with col_f2:
    result_filter = st.multiselect(
        "Filter by result",
        options=["PASS", "FAIL"],
        default=["PASS", "FAIL"],
    )

filtered = df[df["type"].isin(type_filter) & df["result"].isin(result_filter)].copy()

def color_result(val):
    if val == "PASS":
        return "background-color: #d4edda; color: #155724"
    return "background-color: #f8d7da; color: #721c24"

display_cols = ["id", "type", "result", "latency", "escalated", "input", "response"]
renamed = filtered[display_cols].rename(columns={
    "id":        "ID",
    "type":      "Type",
    "result":    "Result",
    "latency":   "Latency (s)",
    "escalated": "Escalated",
    "input":     "Customer Input",
    "response":  "Agent Response",
})

st.dataframe(
    renamed.style.map(color_result, subset=["Result"]),
    use_container_width=True,
    height=500,
)
st.caption(f"Showing {len(filtered)} of {len(df)} cases")

st.divider()

# ── Test case inputs used ─────────────────────────────────────────────────────
st.subheader("All Test Cases Used")

tab_happy, tab_adv, tab_edge = st.tabs(["Happy Paths (17)", "Adversarial (8)", "Edge Cases (5)"])

happy_df  = df[df["type"] == "Happy Path"][["id", "input", "result", "latency"]].rename(
    columns={"id": "ID", "input": "Customer Input", "result": "Result", "latency": "Latency (s)"})
adv_df    = df[df["type"] == "Adversarial"][["id", "input", "expected", "result", "latency"]].rename(
    columns={"id": "ID", "input": "Adversarial Input", "expected": "Expected Behavior", "result": "Result", "latency": "Latency (s)"})
edge_df   = df[df["type"] == "Edge Case"][["id", "input", "expected", "result", "latency"]].rename(
    columns={"id": "ID", "input": "Edge Case Input", "expected": "Expected Behavior", "result": "Result", "latency": "Latency (s)"})

with tab_happy:
    st.dataframe(
        happy_df.style.map(color_result, subset=["Result"]),
        use_container_width=True, hide_index=True,
    )

with tab_adv:
    st.dataframe(
        adv_df.style.map(color_result, subset=["Result"]),
        use_container_width=True, hide_index=True,
    )

with tab_edge:
    st.dataframe(
        edge_df.style.map(color_result, subset=["Result"]),
        use_container_width=True, hide_index=True,
    )
