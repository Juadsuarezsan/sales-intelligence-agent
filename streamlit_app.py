"""Streamlit gallery — research a company end-to-end."""
from __future__ import annotations

import os
import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Sales Intel", page_icon="🎯", layout="wide")
st.title("🎯 B2B Sales Intelligence Agent")
st.caption("Plan-execute-reflect over web search + HackerNews. Builds a structured profile + personalized email.")

EXAMPLES = ["Anthropic", "Stripe", "Vercel", "Supabase", "Vapi", "Pinecone"]

with st.sidebar:
    st.subheader("Backend")
    try:
        h = httpx.get(f"{API_URL}/health", timeout=2).json()
        st.success(f"online · v{h.get('version')}")
        st.caption(f"Tavily: {h.get('tavily_enabled')}")
        st.caption(f"LLM: {h.get('llm_enabled')}")
    except Exception:
        st.error(f"backend offline at {API_URL}")

    st.subheader("Try")
    for c in EXAMPLES:
        if st.button(c, use_container_width=True):
            st.session_state.pending = c

company = st.text_input("Company name", value=st.session_state.pop("pending", ""))
if st.button("Research", type="primary") and company:
    with st.spinner(f"Researching {company}..."):
        r = httpx.post(f"{API_URL}/api/research", timeout=60,
                        json={"company_name": company}).json()
    st.session_state.result = r

r = st.session_state.get("result")
if r:
    profile = r["profile"]
    email = r["email"]
    a, b = st.columns([3, 2])
    with a:
        st.subheader(profile["name"])
        st.write(profile.get("one_liner", ""))
        c1, c2, c3 = st.columns(3)
        c1.metric("Industry", profile.get("industry") or "—")
        c2.metric("Funding", f"${(profile.get('funding_total_usd') or 0)/1e6:.1f}M" if profile.get("funding_total_usd") else "—")
        c3.metric("Location", profile.get("location") or "—")

        if profile.get("key_people"):
            st.markdown("**Key people:**")
            for p in profile["key_people"]:
                st.write(f"- {p['name']}" + (f" — {p.get('role')}" if p.get("role") else ""))

        if profile.get("recent_signals"):
            st.markdown("**Recent signals:**")
            for s in profile["recent_signals"]:
                st.write(f"- `{s['kind']}` {s['description']}")
                if s.get("source_url"):
                    st.caption(s["source_url"])

    with b:
        st.subheader("Outreach email")
        st.write(f"**Subject:** {email['subject']}")
        st.markdown(f"{email['greeting']}\n\n{email['hook']}\n\n{email['value_prop']}\n\n{email['cta']}")
        st.metric("Personalization score", f"{email['personalization_score']:.2f}")

    st.caption(f"{r['iterations']} iterations · {r['tool_calls']} tool calls · {r['latency_ms']} ms")
