import os
import requests
import streamlit as st

API_URL = os.getenv("CAMPUS_CHAT_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Campus Chat", page_icon="🎓", layout="centered")
st.title("🎓 Campus Chat")
st.caption("Teaching demo: trained intent model + governed retrieval + evidence traces")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask about registration, advising, degree requirements, aid, services, or IT support")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    try:
        response = requests.post(f"{API_URL}/chat", json={"message": prompt}, timeout=30)
        response.raise_for_status()
        payload = response.json()
        answer = payload["answer"]
        metadata = (
            f"\n\n---\nIntent: `{payload['predicted_intent']}` | "
            f"Confidence: `{payload['confidence']}` | Request: `{payload['request_id']}`"
        )
        rendered = answer + metadata
    except Exception as exc:
        rendered = (
            "The API is not available. Start it with `uvicorn app.api:app --reload --port 8000`. "
            f"Technical detail: {exc}"
        )
    st.session_state.messages.append({"role": "assistant", "content": rendered})
    with st.chat_message("assistant"):
        st.markdown(rendered)

with st.sidebar:
    st.header("Try these questions")
    st.markdown("- When is the add/drop deadline?\n- How do I schedule advising?\n- What happens to aid if I drop a class?\n- I forgot my password.\n- How long can I borrow library books?")
    st.header("Production artifacts")
    st.markdown("The API records a trace for every request in the `evidence/` folder.")
