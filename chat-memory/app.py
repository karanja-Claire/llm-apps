import json
import os
from pathlib import Path

import streamlit as st
from openai import OpenAI

# STEP 1: Connect to the local model server here Ollama
client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.getenv("LLM_API_KEY", "ollama"),  # ignored locally, but required by the client
)
MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")

# STEP 5: persisted data 
MEMORY_FILE = Path(__file__).parent / "memory.json"


def load_memory():
    if MEMORY_FILE.exists():
        return json.loads(MEMORY_FILE.read_text())
    return []


def save_memory(facts):
    MEMORY_FILE.write_text(json.dumps(facts, indent=2))


def build_system_prompt(facts):
    prompt = "You are a helpful assistant. Be concise."
    if facts:
        prompt += "\n\nThings you know about the user:\n" + "\n".join(f"- {f}" for f in facts)
    return prompt


st.title("Local ChatGPT with Memory")
st.caption(f"{MODEL} running locally. Type /add <fact> to remember something.")

# STEP 3: Short-term memory chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Long-term memory")
    facts = load_memory()
    for f in facts:
        st.write("•", f)
    if st.button("Clear memory"):
        save_memory([])
        st.rerun() #refresh to show memories have been cleared.
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"]) # display message based on role

if prompt := st.chat_input("What is up?"):
    # STEP 5: /add saves a fact instead of calling the model
    if prompt.startswith("/add "):
        facts.append(prompt[5:].strip())
        save_memory(facts)
        st.rerun()
    #Store the user's message in conversation history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

   
    request = [{"role": "system", "content": build_system_prompt(facts)}] # persisted memory
    request += st.session_state.messages[-20:]

    # STEP 6: stream tokens as they are generated
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=MODEL, messages=request, temperature=temperature, stream=True
        )
        reply = st.write_stream(
            chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices
        )

    st.session_state.messages.append({"role": "assistant", "content": reply})