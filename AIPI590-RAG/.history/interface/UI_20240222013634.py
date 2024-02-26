
import streamlit as st
from inference import get_answer
import sys
import os
from model.embedding_model import Encoder
from model.inference_model import LlaMA2
sys.path.insert(1, os.getcwd())





st.title("💬 Chatbot")
st.caption("🚀 A streamlit chatbot powered by OpenAI LLM")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    answer, context = get_answer(question = prompt)
    answer = "A"
    msg = f"The answer of your question is {answer}. I generated the answer based on those context information from Wikipedia: {context}"
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)