import openai
import streamlit as st
import toml
import requests
import sys
import os
from pydantic import BaseModel, Field
from typing import List
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 

from config import CHATBOT_URL, load_env

from os import environ


class Message(BaseModel):
    role: str
    content: str
    
load_env()

st.title("MediAI Chatbot")

openai.api_key = environ["OPENAI_API_KEY"]

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        messages: List[Message]=[
            Message(role=m["role"], content=m["content"])
            for m in st.session_state.messages
        ]

        response =  requests.post(CHATBOT_URL, json=[m.dict() for m in messages])
        st.markdown(response.json()["response"])
    st.session_state.messages.append({"role": "assistant", "content": response.json()["response"]})