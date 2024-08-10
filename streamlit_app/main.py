import openai
import streamlit as st
import toml
import requests
import sys
import os
import time
from typing import List
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 

from config import CHATBOT_URL, load_env
from mediai_bot.models import Message


from os import environ
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
        try:
            response =  requests.post(CHATBOT_URL, json=[m.dict() for m in messages])
            res = response.json()["response"]
        except:
            time.sleep(3)
            response =  requests.post(CHATBOT_URL, json=[m.dict() for m in messages])
            res = response.json()["response"]
        st.markdown(res)
    st.session_state.messages.append({"role": "assistant", "content": res})