"""Streamlit App to run a RAG setup with Azure Open AI and Azure Cognitive Search"""

import streamlit as st

from rag.ui.chat import display_chat, display_conversation_summary
from rag.ui.settings import display_pipeline_settings
from rag.ui.settings import display_prompt_settings
from rag.ui.settings import display_search_settings
from rag.utils.config import load_config, create_chatbot

st.set_page_config(layout="wide")

# session state
if "bot" not in st.session_state:
    st.session_state.bot = create_chatbot(load_config())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# settings sidebar
with st.sidebar:
    st.title("Current Conversation")
    display_conversation_summary(len(st.session_state.chat_history),
                                 sum([t.get_completion_tokens() for t in st.session_state.chat_history]),
                                 sum([t.get_embedding_tokens() for t in st.session_state.chat_history]))

    if st.session_state.chat_history:
        reset = st.button("Restart Session")
        if reset:
            st.session_state.chat_history = []
            st.session_state.bot.chat_history = []
            st.rerun()

    st.title("Settings")
    with st.expander("Pipeline"):
        pipeline_settings = display_pipeline_settings()
    with st.expander("Search"):
        search_settings = display_search_settings()

    st.title("Prompts")
    prompts = display_prompt_settings()

# main section
st.title("RAG Tester")
display_chat(st.session_state.chat_history)
prompt = st.chat_input("Say something", disabled=search_settings.invalid())
if prompt:
    with st.spinner("Processing..."):
        st.session_state.bot.set_prompts(prompts)
        chat_transaction = st.session_state.bot.chat(prompt, search_settings, pipeline_settings)
        st.session_state.chat_history.append(chat_transaction)
    st.rerun()
