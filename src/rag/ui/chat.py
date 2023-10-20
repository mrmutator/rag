"""Utility functions for the chat section of the Streamlit app UI"""

from typing import Dict, List, Any

import streamlit as st

from ..core.models.chat_transaction import ChatTransaction


def display_chat(transactions: List[ChatTransaction]) -> None:
    for chat_transaction in transactions:
        with st.chat_message("user"):
            st.write(chat_transaction.query)
        with st.chat_message("assistant"):
            st.write(chat_transaction.response)
            display_transaction_details(chat_transaction)


def format_score(value: float) -> str:
    return f"`{value:.2f}`" if value is not None else "`None`"


def generate_doc_header(doc: Dict[str, Any]) -> str:
    return f"|search.score={format_score(doc['@search.score'])}," \
           f"reranker_score={format_score(doc['@search.reranker_score'])}| **{doc['title']}** ({doc['section']})"


def display_documents(chat_transaction: ChatTransaction) -> None:
    search_transaction = chat_transaction.search_transactions[0]
    st.markdown(f"Text Query: `{search_transaction.text_query}`")
    st.markdown(f"Vector Query: `{search_transaction.vector_query}`")
    for doc in chat_transaction.get_documents():
        with st.expander(generate_doc_header(doc)):
            st.write(doc["content"])


def display_transaction_overview(chat_transaction: ChatTransaction) -> None:
    display_token_count_summary(chat_transaction.get_completion_tokens(), chat_transaction.get_embedding_tokens())


def display_token_count_summary(completion_tokens: int, embedding_tokens: int) -> None:
    col_completion, col_embedding = st.columns(2)
    col_completion.metric("Completion Tokens", f"{completion_tokens}")
    col_embedding.metric("Embedding Tokens", f"{embedding_tokens}")


def display_conversation_summary(transactions: int, completion_tokens: int, embedding_tokens: int) -> None:
    col_transactions, col_completion, col_embedding = st.columns(3)
    col_transactions.metric("Transactions", f"{transactions}")
    col_completion.metric("Completion Tokens", f"{completion_tokens}")
    col_embedding.metric("Embedding Tokens", f"{embedding_tokens}")


def display_token_count(usage_counts: Dict[str, int]) -> None:
    col_total, col_prompt, col_completion = st.columns(3)
    col_total.metric("Total Tokens", f"{usage_counts['total_tokens']}")
    col_prompt.metric("Prompt Tokens", f"{usage_counts['prompt_tokens']}")
    col_completion.metric("Completion Tokens", f"{usage_counts['completion_tokens']}")


def display_transaction_details(chat_transaction: ChatTransaction) -> None:
    tabs = st.tabs(["Overview", "Documents"] + [t.name for t in chat_transaction.completion_transactions])
    with tabs[0]:
        display_transaction_overview(chat_transaction)
    with tabs[1]:
        display_documents(chat_transaction)
    for i, tab in enumerate(tabs[2:]):
        with tab:
            display_token_count(chat_transaction.completion_transactions[i].completion["usage"])
            with st.expander("Messages"):
                st.write(chat_transaction.completion_transactions[i].messages)
            with st.expander("Completion"):
                st.write(chat_transaction.completion_transactions[i].completion)
