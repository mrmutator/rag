"""Utility functions for the settings section of the Streamlit app UI"""

from typing import Dict

import streamlit as st

from ..core.prompts import DEFAULT_PROMPTS
from ..utils.pipeline_settings import PipelineSettings
from ..utils.prompt_pair import PromptPair
from ..utils.search_settings import SearchSettings

DEFAULT_K = 4
DEFAULT_NUM_HISTORY = 2
DEFAULT_SCORING_PROFILE_NAME = "test"
DEFAULT_SEMANTIC_SEARCH_NAME = "test"
DEFAULT_TEMPERATURE_INPUT_SUMMARIZATION = 0.7
DEFAULT_TEMPERATURE_KB_QUERY = 0.0
DEFAULT_TEMPERATURE_RAG = 0.7
DEFAULT_TOP = 4
DEFAULT_VECTOR_FIELDS = ["sectionVector", "titleVector", "contentVector"]


def display_pipeline_settings() -> PipelineSettings:
    num_history = st.slider("History Length", 0, 10, DEFAULT_NUM_HISTORY)
    rag_temperature = st.slider("RAG Temperature", 0.0, 2.0, DEFAULT_TEMPERATURE_RAG)
    input_summarization = st.toggle("Input Summarization")
    input_summarization_temperature = DEFAULT_TEMPERATURE_INPUT_SUMMARIZATION
    if input_summarization:
        input_summarization_temperature = st.slider("Input Summarization Temperature", 0.0, 2.0,
                                                    DEFAULT_TEMPERATURE_INPUT_SUMMARIZATION)

    return PipelineSettings(num_history=num_history, input_summarization=input_summarization,
                            input_summarization_temperature=input_summarization_temperature,
                            rag_temperature=rag_temperature)


def display_search_settings() -> SearchSettings:
    semantic_configuration_name = None
    vector_search = st.radio("Vector Search", ["off", "user query", "kb query"], index=1, horizontal=True)
    text_search = st.radio("Text Search", ["off", "user query", "kb query"], horizontal=True)
    semantic_search = False
    if text_search != "off":
        semantic_search = st.checkbox("Semantic Search")
        if semantic_search:
            semantic_configuration_name = st.text_input("Semantic Configuration Name", DEFAULT_SEMANTIC_SEARCH_NAME)
    temperature_kb_query = DEFAULT_TEMPERATURE_KB_QUERY
    if vector_search == "kb query" or text_search == "kb query":
        temperature_kb_query = st.slider("KB Query Temperature", 0.0, 2.0, DEFAULT_TEMPERATURE_KB_QUERY)
    top = st.slider("Top", 1, 20, DEFAULT_TOP)
    vector_fields = st.multiselect("Vector Fields", DEFAULT_VECTOR_FIELDS, DEFAULT_VECTOR_FIELDS)
    k = st.slider("k", 1, 20, DEFAULT_K)
    scoring_profile_name = st.text_input("Scoring Profile Name", DEFAULT_SCORING_PROFILE_NAME)

    return SearchSettings(vector_search=vector_search, text_search=text_search, semantic_search=semantic_search,
                          semantic_configuration_name=semantic_configuration_name, top=top, vector_fields=vector_fields,
                          k=k, scoring_profile_name=scoring_profile_name, temperature_kb_query=temperature_kb_query)


def display_prompt_settings() -> Dict[str, PromptPair]:
    modified_prompts = dict()
    for prompt_name, prompt_pair in DEFAULT_PROMPTS.items():
        with st.expander(prompt_name):
            system_prompt = st.text_area("System Prompt", DEFAULT_PROMPTS[prompt_name].system_prompt, height=300)
            user_prompt = None
            if DEFAULT_PROMPTS[prompt_name].user_prompt:
                user_prompt = st.text_area("User Prompt", DEFAULT_PROMPTS[prompt_name].user_prompt, height=300)

            modified_prompts[prompt_name] = PromptPair(system_prompt=system_prompt, user_prompt=user_prompt)

    return modified_prompts
