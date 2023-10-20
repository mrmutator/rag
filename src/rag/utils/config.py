"""Config module that provides helper functions to set up the app"""

import os
from typing import Dict

from dotenv import load_dotenv

from ..core.chatbot import Chatbot
from ..core.llm import LLM, configure_openapi
from ..core.search import SearchService

AZURE_COGNITIVE_SEARCH_ENDPOINT = "AZURE_COGNITIVE_SEARCH_ENDPOINT"
AZURE_COGNITIVE_SEARCH_INDEX_NAME = "AZURE_COGNITIVE_SEARCH_INDEX_NAME"
AZURE_COGNITIVE_SEARCH_KEY = "AZURE_COGNITIVE_SEARCH_KEY"
AZURE_OPEN_AI_ENDPOINT = "AZURE_OPEN_AI_ENDPOINT"
AZURE_OPEN_AI_KEY = "AZURE_OPEN_AI_KEY"
AZURE_OPEN_AI_EMBEDDING_DEPLOYMENT = "AZURE_OPEN_AI_EMBEDDING_DEPLOYMENT"
AZURE_OPEN_AI_CHAT_DEPLOYMENT = "AZURE_OPEN_AI_CHAT_DEPLOYMENT"

REQUIRED_ENV_VARIABLES = [AZURE_COGNITIVE_SEARCH_ENDPOINT,
                          AZURE_COGNITIVE_SEARCH_INDEX_NAME,
                          AZURE_COGNITIVE_SEARCH_KEY,
                          AZURE_OPEN_AI_ENDPOINT,
                          AZURE_OPEN_AI_KEY,
                          AZURE_OPEN_AI_EMBEDDING_DEPLOYMENT,
                          AZURE_OPEN_AI_CHAT_DEPLOYMENT]


def load_config() -> Dict[str, str]:
    load_dotenv()
    config = dict()
    for var_name in REQUIRED_ENV_VARIABLES:
        var_value = os.getenv(var_name)
        if var_value is None:
            raise ValueError(f"Environment variable not defined: {var_name}")
        config[var_name] = var_value
    return config


def create_chatbot(config: Dict[str, str]) -> Chatbot:
    configure_openapi(endpoint=config[AZURE_OPEN_AI_ENDPOINT], key=config[AZURE_OPEN_AI_KEY])

    llm = LLM(chat_deployment_name=config[AZURE_OPEN_AI_CHAT_DEPLOYMENT],
              embedding_deployment_name=config[AZURE_OPEN_AI_EMBEDDING_DEPLOYMENT])
    search = SearchService(endpoint=config[AZURE_COGNITIVE_SEARCH_ENDPOINT],
                           search_key=config[AZURE_COGNITIVE_SEARCH_KEY],
                           index_name=config[AZURE_COGNITIVE_SEARCH_INDEX_NAME],
                           llm=llm)

    chatbot = Chatbot(llm=llm, search=search)
    return chatbot
