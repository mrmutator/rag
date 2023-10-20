"""Config module that provides helper functions to set up the app"""

from typing import Dict, Union

import yaml
from yaml.loader import SafeLoader

from ..core.chatbot import Chatbot
from ..core.llm import LLM, configure_openapi
from ..core.search import SearchService

CHAT_DEPLOYMENT = "chat_deployment"
EMBEDDING_DEPLOYMENT = "embedding_deployment"
ENDPOINT = "endpoint"
HYBRID_SEARCH = "hybrid_search"
KEY = "key"
OPEN_AI = "openai"
SEARCH = "search"
VECTOR_FIELDS = "vector_fields"


def load_config(file_path: str = "config.yaml") -> dict:
    with open(file_path) as f:
        return yaml.load(f, Loader=SafeLoader)


def to_yaml(config: Dict[str, str]):
    return yaml.dump(config)


def from_yaml(s: str):
    return yaml.safe_load(s)


def create_chatbot(config: Dict[str, Union[str, Dict[str, str]]]) -> Chatbot:
    openai_config = config[OPEN_AI]
    search_config = config[SEARCH]

    configure_openapi(openai_config[ENDPOINT], openai_config[KEY])

    llm = LLM(openai_config[CHAT_DEPLOYMENT], openai_config[EMBEDDING_DEPLOYMENT])
    search = SearchService(llm=llm, **search_config)

    chatbot = Chatbot(llm, search)
    return chatbot
