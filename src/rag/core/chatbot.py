"""Chatbot module that provides the main Chatbot class."""

import json
from typing import Dict, Tuple, List

from .llm import LLM
from .models.chat_transaction import ChatTransaction
from .models.completion_transaction import CompletionTransaction
from .prompts import REPHRASE_USER_QUERY_PROMPT_NAME, KNOWLEDGE_BASE_QUERY_PROMPT_NAME, RAG_PROMPT_NAME
from .search import SearchService
from ..utils.pipeline_settings import PipelineSettings
from ..utils.prompt_pair import PromptPair
from ..utils.search_settings import SearchSettings

# constants used to stringify chat history
ASSISTANT = "assistant"
CHAT_GREETING = "How can I help you?"
USER = "user"


class Chatbot:
    """
    Class that represents a session of the chatbot.

    The chatbot is stateful and stores the history of the conversation.
    The prompts can be modified at any time.
    Settings regarding the RAG configuration are passed along together with each chat input.
    """

    def __init__(self, llm: LLM, search: SearchService) -> None:
        self.llm = llm
        self.search = search
        self.chat_history: List[Tuple[str, str]] = []
        self.prompts = None  # will be set via setter

    def __get_chat_history_str(self) -> str:
        chat_history_str = f"{ASSISTANT}: {CHAT_GREETING.strip()}\n"
        for (user_message, bot_message) in self.chat_history:
            chat_history_str += f"{USER}: {user_message}\n"
            chat_history_str += f"{ASSISTANT}: {bot_message}\n"
        return chat_history_str

    def __rephrase_user_intent(self, query: str, temperature: float = 0.7) -> CompletionTransaction:
        prompt_pair = self.prompts[REPHRASE_USER_QUERY_PROMPT_NAME]
        completion_transaction = self.llm.chat(system_message=prompt_pair.system_prompt,
                                               user_message=prompt_pair.user_prompt.format(
                                                   chat_history=self.__get_chat_history_str(), user_query=query),
                                               temperature=temperature)
        completion_transaction.set_name("Rephrase User Intent")
        completion_transaction.set_json_key("rephrased")
        return completion_transaction

    def __generate_knowledge_base_query(self, query: str, temperature: float = 0.0) -> CompletionTransaction:
        prompt_pair = self.prompts[KNOWLEDGE_BASE_QUERY_PROMPT_NAME]
        completion_transaction = self.llm.chat(system_message=prompt_pair.system_prompt,
                                               user_message=prompt_pair.user_prompt.format(query=query),
                                               temperature=temperature, max_tokens=200)
        completion_transaction.set_name("Generate Knowledge Base Query")
        completion_transaction.set_json_key("search_expression")
        return completion_transaction

    def __rag(self, context_list: list, query: str, temperature: float = 0.7) -> CompletionTransaction:
        context = json.dumps([{"source": doc["path"], "text": doc["content"]} for doc in context_list])
        prompt_pair = self.prompts[RAG_PROMPT_NAME]
        completion_transaction = self.llm.chat(system_message=prompt_pair.system_prompt.format(context=context),
                                               user_message=query, history=self.chat_history, temperature=temperature)
        completion_transaction.set_name("Generate Response")
        return completion_transaction

    def __trim_history(self, max_history_length: int) -> None:
        max_history_length = min(len(self.chat_history), max_history_length)
        self.chat_history = self.chat_history[-max_history_length:] if max_history_length else []

    def set_prompts(self, prompts: Dict[str, PromptPair]) -> None:
        """Set the LLM prompts that will be used for all further chat interactions."""
        self.prompts = prompts

    def chat(self, query: str, search_settings: SearchSettings, pipeline_settings: PipelineSettings) -> ChatTransaction:
        """
        Main method to interact with chatbot.

        Apart from the query, the search settings and the pipeline settings used for this transaction need to be passed
        along explicitly.
        """

        # trim chat history for next request
        self.__trim_history(pipeline_settings.num_history)

        chat_transaction = ChatTransaction(query)

        # rephrase query (optional)
        rephrased_query = query
        if pipeline_settings.input_summarization:
            rephrased_query_transaction = self.__rephrase_user_intent(query,
                                                                      temperature=pipeline_settings.input_summarization_temperature)
            chat_transaction.add_completion_transaction(rephrased_query_transaction)
            rephrased_query = rephrased_query_transaction.get_response()

        # generate knowledge base query (optional)
        knowledge_base_query = rephrased_query
        if search_settings.requires_kb_query():
            knowledge_base_query_transaction = self.__generate_knowledge_base_query(rephrased_query,
                                                                                    temperature=search_settings.temperature_kb_query)
            chat_transaction.add_completion_transaction(knowledge_base_query_transaction)
            knowledge_base_query = knowledge_base_query_transaction.get_response()

        # perform search in knowledge base
        search_transaction = self.search.search(user_query=rephrased_query, kb_query=knowledge_base_query,
                                                search_settings=search_settings)
        chat_transaction.add_search_transaction(search_transaction)

        # generate response based on found documents
        rag_transaction = self.__rag(search_transaction.documents, query, temperature=pipeline_settings.rag_temperature)
        chat_transaction.add_completion_transaction(rag_transaction)
        chat_transaction.set_response(rag_transaction.get_response())

        # keep history
        self.chat_history.append((query, rag_transaction.get_response()))

        return chat_transaction
