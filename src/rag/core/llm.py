"""LLM module that provides LLM class based on Azure Open AI."""

from typing import Any, Dict, List, Tuple, Optional

import openai

from .models.completion_transaction import CompletionTransaction

ASSISTANT = "assistant"
SYSTEM = "system"
USER = "user"


def configure_openapi(endpoint: str, key: str):
    """Set the API configuration for OpenAI globally."""
    openai.api_type = "azure"
    openai.api_base = endpoint
    openai.api_version = "2023-03-15-preview"
    openai.api_key = key


class LLM:
    """
    Client for Azure Open AI LLMs.

    Supports the Chat API and the Embedding API.
    """

    def __init__(self, chat_deployment_name: str, embedding_deployment_name: str):
        self.chat_deployment_name = chat_deployment_name
        self.embedding_deployment_name = embedding_deployment_name

    def chat(self, system_message: str, user_message: str, history: Optional[List[Tuple[str, str]]] = None,
             temperature: float = 0.7, max_tokens: int = 1024, n: int = 1) -> CompletionTransaction:
        """
        Performs a chat request to the generative LLM.

        The system message (system prompt) primes the model and the user message is the actual chat input for this
        request.
        If available, a chat history can be passed along to provide context to the model.
        """

        messages = [
            {
                "role": SYSTEM,
                "content": system_message,
            }
        ]
        if history is not None:
            for u_m, a_m in history:
                messages.append({"role": USER, "content": u_m})
                messages.append({"role": ASSISTANT, "content": a_m})
        messages.append({"role": USER, "content": user_message})

        chat_intent_completion = openai.ChatCompletion.create(
            deployment_id=self.chat_deployment_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
        )
        return CompletionTransaction(chat_intent_completion, messages)

    def embedding(self, text: str) -> Dict[str, Any]:
        """Performs an embedding request to the embedding model."""
        embedding_completion = openai.Embedding.create(engine=self.embedding_deployment_name, input=text)
        return embedding_completion
