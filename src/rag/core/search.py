"""Search module provides service to interact with Azure Cognitive Search."""

from typing import Optional

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import Vector

from .llm import LLM
from .models.search_transaction import SearchTransaction
from ..utils.search_settings import SearchSettings


def choose_query_option(vector_search: str, user_query: str, kb_query: Optional[str]) -> str:
    chosen = None
    if vector_search == "user query":
        chosen = user_query
    elif vector_search == "kb query":
        chosen = kb_query
    return chosen


class SearchService:
    """Client to interact with Azure Cognitive Search."""

    def __init__(self, endpoint: str, search_key: str, index_name: str, llm: LLM) -> None:
        self.search_client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(search_key),
        )
        self.llm = llm

    def search(self,
               user_query: Optional[str],
               kb_query: Optional[str],
               search_settings: Optional[SearchSettings]) -> SearchTransaction:
        """
        Performs a search request to Azure Cognitive Search using the query strings and search settings provided.

        For vector search, it uses the Open AI embedding service to vectorize the query string first.
        """
        vector_query = choose_query_option(search_settings.vector_search, user_query, kb_query)
        text_query = choose_query_option(search_settings.text_search, user_query, kb_query)

        vectors = []
        embedding_completion = None
        if search_settings.vector_fields and vector_query is not None:
            embedding_completion = self.llm.embedding(vector_query)
            vectors = [Vector(
                value=embedding_completion["data"][0]["embedding"],
                fields=",".join(search_settings.vector_fields),
                k=search_settings.k,
            )
            ]

        query_type = None
        query_language = None
        semantic_configuration_name = None
        if search_settings.semantic_search and search_settings.semantic_configuration_name:
            query_type = "semantic"
            query_language = "en-US"
            semantic_configuration_name = search_settings.semantic_configuration_name

        scoring_profile = None
        if search_settings.scoring_profile_name:
            scoring_profile = search_settings.scoring_profile_name

        documents = self.search_client.search(search_text=text_query, vectors=vectors, query_language=query_language,
                                              semantic_configuration_name=semantic_configuration_name,
                                              query_type=query_type, top=search_settings.top,
                                              scoring_profile=scoring_profile)
        return SearchTransaction(documents=list(documents), embedding_completion=embedding_completion,
                                 text_query=text_query, vector_query=vector_query)
