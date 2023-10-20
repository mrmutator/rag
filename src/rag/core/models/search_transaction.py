from typing import Optional, Dict, Any, List


class SearchTransaction:
    """
    Represents transaction of a search request to the knowledge database.

    Class also includes information about the embedding request if available.
    """

    def __init__(self, documents: List[Any], embedding_completion: Optional[Dict[str, Any]], text_query: Optional[str],
                 vector_query: Optional[str]) -> None:
        self.documents = documents
        self.embedding_completion = embedding_completion
        self.text_query = text_query
        self.vector_query = vector_query

    def get_tokens(self) -> int:
        return self.embedding_completion['usage']['total_tokens'] if self.embedding_completion else 0
