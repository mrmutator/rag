from typing import Any, List

from .completion_transaction import CompletionTransaction
from .search_transaction import SearchTransaction


class ChatTransaction:
    """Class used to represent an entire interaction between the user and the assistant."""

    def __init__(self, query: str) -> None:
        self.query = query
        self.completion_transactions: List[CompletionTransaction] = []
        self.search_transactions: List[SearchTransaction] = []
        self.response = None  # will be set in setter

    def set_response(self, response: str) -> None:
        self.response = response

    def add_completion_transaction(self, completion_transaction: CompletionTransaction) -> None:
        self.completion_transactions.append(completion_transaction)

    def add_search_transaction(self, search_transaction: SearchTransaction) -> None:
        self.search_transactions.append(search_transaction)

    def get_completion_tokens(self) -> int:
        return sum([t.get_tokens() for t in self.completion_transactions])

    def get_embedding_tokens(self) -> int:
        return sum([t.get_tokens() for t in self.search_transactions])

    def get_documents(self) -> List[Any]:
        return self.search_transactions[0].documents
