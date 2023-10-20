import json
from typing import Any, List


class CompletionTransaction:
    """Represents transaction of a completion request to a generative LLM."""

    def __init__(self, completion: Any, messages: List[Any]) -> None:
        self.completion = completion
        self.messages = messages
        self.json_key = None
        self.name = None  # will be set later

    def set_name(self, name: str) -> None:
        self.name = name

    def get_response(self) -> str:
        response = self.completion.choices[0].message.content
        if self.json_key is None:
            return response
        return json.loads(response)[self.json_key]

    def set_json_key(self, json_key: str) -> None:
        self.json_key = json_key

    def get_tokens(self) -> int:
        return self.completion['usage']['total_tokens']
