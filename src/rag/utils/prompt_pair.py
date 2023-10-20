from typing import Optional


class PromptPair:
    """Class that represents a prompt pair, i. e. a system prompt and a user prompt for the Open AI Chat API."""

    def __init__(self, system_prompt: str, user_prompt: Optional[str] = None) -> None:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
