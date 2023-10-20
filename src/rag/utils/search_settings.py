from typing import Optional, List


class SearchSettings:
    """Class that represents search settings for Azure Cognitive Search request."""

    def __init__(self,
                 vector_search: str,
                 text_search: str,
                 semantic_search: bool,
                 semantic_configuration_name: Optional[str],
                 top: int,
                 vector_fields: Optional[List[str]],
                 k: int,
                 scoring_profile_name: Optional[str],
                 temperature_kb_query: float
                 ) -> None:
        self.vector_search = vector_search
        self.text_search = text_search
        self.semantic_search = semantic_search
        self.semantic_configuration_name = semantic_configuration_name
        self.top = top
        self.vector_fields = vector_fields
        self.k = k
        self.scoring_profile_name = scoring_profile_name
        self.temperature_kb_query = temperature_kb_query

    def requires_kb_query(self) -> bool:
        return self.vector_search == "kb query" or self.text_search == "kb query"

    def invalid(self) -> bool:
        return self.text_search == "off" and (self.vector_search == "off" or not self.vector_fields)
