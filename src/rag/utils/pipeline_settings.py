class PipelineSettings:
    """Class that represents settings for the RAG pipeline."""

    def __init__(self, num_history: int, input_summarization: bool, input_summarization_temperature: float,
                 rag_temperature: float) -> None:
        self.num_history = num_history
        self.input_summarization = input_summarization
        self.input_summarization_temperature = input_summarization_temperature
        self.rag_temperature = rag_temperature
