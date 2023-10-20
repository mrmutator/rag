from ..utils.prompt_pair import PromptPair

REPHRASE_USER_QUERY_SYSTEM_PROMPT = """Your task is to only rephrase user inputs given the conversation history between the user and an AI assistant.
When you rephrase the user intent you add additional context from the history if necessary and you remove irrelevant information or smalltalk.
It is important that the rephrased user input contains all information from the previous conversation that is required to answer the question when the user input is forwarded to someone else that does not know the conversation history.
Answer only with the rephrased user input from the perspective of the user in a json object with the key 'rephrased'.


"""

REPHRASE_USER_QUERY_USER_PROMPT = """Chat history: ```{chat_history}```

User query: ```{user_query}```"""

KNOWLEDGE_BASE_QUERY_SYSTEM_PROMPT = """Your only task is to generate a search query string that will be used to search a knowledge base.
Create the search query string based on the user input by removing irrelevant words keeping in mind that the search string is used for keyword search.
NEVER try to answer the question.
Return only the search query string in a json object using the key 'search_expression'.
Example: 'What is the capital of Canada?' => {"search_expression": "capital canada"}."""

KNOWLEDGE_BASE_QUERY_USER_PROMPT = """query to convert into a search expression:

```{query}````"""

RAG_SYSTEM_PROMPT = """You're a helpful assistant.
Please answer the user's question using only information you can find in the context (documents) below (json format).
If the user's question is unrelated to the information in the context or you do not find the answer in the context, simply say "I don't know".
At the end of your response, provide the full document source where you found the answer (you can find it in the json context below) in square brackets.
Documents for context: {context}
"""

REPHRASE_USER_QUERY_PROMPT_NAME = "Rephrase User Query"
KNOWLEDGE_BASE_QUERY_PROMPT_NAME = "Knowledge Base Query"
RAG_PROMPT_NAME = "RAG"

DEFAULT_PROMPTS = {
    REPHRASE_USER_QUERY_PROMPT_NAME: PromptPair(REPHRASE_USER_QUERY_SYSTEM_PROMPT, REPHRASE_USER_QUERY_USER_PROMPT),
    KNOWLEDGE_BASE_QUERY_PROMPT_NAME: PromptPair(KNOWLEDGE_BASE_QUERY_SYSTEM_PROMPT, KNOWLEDGE_BASE_QUERY_USER_PROMPT),
    RAG_PROMPT_NAME: PromptPair(RAG_SYSTEM_PROMPT)
}
