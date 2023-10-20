# RAG - Retrieval Augmented Generation

This is a Streamlit application that allows setting up and testing a RAG application with Azure Cognitive Search and Azure Open AI.

## Setup
TBD

## Running the application

Run the application with streamlit using the following command:

```python -m streamlit run app.py```

The following environment variables need to be defined:

| Variable name                      | Description                                          |
|------------------------------------|------------------------------------------------------|
| AZURE_COGNITIVE_SEARCH_ENDPOINT    | Endpoint of the Azure Cognitive Search service       |
| AZURE_COGNITIVE_SEARCH_INDEX_NAME  | Name of the Azure Cognitive Search index             |
| AZURE_COGNITIVE_SEARCH_KEY         | Search key of the Azure Cognitive Search service     |
| AZURE_OPEN_AI_ENDPOINT             | Endpoint of the Azure Open AI service                |
| AZURE_OPEN_AI_KEY                  | API key of the Azure Open AI service                 |
| AZURE_OPEN_AI_EMBEDDING_DEPLOYMENT | Name of the Azure Open AI embedding model deployment |
| AZURE_OPEN_AI_CHAT_DEPLOYMENT      | Name of the Azure Open AI Chat model deployment      |