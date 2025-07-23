import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from langchain_community.chat_models import AzureChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_community.tools import StructuredTool  # Import StructuredTool
from langchain.memory import ConversationBufferMemory
from typing import Optional
from pydantic import BaseModel, Field
from langchain_community.callbacks import get_openai_callback

# LLM setup
llm = AzureChatOpenAI(
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_GPT4O_DEPLOYMENT_NAME"),
    temperature=0.5,
    max_tokens=4096
)

from elasticsearch import Elasticsearch
# Elasticsearch Setup
try:
    # Elasticsearch setup
    es_endpoint = os.environ.get("ELASTIC_CLOUD_ENDPOINT")
    print("Elasticsearch endpoint: ", es_endpoint)
    es_client = Elasticsearch(
        es_endpoint,
        api_key=os.environ.get("ELASTIC_API_KEY")
    )
except Exception as e:
    es_client = None

# Define a function to check ES status
def es_ping(*args, **kwargs):
    if es_client is None:
        return "ES client is not initialized."
    else:
        try:
            if es_client.ping():
                return "ES ping returning True, ES is connected."
            else:
                return "ES is not connected."
        except Exception as e:
            return f"Error pinging ES: {e}"

es_status_tool = Tool(
    name="ES_Status",
    func=es_ping,
    description="Checks if Elasticsearch is connected.",
)

# Define the RAG search function
def rag_search(query: str, dates: str):
    if es_client is None:
        return "ES client is not initialized."
    else:
        try:
            # Build the Elasticsearch query
            must_clauses = []

            # If dates are provided, parse and include in query
            if dates:
                # Dates must be in format 'YYYY-MM-DD' or 'YYYY-MM-DD to YYYY-MM-DD'
                date_parts = dates.strip().split(' to ')
                if len(date_parts) == 1:
                    # Single date
                    start_date = date_parts[0]
                    end_date = date_parts[0]
                elif len(date_parts) == 2:
                    start_date = date_parts[0]
                    end_date = date_parts[1]
                else:
                    return "Invalid date format. Please use YYYY-MM-DD or YYYY-MM-DD to YYYY-MM-DD."

                date_range = {
                    "range": { 
                        "created_at": {
                            "gte": start_date,
                            "lte": end_date
                        }
                    }
                }
                must_clauses.append(date_range)

            # Add the main query clause
            main_query = {
                        "sparse_vector": {
                            "inference_id": ".elser_model_2_linux-x86_64",
                            "field": "title_elser_embedding",
                            "query": query
                        }
                    }
            must_clauses.append(main_query)

            es_query = {
                "_source": ["description", "title_x", "created_at"],
                "query": {
                    "bool": {
                        "must": must_clauses
                    }
                },
                "size": 3
            }

            response = es_client.search(index="gear_products_2", body=es_query)
            hits = response["hits"]["hits"]
            if not hits:
                return "No products found for your query."
            result_docs = []
            for hit in hits:
                source = hit["_source"]
                title = source.get("title_x", "No Title")
                text_content = source.get("description", "")
                date = source.get("created_at", "No Date")
                doc = f"Title: {title}\nDate: {date}\n{text_content}\n"
                result_docs.append(doc)
            return "\n".join(result_docs)
        except Exception as e:
            return f"Error during RAG search: {e}"
        
class RagSearchInput(BaseModel):
    query: str = Field(..., description="The search query for the product catalog.")
    dates: str = Field(
        ...,
        description="Date or date range for filtering results. Specify in format YYYY-MM-DD or YYYY-MM-DD to YYYY-MM-DD."
    )

# Define the RAG search tool using StructuredTool
rag_search_tool = StructuredTool(
    name="RAG_Search",
    func=rag_search,
    description=(
        "Use this tool to search for products in the product catalog. "
        "**Input must include a search query and a date or date range.** "
        "Dates must be specified in one of two formats.  The first allowed date format is: YYYY-MM-DD"
        "The second allowed date format is: YYYY-MM-DD to YYYY-MM-DD"
    ),
    args_schema=RagSearchInput
)        

tools = [es_status_tool, rag_search_tool]

# Initialize memory to keep track of the conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize agent
agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
    system_message="""
    You are an AI assistant that helps with questions about products in an online shopping catalog. Be concise, sharp, to the point, and respond in one paragraph.
    You have access to the following tools:
    - **ES_Status**: Checks if Elasticsearch is connected.
    - **RAG_Search**: Use this to search for information in the product catalog. **Input must include a search query and a date or date range.** Dates must be specified in this format YYYY-MM-DD or YYYY-MM-DD to YYYY-MM-DD.
    **Important Instructions:**
    - **Extract dates or date ranges from the user's question.**
    - **If the user does not provide a date or date range, politely ask them to provide one before proceeding.**
    When you decide to use a tool, use the following format *exactly*:
    Thought: [Your thought process about what you need to do next]
    Action: [The action to take, should be one of [ES_Status, RAG_Search]]
    Action Input: {"query": "the search query", "dates": "the date or date range"}
    If you receive an observation after an action, you should consider it and then decide your next step. If you have enough information to answer the user's question, respond with:
    Thought: [Your thought process]
    Assistant: [Your final answer to the user]
    **Examples:**
    - **User's Question:** "Tell me about products related to sports teams added since 2020."
      Thought: I need to search the product catalog for products dated 2020 or later, related to sports teams.
      Action: RAG_Search
      Action Input: {"query" : "sports teams", "dates" : "2020-01-01 to present"}
    - **User's Question:** "are there any products for fishing"
      Thought: The user didn't specify a date. I should ask for a date range.
      Assistant: Could you please specify the date or date range for the products you're interested in?
    Always ensure that your output strictly follows one of the above formats, and do not include any additional text or formatting.
    Remember:
    - **Do not** include any text before or after the specified format.
    - **Do not** add extra explanations.
    - **Do not** include markdown, bullet points, or numbered lists unless it is part of the Assistant's final answer.
    Your goal is to assist the user by effectively using the tools when necessary and providing clear and concise answers.
    """
)

# Interactive conversation with the agent
def main():
    with get_openai_callback() as cb: # reference about callback: https://python.langchain.com/docs/integrations/chat/azure_chat_openai/
        print("Welcome to the chat agent. Type 'exit' to quit.")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            response = agent_chain.run(input=user_input)
            print("Assistant:", response)
            print(f"Total Cost (USD): ${format(cb.total_cost, '.6f')}")

if __name__ == "__main__":
    main()

