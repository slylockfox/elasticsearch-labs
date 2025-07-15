import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from langchain_community.chat_models import AzureChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_community.tools import StructuredTool  # Import StructuredTool
from langchain.memory import ConversationBufferMemory
from typing import Optional
from pydantic import BaseModel, Field

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
    name="ES Status",
    func=es_ping,
    description="Checks if Elasticsearch is connected.",
)

tools = [es_status_tool]

# Initialize memory to keep track of the conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize agent
agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
)

# Interactive conversation with the agent
def main():
    print("Welcome to the chat agent. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        response = agent_chain.run(input=user_input)
        print("Assistant:", response)

if __name__ == "__main__":
    main()

