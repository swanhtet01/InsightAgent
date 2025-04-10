# agent_server.py
from openai import AssistantClient
from insightagent import TOOLS

client = AssistantClient()
client.serve(tools=TOOLS)
