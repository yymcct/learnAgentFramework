import os
from agent_framework.openai import OpenAIResponsesClient
from dotenv import load_dotenv

load_dotenv()

client = OpenAIResponsesClient(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    model_id=os.environ.get("OPENAI_CHAT_MODEL_ID"),
)
