# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os

from agent_framework.openai import OpenAIChatClient, OpenAIResponsesClient
from dotenv import load_dotenv
import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel





class OutputStruct(BaseModel):
    """A structured output for testing purposes."""
    city: str
    description: str
    
OPENAI_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_CHAT_MODEL_ID = os.getenv("OPENAI_CHAT_MODEL_ID")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


from myTeam.openai_client import client

async def non_streaming_example() -> None:
    print("=== Non-streaming example ===")

    agent = client.as_agent(
        name="CityAgent",
        instructions="You are a helpful agent that describes cities in a structured format.",
        default_options={"response_format": OutputStruct}
    )

    query = "Tell me about Paris, France"
    print(f"User: {query}")

    result = await agent.run(query,options={"response_format": OutputStruct})

    if structured_data := result.value:
        print("Structured Output Agent:")
        print(f"City: {structured_data.city}")
        print(f"Description: {structured_data.description}")
    else:
        print(f"Failed to parse response: {result.text}")


async def main() -> None:
    print("=== OpenAI Responses Agent with Structured Output ===")
    await non_streaming_example()
 


if __name__ == "__main__":
    asyncio.run(main())