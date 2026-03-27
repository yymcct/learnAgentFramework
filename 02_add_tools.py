# Copyright (c) Microsoft. All rights reserved.

import asyncio
from pydantic import Field
import os
from random import randint
from typing_extensions import Annotated

from openai import AsyncOpenAI

from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatClient 

from dotenv import load_dotenv
import httpx

load_dotenv()

OPENAI_API_KEY= os.getenv("OPENROUTER_API_KEY")         
OPENAI_CHAT_MODEL_ID= os.getenv("OPENAI_CHAT_MODEL_ID") 

# <define_tool>
# NOTE: approval_mode="never_require" is for sample brevity.
# Use "always_require" in production for user confirmation before tool execution.
@tool(approval_mode="never_require")
def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """Get the weather for a given location."""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


async def main() -> None:
    # <create_agent>
    client = OpenAIChatClient(
        model_id=OPENAI_CHAT_MODEL_ID,
        async_client= AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            http_client=httpx.AsyncClient(proxy="socks5://127.0.0.1:7890"),
        ),
    )
    #TODO : 我如何查看agent和model的对话内容？我想看看agent是如何调用工具的。
    agent = Agent(
        client=client,
        name="WeatherAgent",
        instructions="You are a helpful weather agent. Use the get_weather tool to answer questions.",
        tools=[get_weather],
    )
    # </create_agent>

    # Create a session to maintain conversation history
    session = agent.create_session()

    # <run_agent>
    # Non-streaming: get the complete response at once
    result = await agent.run("What's the weather like in Seattle?", session=session)
    print(f"Agent: {result}")
    
    print(f"Session: {session.to_dict()}")
    
    # </run_agent>




if __name__ == "__main__":
    asyncio.run(main())