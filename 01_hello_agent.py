# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os

from openai import AsyncOpenAI

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient 

from dotenv import load_dotenv
import httpx

load_dotenv()

 #从.env获取
OPENAI_API_KEY= os.getenv("OPENROUTER_API_KEY")         
OPENAI_CHAT_MODEL_ID= os.getenv("OPENAI_CHAT_MODEL_ID") 

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

    agent = Agent(
        client=client,
        name="HelloAgent",
        instructions="You are a friendly assistant. Keep your answers brief.",
    )
    # </create_agent>

    # <run_agent>
    # Non-streaming: get the complete response at once
    result = await agent.run("What is the capital of France?")
    print(f"Agent: {result}")
    # </run_agent>

    # <run_agent_streaming>
    # Streaming: receive tokens as they are generated
    print("Agent (streaming): ", end="", flush=True)
    async for chunk in agent.run("Tell me a one-sentence fun fact.", stream=True):
        if chunk.text:
            print(chunk.text, end="", flush=True)
    print()
    # </run_agent_streaming>


if __name__ == "__main__":
    asyncio.run(main())