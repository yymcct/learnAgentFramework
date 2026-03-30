# Copyright (c) Microsoft. All rights reserved.

import asyncio
import base64
from pydantic import Field
import os
from random import randint
from typing_extensions import Annotated

from openai import AsyncOpenAI

from agent_framework import Agent, Content, Message, tool
from agent_framework.openai import OpenAIChatClient 

from dotenv import load_dotenv
import httpx

load_dotenv()

OPENAI_API_KEY= os.getenv("OPENROUTER_API_KEY")         
OPENAI_CHAT_MODEL_ID= os.getenv("OPENAI_CHAT_MODEL_ID") 




async def main() -> None:
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
        name="FurnitureAgent",
        instructions="你是我的家具销售顾问，你能从图片中找出不同的家具元素，并给我一些购买建议吗？",
    )

    image_path = "./files/home.png"
    with open(image_path, "rb") as image_file:
        image_b64 = base64.b64encode(image_file.read()).decode()
        image_uri = f"data:image/png;base64,{image_b64}"
        message = Message(
            role="user",
            contents=[
                Content.from_text(text="图片中有什么家具元素？"),
                Content.from_uri(uri=image_uri, media_type="image/png"),
            ]
        )

        session = agent.create_session()


        result = await agent.run(message, session=session)
        print(f"Agent: {result}\n")
    
        from utils import print_session
        print_session(session)
 

if __name__ == "__main__":
    asyncio.run(main())