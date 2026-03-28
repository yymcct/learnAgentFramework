import asyncio
import os

from openai import AsyncOpenAI

from agent_framework import Agent
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
        name="HelloAgen",
        instructions='''
        你是一个严格的用户代理，负责发起爆款文章生成任务、确认选题、审批最终文章，并决定是否继续迭代。
        你的风格：简洁、专业、果断。只在必要时介入，不要主动生成内容。
        当前任务：帮助用户生成一篇适合知乎/小红书/微博/公众号的爆款文章。
        当收到最终交付包时，请仔细阅读并给出“通过”或“继续迭代第X轮 + 具体要求”的反馈。
        ''',
    )


    result = await agent.run("请发起一个爆款文章生成任务，主题是“AI在医疗领域的应用”，并确认选题。")
    print(f"Agent: {result}")



if __name__ == "__main__":
    asyncio.run(main())