import asyncio
import os

from agent_framework.openai import OpenAIChatClient, OpenAIResponsesClient
from agent_framework import MCPStreamableHTTPTool
from dotenv import load_dotenv

os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
os.environ.pop("ALL_PROXY", None)
os.environ.pop("all_proxy", None)

load_dotenv()

client = OpenAIChatClient(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    model_id=os.environ.get("OPENAI_CHAT_MODEL_ID"),
)

agent = client.as_agent(
    name="MCPAgent",
    instructions='''
    您是一位乐于助人的助手，可以帮助解答有关微软文档的问题。
    '''
    )

session = agent.create_session()

async def userProxy():
    async with MCPStreamableHTTPTool(
        name="Microsoft Learn MCP",
        url="https://learn.microsoft.com/api/mcp",
    ) as mcp_tool:
        
        await mcp_tool.load_tools() # 连接成功后工具已自动加载，直接访问即可
        for tool in mcp_tool.functions:
            print(f"{tool.name}: {tool.description}")
        
        result1 = await agent.run("如何使用 az cli 创建 Azure 存储帐户？", tools=mcp_tool, session=session)
        print(f"{agent.name}: {result1}\n")
        
        print("\n=======================================\n")

        query2 = "什么是 Microsoft Agent Framework?"
        print(f"User: {query2}")
        result2 = await agent.run(query2, tools=mcp_tool, session=session)
        print(f"{agent.name}: {result2}\n")
        
        
        from utils import print_session
        print_session(session)

if __name__ == "__main__":
    asyncio.run(userProxy())