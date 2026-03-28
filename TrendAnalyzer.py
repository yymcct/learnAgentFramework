import asyncio
import os
from typing_extensions import Annotated

from openai import AsyncOpenAI

from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatClient

from dotenv import load_dotenv
import httpx
from tavily import TavilyClient
from pydantic import Field

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_CHAT_MODEL_ID = os.getenv("OPENAI_CHAT_MODEL_ID")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


@tool(approval_mode="never_require")
def tavily_search(
    query: Annotated[str, Field(description="搜索关键词")],
    max_results: Annotated[int, Field(description="返回结果数量 (1-10)")] = 5,
) -> str:
    """使用 Tavily 搜索网页，返回结果摘要。"""
    client = TavilyClient(api_key=TAVILY_API_KEY)
    response = client.search(query=query, max_results=max_results)
    results = response.get("results", [])
    if not results:
        return "No results found."
    output = []
    for i, r in enumerate(results, 1):
        output.append(
            f"{i}. [{r.get('title', 'No title')}]({r.get('url', '')}): {r.get('content', '')}"
        )
    return "\n".join(output)


REACT_INSTRUCTIONS = """
你是一个专业的热点猎手，采用 ReAct（推理-行动-观察）模式工作，为中文内容平台（知乎、小红书、微博、视频号、公众号）寻找高潜力爆款选题。

## 工作模式
每一轮请严格按照以下格式输出：

**[思考]** 分析当前已知信息，判断下一步需要搜索什么、为什么。
**[行动]** Call: tavily_search  执行搜索（本轮可调用多次，覆盖不同角度）。
**[观察]** 总结搜索结果的关键发现，标注哪些方向有潜力、哪些需要进一步验证。
**[结论]** 本轮阶段性结论，说明下一步计划（若还需迭代）。

## 选题评分维度
- 时效性：是否是近期上升趋势（非过热话题）
- 情绪共鸣：能否触发好奇 / 焦虑 / 共情 / 争议
- 平台适配：是否符合目标平台算法偏好
- 竞争度：同类内容是否已饱和

## 最终输出格式（仅在最后一轮给出）
每个选题包含：
1. 候选标题（2-3 个变体）
2. 平台来源 + 热度趋势（上升 / 稳定 / 高峰）
3. 爆款理由（情绪点 / 用户痛点 / 算法偏好）
4. 推荐切入角度
5. 综合评分（1-10 分）
"""


async def main() -> None:
    topic = "郑州小升初择校"

    client = OpenAIChatClient(
        model_id=OPENAI_CHAT_MODEL_ID,
        async_client=AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            http_client=httpx.AsyncClient(proxy="socks5://127.0.0.1:7890"),
        ),
    )

    agent = Agent(
        client=client,
        name="TrendAnalyzer",
        instructions=REACT_INSTRUCTIONS,
        tools=[tavily_search],
    )
    session = agent.create_session()

    # ── ReAct 多轮迭代 ──────────────────────────────────────────────

    # Round 1: 广度扫描 —— 发现各平台热点方向
    print("=" * 60)
    print("[Round 1] 广度发现：多平台热点扫描")
    print("=" * 60)
    result = await agent.run(
        f"请针对主题「{topic}」，在知乎热榜、微博热搜、小红书热门上分别搜索相关讨论。"
        "按 ReAct 格式：先思考搜索策略，再执行搜索，最后观察并列出 3-5 个初步有潜力的子方向。",
        session=session,
    )
    print(result)
    print()

    # Round 2: 深度验证 —— 交叉搜索潜力方向的真实案例和用户情绪
    print("=" * 60)
    print("[Round 2] 深度验证：交叉搜索潜力方向")
    print("=" * 60)
    result = await agent.run(
        "根据上一轮发现的潜力方向，针对其中最有爆款潜力的 2-3 个方向，"
        "分别搜索近期真实案例、爆款文章标题样本、用户讨论情绪。"
        "按 ReAct 格式：先推理应验证什么，再搜索，再观察讨论热度与情绪强度。",
        session=session,
    )
    print(result)
    print()

    # Round 3: 时效性与竞争度评估
    print("=" * 60)
    print("[Round 3] 时效性与竞争度评估")
    print("=" * 60)
    result = await agent.run(
        "对上一轮验证过的子方向，搜索它们近一个月内的最新动态与内容饱和度。"
        "判断哪些话题仍处于上升期、哪些已过热。"
        "按 ReAct 格式：先分析评估思路，再搜索，再输出时效性与竞争度评分。",
        session=session,
    )
    print(result)
    print()

    # Round 4: 最终综合推荐
    print("=" * 60)
    print("[Round 4] 最终选题推荐")
    print("=" * 60)
    result = await agent.run(
        f"综合前三轮的所有搜索结果与分析，给出针对主题「{topic}」的最终 Top 5 选题推荐。"
        "严格按照最终输出格式，包含候选标题变体、平台来源、热度趋势、爆款理由、切入角度和综合评分（1-10 分）。",
        session=session,
    )
    print(result)
    print()

    # 友好打印 session 对话内容
    from utils import print_session
    print_session(session)
    
if __name__ == "__main__":
    asyncio.run(main())
