import asyncio
import base64
import os

import logging

from dotenv import load_dotenv


from agent_framework import AgentResponse, Content, Message, WorkflowBuilder, WorkflowViz
from typing import cast
from typing_extensions import Never
from agent_framework.openai import OpenAIChatClient 

load_dotenv(".env")

    
chat_client = OpenAIChatClient(
    base_url=os.environ.get("GITHUB_ENDPOINT"),    
    api_key=os.environ.get("GITHUB_TOKEN"),      
    model_id=os.environ.get("GITHUB_MODEL_ID") ,
)

SalesAgentName = "Sales-Agent"
SalesAgentInstructions = "你是我的家具销售顾问，你能从图片中找出不同的家具元素，并给我一些购买建议,不要给出价格，价格由另一个定价顾问提供。"
    
PriceAgentName = "Price-Agent"
PriceAgentInstructions = """您是一位家具定价专家和预算顾问。您的职责包括：
        1. 分析家具产品，并根据质量、品牌和市场标准提供合理的价格范围
        2. 将价格细分为各个家具单品
        3. 提供经济实惠的替代方案和高端选择
        4. 考虑不同的价格档次（经济型、中档、高端）
        5. 提供房间布置的预估总成本
        6. 提供最佳优惠购买渠道和购物建议
        7. 将运费、组装费和配件费等额外成本考虑在内
        8. 提供季节性价格信息和最佳购买时机
        请始终以清晰的价格细分和定价依据解释的形式回复。"""

QuoteAgentName = "Quote-Agent"
QuoteAgentInstructions = """您是一名助理，负责创建家具采购报价单。

1. 创建一份结构清晰的报价单，包含以下内容：

2. 标题页，包含文档标题、日期和客户名称

3. 引言，概述文档用途

4. 汇总部分，包含预估总成本和建议

5. 使用清晰的标题、项目符号和表格，以便于阅读

6. 所有报价均以 Markdown 格式呈现。
"""    

sales_agent   = chat_client.as_agent(
        instructions=(
           SalesAgentInstructions
        ),
        name=SalesAgentName,
    )

price_agent = chat_client.as_agent(
        instructions=(
            PriceAgentInstructions
        ),
        name=PriceAgentName,
    )

quote_agent = chat_client.as_agent(
        instructions=(
            QuoteAgentInstructions
        ),
        name=QuoteAgentName,
    )



workflow = WorkflowBuilder(start_executor=sales_agent).add_edge(sales_agent, price_agent).add_edge(price_agent, quote_agent).build()


image_path = "./files/home.png"
with open(image_path, "rb") as image_file:
    image_b64 = base64.b64encode(image_file.read()).decode()
image_uri = f"data:image/png;base64,{image_b64}"

message = Message(
        role="user",
        contents=[
            Content.from_text(text="请根据图片找出相关的家具，并给出每件家具对应的价格。"),
            Content.from_uri(uri=image_uri, media_type="image/png")
        ]
)

async def main() -> None:
    events = await workflow.run(message)

    outputs = events.get_outputs()
  
    outputs = cast(list[AgentResponse], outputs)
    for output in outputs:
        print(f"{output.messages[0].author_name}: {output.text}\n")
        
  
    print("Final state:", events.get_final_state())


if __name__ == "__main__":
    asyncio.run(main())