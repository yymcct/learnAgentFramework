import asyncio
import base64
import os

import logging

from agent_framework_orchestrations import ConcurrentBuilder
from dotenv import load_dotenv


from agent_framework import AgentResponse, Content, Message, WorkflowBuilder, WorkflowViz
from typing import Any, Any, cast
from typing_extensions import Never
from agent_framework.openai import OpenAIChatClient 


# 构建并发智能体

load_dotenv(dotenv_path=".env")


chat_client = OpenAIChatClient(
    base_url=os.environ.get("GITHUB_ENDPOINT"),    
    api_key=os.environ.get("GITHUB_TOKEN"),      
    model_id=os.environ.get("GITHUB_MODEL_ID"),
)

ResearcherAgentName = "Researcher-Agent"
ResearcherAgentInstructions = """
    你是我的旅行研究员，与我一起分析目的地，
    列出相关景点，并为每个景点制定详细的游览计划。
    """
    
PlanAgentName = "Plan-Agent"
PlanAgentInstructions = """
    你是我的旅行规划师，与我一起根据研究人员的发现制定详细的旅行计划。
    """

research_agent   = chat_client.as_agent(
        instructions=(
           ResearcherAgentInstructions
        ),
        name=ResearcherAgentName,
    )

plan_agent = chat_client.as_agent(
        instructions=(
            PlanAgentInstructions
        ),
        name=PlanAgentName,
    )



workflow = ConcurrentBuilder(participants=[research_agent, plan_agent]).build()


print("Generating workflow visualization...")
viz = WorkflowViz(workflow)
print("Mermaid string: \n=======")
print(viz.to_mermaid())
print("=======")
print("DiGraph string: \n=======")
print(viz.to_digraph())
print("=======")
svg_file = viz.export(format="svg")
print(f"SVG file saved to: {svg_file}")


#TODO 并行执行的时候，两个智能体进行通信么？如果不进行通信，那么它们的输出结果是独立的，还是会有某种形式的交互？如果有交互，那么它们是如何协调的？这些都是需要考虑的问题。
async def main() -> None:
    events = await workflow.run("计划十二月去昆明旅行。")

    outputs = events.get_outputs()
  
    for output in outputs:
        messages: list[Message] | Any = output
        for i, msg in enumerate(messages, start=1):
            name = msg.author_name if msg.author_name else "user"
            print(f"{'-' * 60}\n\n{i:02d} [{name}]:\n{msg.text}")
        
  
    print("Final state:", events.get_final_state())


if __name__ == "__main__":
    asyncio.run(main())