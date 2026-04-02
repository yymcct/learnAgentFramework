import asyncio
import os

import logging

from dotenv import load_dotenv


from agent_framework import AgentResponse, WorkflowBuilder, WorkflowViz
from typing import cast
from typing_extensions import Never
from agent_framework.openai import OpenAIChatClient 

load_dotenv(".env")

import json
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.DEBUG)

# class PrintRequestMiddleware:
#     async def on_chat_options(self, options, next):
#         print("=== 发给模型的完整消息 ===")
#         for msg in options.get("messages", []):
#             print(json.dumps(msg, ensure_ascii=False, indent=2))
#         return await next(options)
    
chat_client = OpenAIChatClient(
    base_url=os.environ.get("GITHUB_ENDPOINT"),    
    api_key=os.environ.get("GITHUB_TOKEN"),      
    model_id=os.environ.get("GITHUB_MODEL_ID") ,
)

REVIEWER_NAME = "Concierge"
REVIEWER_INSTRUCTIONS = """
    You are an are hotel concierge who has opinions about providing the most local and authentic experiences for travelers.
    The goal is to determine if the front desk travel agent has recommended the best non-touristy experience for a traveler.
    If so, state that it is approved.
    If not, provide insight on how to refine the recommendation without using a specific example. 
    """
    
FRONTDESK_NAME = "FrontDesk"
FRONTDESK_INSTRUCTIONS = """
    You are a Front Desk Travel Agent with ten years of experience and are known for brevity as you deal with many customers.
    The goal is to provide the best activities and locations for a traveler to visit.
    Only provide a single recommendation per response.
    You're laser focused on the goal at hand.
    Don't waste time with chit chat.
    Consider suggestions when refining an idea.
    """
    
reviewer_agent   = chat_client.as_agent(
        instructions=(
           REVIEWER_INSTRUCTIONS
        ),
        name=REVIEWER_NAME,
    )

front_desk_agent = chat_client.as_agent(
        instructions=(
            FRONTDESK_INSTRUCTIONS
        ),
        name=FRONTDESK_NAME,
    )

#TODO：front_desk_agent传给reviewer_agent了什么？ reviewer_agent最终发送给模型的是什么
workflow = WorkflowBuilder(start_executor=front_desk_agent).add_edge(front_desk_agent, reviewer_agent).build()


# print("Generating workflow visualization...")
# viz = WorkflowViz(workflow)
# # Print out the mermaid string.
# print("Mermaid string: \n=======")
# print(viz.to_mermaid())
# print("=======")
# # Print out the DiGraph string.
# print("DiGraph string: \n=======")
# print(viz.to_digraph())
# print("=======")
# svg_file = viz.export(format="svg")
# print(f"SVG file saved to: {svg_file}")



async def main() -> None:
    events = await workflow.run("I would like to go to Paris.")

    outputs = events.get_outputs()
    outputs = cast(list[AgentResponse], outputs)
    for output in outputs:
        print(f"{output.messages[0].author_name}: {output.text}\n")

        # Summarize the final run state (e.g., COMPLETED)
    print("Final state:", events.get_final_state())


if __name__ == "__main__":
    asyncio.run(main())