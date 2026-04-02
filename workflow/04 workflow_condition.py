import asyncio
import base64
import os

import logging

from agent_framework_azure_ai import AzureAIAgentClient, AzureAIAgentsProvider
from azure.identity import AzureCliCredential
from azure.identity import AzureCliCredential
from dotenv import load_dotenv


from agent_framework import (
    AgentExecutor,
    AgentExecutorRequest,
    AgentExecutorResponse,
    AgentExecutorResponse,
    AgentResponse,
    Content,
    Message,
    WorkflowBuilder,
    WorkflowContext,
    WorkflowEvent,
    WorkflowViz,
    executor,
    tool,
)
from typing import Annotated, cast
from typing_extensions import Never
from agent_framework.openai import OpenAIChatClient
from pydantic import BaseModel, Field
from typing_extensions import Literal
from dataclasses import dataclass
from tavily import TavilyClient

# 条件工作流模式

load_dotenv(".env")


EvangelistInstructions = """
您是一位技术推广者，请创建一份技术教程的初稿。

1. 大纲中的每个知识点都必须包含一个链接。点击链接即可访问与大纲中知识点相关的内容。并在此基础上进行扩展。

2. 每个知识点都必须详细解释。

3. 根据参赛要求重写内容，包括标题、大纲和相应内容。无需完全按照大纲顺序编写。

4. 内容必须超过 200 字。

5. 将草稿输出为 Markdown 格式。将“draft_content”字段设置为草稿内容。

6. 返回包含“draft_content”（字符串）字段的 JSON 数据。

返回的JSON格式如下：
{   
    "draft_content": "原始草稿内容"
}
"""

ContentReviewerInstructions = """
您是一家出版社的内容审核员。您需要检查教程草稿内容是否符合以下要求：

1. 如果草稿内容少于 200 字，则将 'review_result' 设置为 'No'，'reason' 设置为 '内容太短'。如果草稿内容超过 200 字，则将 'review_result' 设置为 'Yes'，'reason' 设置为 '内容良好'。

2. 将 'draft_content' 设置为原始草稿内容。

3. 返回包含 'review_result'（值为 Yes 或 No 之一）、'reason'（字符串）和 'draft_content'（字符串）三个字段的 JSON 数据。

返回的JSON格式如下：
{
    "review_result": "Yes" 或 "No",         
    "reason": "如果 review_result 是 No，这里应该说明原因；如果 review_result 是 Yes，这里可以写 '内容良好'",
    "draft_content": "原始草稿内容"
}
"""

PublisherInstructions = """
您是内容发布者，请将教程草稿内容保存为 Markdown 文件。保存的文件名称会包含当前日期和时间，格式为“年月日时分秒”。请注意，如果日期是 1 到 9，则需要添加 0，例如 20240101123045.md。
"""


OUTLINE_Content = """
# 介绍 AI 代理

## 什么是 AI 代理

https://github.com/microsoft/ai-agents-for-beginners/tree/main/01-intro-to-ai-agents

***注意*** 请勿创建任何示例代码

## 介绍 Azure AI Foundry 代理服务

https://learn.microsoft.com/en-us/azure/ai-foundry/agents/overview

***注意*** 请勿创建任何示例代码

## Microsoft 代理框架

https://github.com/microsoft/agent-framework/tree/main/docs/docs-templates

***注意*** 请勿创建任何示例代码
"""


class EvangelistAgent(BaseModel):
    draft_content: str


class ReviewAgent(BaseModel):
    review_result: Literal["Yes", "No"]
    reason: str
    draft_content: str


class PublisherAgent(BaseModel):
    file_path: str


@dataclass
class ReviewResult:
    review_result: str
    reason: str
    draft_content: str


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


# TODO 如何理解send_message
@executor(id="to_reviewer_result")
async def to_reviewer_result(
    response: AgentExecutorResponse, ctx: WorkflowContext[ReviewResult]
) -> None:
    # Get the last message text from the response
    response_text = response.agent_response.text.strip()
    print(f"Raw response from reviewer agent: {response_text}")

    parsed = ReviewAgent.model_validate_json(response_text)
    await ctx.send_message(
        ReviewResult(
            review_result=parsed.review_result,
            reason=parsed.reason,
            draft_content=parsed.draft_content,
        )
    )


def select_targets(review: ReviewResult, target_ids: list[str]) -> list[str]:
    # Order: [handle_review, submit_to_email_assistant, summarize_email, handle_uncertain]
    handle_review_id, save_draft_id = target_ids
    if review.review_result == "Yes":
        return [save_draft_id]
    else:
        return [handle_review_id]


# TODO 这里的ctx.send_message和ctx.yield_output有什么区别？
@executor(id="handle_review")
async def handle_review(review: ReviewResult, ctx: WorkflowContext[str]) -> None:
    if review.review_result == "No":
        await ctx.yield_output(
            f"Review failed: {review.reason}, please revise the draft."
        )
    else:
        await ctx.send_message(
            AgentExecutorRequest(
                messages=[Message("user", text=review.draft_content)],
                should_respond=True,
            )
        )


@executor(id="save_draft")
async def save_draft(
    review: ReviewResult, ctx: WorkflowContext[AgentExecutorRequest]
) -> None:
    # Only called for long NotSpam emails by selection_func
    await ctx.send_message(
        AgentExecutorRequest(
            messages=[Message("user", text=review.draft_content)], should_respond=True
        )
    )


# from IPython.display import SVG, display, HTML
class DatabaseEvent(WorkflowEvent): ...


async def main() -> None:
    client = OpenAIChatClient(
        base_url=os.environ.get("GITHUB_ENDPOINT"),
        api_key=os.environ.get("GITHUB_TOKEN"),
        model_id=os.environ.get("GITHUB_MODEL_ID"),
    )

    evangelist_agent = AgentExecutor(
        client.as_agent(
            name="evangelist-agent",
            instructions=(EvangelistInstructions),
            tools=[tavily_search],
        ),
        id="evangelist_agent",
    )

    reviewer_agent = AgentExecutor(
        client.as_agent(
            name="reviewer-agent",
            instructions=ContentReviewerInstructions,
        ),
        id="reviewer_agent",
    )

    publisher_agent = AgentExecutor(
        client.as_agent(
            name="publisher-agent",
            instructions=PublisherInstructions,
        ),
        id="publisher_agent",
    )

    workflow = (
        WorkflowBuilder(start_executor=evangelist_agent)
        .add_edge(evangelist_agent, reviewer_agent)
        .add_edge(reviewer_agent, to_reviewer_result)
        .add_multi_selection_edge_group(
            to_reviewer_result,
            [handle_review, save_draft],
            selection_func=select_targets,
        )
        .add_edge(save_draft, publisher_agent)
        .build()
    )

    print("Generating workflow visualization...")
    viz = WorkflowViz(workflow)
    # Print out the mermaid string.
    print("Mermaid string: \n=======")
    print(viz.to_mermaid())
    print("=======")
    # Print out the DiGraph string.
    print("DiGraph string: \n=======")
    print(viz.to_digraph())
    print("=======")
    svg_file = viz.export(format="svg")
    print(f"SVG file saved to: {svg_file}")

    # if svg_file and os.path.exists(svg_file):
    #     try:
    #         display(SVG(filename=svg_file))
    #     except Exception as e:
    #         print(f"⚠️ Direct SVG render failed: {e}. Falling back to raw HTML.")
    #         try:
    #             with open(svg_file, "r", encoding="utf-8") as f:
    #                 svg_text = f.read()
    #             display(HTML(svg_text))
    #         except Exception as inner:
    #             print(f"❌ Fallback HTML render also failed: {inner}")
    # else:
    #     print("❌ SVG file not found. Ensure viz.export(format='svg') ran successfully.")

    task = (
        """
        您是一位布道者，需要根据以下大纲和大纲对应链接中的内容撰写一份草稿。
        草稿完成后，审核员会进行检查。如果符合要求，草稿将提交给发布者并保存为 Markdown 文件；
        否则，需要修改草稿直至符合要求。

        提供的大纲内容及相关链接如下:

        """
        + OUTLINE_Content
    )

    events = await workflow.run(task)

    outputs = events.get_outputs()
    # The outputs of the workflow are whatever the agents produce. So the outputs are expected to be a list
    # of `AgentResponse` from the agents in the workflow.
    outputs = cast(list[AgentResponse], outputs)
    for output in outputs:
        print(f"{output.messages[0].author_name}: {output.text}\n")

        # Summarize the final run state (e.g., COMPLETED)
    print("Final state:", events.get_final_state())


if __name__ == "__main__":
    asyncio.run(main())
