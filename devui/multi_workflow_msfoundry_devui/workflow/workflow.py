"""Conditional Workflow for Content Review with Azure AI Foundry Agents"""

import os
from dataclasses import dataclass
import re
from typing import Annotated

from dotenv import load_dotenv

from agent_framework import (
    AgentExecutor,
    AgentExecutorRequest,
    AgentExecutorResponse,
    Message,
    WorkflowBuilder,
    WorkflowContext,
    executor,
    tool,
)

from evangelist_agent import EVANGELIST_NAME, EVANGELIST_INSTRUCTIONS
from contentreview_agent import ReviewAgent, REVIEWER_NAME, REVIEWER_INSTRUCTIONS
from publisher_agent import PUBLISHER_NAME, PUBLISHER_INSTRUCTIONS
from tavily import TavilyClient
from pydantic import BaseModel, Field
from agent_framework.openai import OpenAIChatClient


env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
load_dotenv(env_path)

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


@dataclass
class ReviewResult:
    """Data class to hold review results"""
    review_result: str
    reason: str
    draft_content: str


@executor(id="to_reviewer_result")
async def to_reviewer_result(
    response: AgentExecutorResponse, 
    ctx: WorkflowContext[ReviewResult]
) -> None:
    """Convert reviewer agent response to structured format"""
    response_text = response.agent_response.text.strip()
    print(f"🔍 [Workflow] Raw response from reviewer agent: {response_text}")
    response_text = re.sub(r"^```json|^```|```$", "", response_text, flags=re.MULTILINE).strip()
    parsed = ReviewAgent.model_validate_json(response_text)
    await ctx.send_message(
        ReviewResult(
            review_result=parsed.review_result,
            reason=parsed.reason,
            draft_content=parsed.draft_content,
        )
    )


def select_targets(review: ReviewResult, target_ids: list[str]) -> list[str]:
    """
    Select workflow path based on review result
    
    Args:
        review: The review result containing decision
        target_ids: List of [handle_review_id, save_draft_id]
    
    Returns:
        List containing the selected target executor ID
    """
    handle_review_id, save_draft_id = target_ids
    if review.review_result == "Yes":
        print(f"✅ [Workflow] Review passed - routing to save_draft")
        return [save_draft_id]
    else:
        print(f"❌ [Workflow] Review failed - routing to handle_review")
        return [handle_review_id]


@executor(id="handle_review")
async def handle_review(review: ReviewResult, ctx: WorkflowContext[str]) -> None:
    """Handle review failures"""
    if review.review_result == "No":
        message = f"Review failed: {review.reason}, please revise the draft."
        print(f"⚠️ [Workflow] {message}")
        await ctx.yield_output(message)
    else:
        await ctx.send_message(
            AgentExecutorRequest(
                messages=[Message("user", text=review.draft_content)], 
                should_respond=True
            )
        )


@executor(id="save_draft")
async def save_draft(review: ReviewResult, ctx: WorkflowContext[AgentExecutorRequest]) -> None:
    """Save draft content by sending to publisher agent"""
    # Only called for approved drafts by selection_func
    await ctx.send_message(
        AgentExecutorRequest(
            messages=[Message("user", text=review.draft_content)], 
            should_respond=True
        )
    )


# Keep provider/credential alive — they must not be closed while DevUI serves.
# main.py runs create_workflow() and uvicorn in a single asyncio.run(),
# so the references here keep the HTTP sessions alive for the entire process.

_client = None


async def create_workflow():
    """Create the conditional workflow with Azure AI Foundry agents.

    The credential, provider, and client are stored as module globals so they
    remain alive for the duration of the process. main.py ensures that
    create_workflow() and uvicorn run in the same event loop.
    """
    global  _client

    _client = OpenAIChatClient(
        base_url=os.environ.get("GITHUB_ENDPOINT"),
        api_key=os.environ.get("GITHUB_TOKEN"),
        model_id=os.environ.get("GITHUB_MODEL_ID"),
    )

    # Create evangelist agent with Bing Search
    evangelist_agent_obj =  _client.as_agent(
        name=EVANGELIST_NAME,
        instructions=EVANGELIST_INSTRUCTIONS,
        tools=[tavily_search],
    )
    evangelist_executor = AgentExecutor(evangelist_agent_obj, id="evangelist_agent")
    
    # Create reviewer agent
    reviewer_agent_obj =  _client.as_agent(
        name=REVIEWER_NAME,
        instructions=REVIEWER_INSTRUCTIONS,
    )
    reviewer_executor = AgentExecutor(reviewer_agent_obj, id="reviewer_agent")
    
    # Create publisher agent with Code Interpreter
    publisher_agent_obj =  _client.as_agent(
        name=PUBLISHER_NAME,
        instructions=PUBLISHER_INSTRUCTIONS,
    )
    publisher_executor = AgentExecutor(publisher_agent_obj, id="publisher_agent")

    # Build the conditional workflow
    workflow = (
        WorkflowBuilder(start_executor=evangelist_executor)
        .add_edge(evangelist_executor, reviewer_executor)
        .add_edge(reviewer_executor, to_reviewer_result)
        .add_multi_selection_edge_group(
            to_reviewer_result,
            [handle_review, save_draft],
            selection_func=select_targets,
        )
        .add_edge(save_draft, publisher_executor)
        .build()
    )
    
    return workflow

