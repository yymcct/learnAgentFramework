

import asyncio
import os
from typing import Any

from agent_framework import Agent, AgentSession, BaseContextProvider, InMemoryHistoryProvider, SessionContext
from dotenv import load_dotenv
import httpx
from openai import AsyncOpenAI

from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatClient 
from azure.identity import AzureCliCredential

"""
Agent Memory with Context Providers and Session State

Context providers inject dynamic context into each agent call. This sample
shows a provider that stores the user's name in session state and personalizes
responses — the name persists across turns via the session.
"""

load_dotenv()

OPENAI_API_KEY= os.getenv("OPENROUTER_API_KEY")         
OPENAI_CHAT_MODEL_ID= os.getenv("OPENAI_CHAT_MODEL_ID") 

# extend_instructions 是如何被注入到
# <context_provider>
class UserMemoryProvider(BaseContextProvider):
    """A context provider that remembers user info in session state."""

    DEFAULT_SOURCE_ID = "user_memory"

    def __init__(self):
        super().__init__(self.DEFAULT_SOURCE_ID)

    async def before_run(
        self,
        *,
        agent: Any,
        session: AgentSession | None,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        """Inject personalization instructions based on stored user info."""
        user_name = state.get("user_name")
        if user_name:
            context.extend_instructions(
                self.source_id,
                f"The user's name is {user_name}. Always address them by name.",
            )
        else:
            context.extend_instructions(
                self.source_id,
                "You don't know the user's name yet. Ask for it politely.",
            )

    async def after_run(
        self,
        *,
        agent: Any,
        session: AgentSession | None,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        """Extract and store user info in session state after each call."""
        for msg in context.input_messages:
            text = msg.text if hasattr(msg, "text") else ""
            if isinstance(text, str) and "my name is" in text.lower():
                state["user_name"] = text.lower().split("my name is")[-1].strip().split()[0].capitalize()


# </context_provider>


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
        name="MemoryAgent",
        instructions="You are a friendly assistant.",
        context_providers=[UserMemoryProvider(),InMemoryHistoryProvider()],
    )
    # </create_agent>

    # <run_with_memory>
    session = agent.create_session()

    
    result = await agent.run("Hello! What's the square root of 9?", session=session)
    print(f"Agent: {result}\n")

   
    result = await agent.run("My name is Alice", session=session)
    print(f"Agent: {result}\n")

   
    result = await agent.run("What is 2 + 2?", session=session)
    print(f"Agent: {result}\n")

   
    provider_state = session.state.get("user_memory", {})
    print(f"[Session State] Stored user name: {provider_state.get('user_name')}")
    # </run_with_memory>

    from utils import print_session
    print_session(session)
    
    print(f"session: {session.to_dict()}")
    
if __name__ == "__main__":
    asyncio.run(main())