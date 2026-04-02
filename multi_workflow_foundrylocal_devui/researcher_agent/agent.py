"""Research / expansion agent for FoundryLocal workflow.

Consumes the structured plan (topic + outline) from the planning agent
and produces a first full draft. This is analogous to the evangelist
agent in the ghmodel example but with a different upstream signal.
"""

import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
load_dotenv(env_path)
print(f"env_path: {env_path}")

print(f"1os.environ.get('GITHUB_ENDPOINT'): {os.environ.get('GITHUB_ENDPOINT')}")
print(f"1os.environ.get('GITHUB_MODEL_ID'): {os.environ.get('GITHUB_MODEL_ID')}")
print(f"1os.environ.get('GITHUB_TOKEN'): {os.environ.get('GITHUB_TOKEN')}")

try:
    from agent_framework.openai import OpenAIChatClient  # type: ignore
except ImportError:  # pragma: no cover
    raise SystemExit(
        "agent_framework package not found. Install project dependencies first."
    )

RESEARCHER_AGENT_NAME = "Researcher-Agent"
RESEARCHER_AGENT_INSTRUCTIONS = (
    "你是我的研究员，和我一起分析一些问题。"
)


def _build_client() -> OpenAIChatClient:
    base_url = os.environ.get("GITHUB_ENDPOINT")
    model_id = os.environ.get("GITHUB_MODEL_ID")
    api_key = os.environ.get("GITHUB_TOKEN")
    if not base_url:
        raise RuntimeError(
            "No model endpoint configured. Set FOUNDRYLOCAL_ENDPOINT or GITHUB_ENDPOINT."
        )
    return OpenAIChatClient(base_url=base_url, api_key=api_key, model_id=model_id)


try:
    _client = _build_client()
    researcher_agent = _client.as_agent(
        instructions=RESEARCHER_AGENT_INSTRUCTIONS,
        name=RESEARCHER_AGENT_NAME,
    )
except Exception as e:  # pragma: no cover
    print(f"[researcher_agent] initialization warning: {e}")
    researcher_agent = None  # type: ignore
