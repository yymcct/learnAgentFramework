import os
from dotenv import load_dotenv
import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from typing import List
from agent_framework import Message, Role, ChatOptions
from agent_framework.openai import OpenAIChatClient, OpenAIResponsesClient

from OutputStruct import OutputStruct

load_dotenv()

os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

OPENAI_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_CHAT_MODEL_ID = os.getenv("OPENAI_CHAT_MODEL_ID")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

AGENT_NAME = "TravelAgent"

# 📋 System Instructions for Planning Agent
# This agent acts as a coordinator that decides which specialized agents to use
AGENT_INSTRUCTIONS = """You are a planner agent.
    Your job is to decide which agents to run based on the user's request.
    Below are the available agents specialized in different tasks:
    - FlightBooking: For booking flights and providing flight information
    - HotelBooking: For booking hotels and providing hotel information
    - CarRental: For booking cars and providing car rental information
    - ActivitiesBooking: For booking activities and providing activity information
    
"""

class SubTask(BaseModel):
    assigned_agent: str = Field(
        description="The specific agent assigned to handle this subtask")
    task_details: str = Field(
        description="Detailed description of what needs to be done for this subtask")


class TravelPlan(BaseModel):
    main_task: str = Field(
        description="The overall travel request from the user")
    subtasks: List[SubTask] = Field(
        description="List of subtasks broken down from the main task, each assigned to a specialized agent")
    
options = ChatOptions(response_format=TravelPlan)

client = OpenAIChatClient(base_url="https://openrouter.ai/api/v1", 
                              api_key=os.environ.get("OPENROUTER_API_KEY"), 
                              model_id=os.environ.get("OPENAI_CHAT_MODEL_ID"))


agent =  client.as_agent(name= AGENT_NAME , 
                         instructions=AGENT_INSTRUCTIONS,
                         default_options={"response_format": TravelPlan})

async def main():
    messages = [
            Message(role="user", text="Create a travel plan for a family of 4, with 2 kids, from Singapore to Melbourne")
        ]

    response = await agent.run(messages,response_format=TravelPlan)
    print(response)

# Run the async function
import asyncio
asyncio.run(main())