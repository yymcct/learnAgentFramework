import os 

from agent_framework.openai import OpenAIChatClient  
from dotenv import load_dotenv # 📁 Secure configuration loading

env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
load_dotenv(env_path)

chat_client = OpenAIChatClient(
    base_url=os.environ.get("GITHUB_ENDPOINT"),    # 🌐 GitHub Models API endpoint
    api_key=os.environ.get("GITHUB_TOKEN"),        # 🔑 Authentication token
    model_id=os.environ.get("GITHUB_MODEL_ID")     # 🎯 Selected AI model
)

REVIEWER_NAME = "Concierge"
REVIEWER_INSTRUCTIONS = """
您是一位酒店礼宾，对如何为旅客提供最具当地特色和地道风情的体验有独到的见解。

您的目标是判断前台旅行社推荐的非旅游化体验是否符合旅客的需求。

如果符合，请说明该推荐是否被认可。

如果不符合，请在不提供具体示例的情况下，提出改进建议。
    """



reviewer_agent = chat_client.as_agent(
        instructions=(
            REVIEWER_INSTRUCTIONS
        ),
        name=REVIEWER_NAME,
)