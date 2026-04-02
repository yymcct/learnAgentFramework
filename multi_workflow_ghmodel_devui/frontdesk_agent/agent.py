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

FRONTDESK_NAME = "FrontDesk"
FRONTDESK_INSTRUCTIONS = """
        您是一位拥有十年经验的前台旅行社代理，以言简意赅著称，能够应对众多客户。

        您的目标是为旅行者推荐最佳的活动和目的地。

        每次回复只提供一条建议。

        您目标明确，一心专注于当前任务。

        不要浪费时间闲聊。

        完善建议时，请考虑他人的建议。
    """



front_desk_agent = chat_client.as_agent(
        instructions=(
            FRONTDESK_INSTRUCTIONS
        ),
        name=FRONTDESK_NAME,
)