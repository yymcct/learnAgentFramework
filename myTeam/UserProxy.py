import asyncio
from agent_framework import Agent
from myTeam.openai_client import client


agent = client.as_agent(
    name="UserProxy",
    instructions='''
    你是一个严格的用户代理，负责发起爆款文章生成任务、确认选题、审批最终文章，并决定是否继续迭代。
    你的风格：简洁、专业、果断。只在必要时介入，不要主动生成内容。
    当前任务：帮助用户生成一篇适合知乎/小红书/微博/公众号的爆款文章。
    任务流程：收集用户需求->通知内容团队生成初稿
    
    当用户提供需求时，确认需求是否清晰完整。如果不清晰，要求用户补充细节。收集完毕后，明确回复收集完毕，并通知内容团队开始写作。
    
    ''',
)

session = agent.create_session()


async def userProxy():
    while True:
        user_input = input("请输入用户指令（或输入 'exit' 退出）：")
        if user_input.lower() == "exit":
            print("退出用户代理。")
            break

        response = await agent.run(user_input, session=session)
        print(f"代理回复: {response.text}")
        
        if "收集完毕" in response.text:
            print("内容团队已被通知开始写作。")
            break


if __name__ == "__main__":
    asyncio.run(userProxy())
    