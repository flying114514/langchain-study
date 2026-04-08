import os

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.chat_models import init_chat_model


class MessageTrimmerMiddleware(AgentMiddleware):
    def __init__(self, max_messages=5):
        super().__init__()
        self.max_messages = max_messages

    def before_model(self, state, runtime):
        messages = state.get('messages', [])
        if len(messages) > self.max_messages:
            # 只保留最近的 N 条消息
            return {"messages": messages[-self.max_messages:]}
        return None


model = init_chat_model(
    "groq:llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# 使用中间件
agent = create_agent(
    model=model,
    tools=[],
    middleware=[MyMiddleware()]
)
