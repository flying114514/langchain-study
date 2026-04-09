from typing import TypedDict

from langchain.agents import create_agent
from langgraph.graph import StateGraph

# 1. 定义共享状态
class TeamState(TypedDict):
    task: str
    current_agent: str
    messages: list
    final_result: str

# 2. 创建专业化 Agent
researcher = create_agent(
    model="openai:gpt-4o",
    tools=[search_tool, wikipedia_tool],
    system_prompt="你是一个研究员，专门收集和整理信息。"
)

writer = create_agent(
    model="openai:gpt-4o",
    tools=[],
    system_prompt="你是一个作家，擅长将信息组织成清晰的文章。"
)

# 3. 创建监督者逻辑
def supervisor(state: TeamState) -> str:
    # 决定下一个执行的 Agent
    if "需要研究" in state["task"]:
        return "researcher"
    elif "需要写作" in state["task"]:
        return "writer"
    else:
        return "end"