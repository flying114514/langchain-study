from typing import TypedDict, Annotated

from langgraph.constants import START, END
from langgraph.graph.message import add_messages


# 1. 定义状态 - 使用 TypedDict
# 状态是在流程图中统一使用的数据结构，包含所有节点共享的信息。通过使用 TypedDict，我们可以明确状态的结构和类型。
class State(TypedDict):
    messages: Annotated[list, add_messages]  # 消息列表，自动累加
    current_step: str  # 当前步骤


# 2. 定义节点 - 接收和返回状态的函数
def my_node(state: State) -> dict:
    # 处理逻辑,更新状态
    return {"current_step": "completed"}


def router(state: State):
    question = state["messages"][-1]

    if "天气" in question:
        return {"intent": "weather", "need_search": False}
    else:
        return {"intent": "qa", "need_search": True}


def route_decision(state: State):
    if state["need_search"]:
        return "search"
    else:
        return "chat"


def search_node(state: State):
    query = state["messages"][-1]

    docs = ["文档1", "文档2"]  # 模拟检索
    return {"documents": docs}

def llm_node(state: State):
    docs = state.get("documents", [])
    question = state["messages"][-1]

    answer = f"基于 {docs} 回答: {question}"
    return {"final_answer": answer}



# 3. 创建图
from langgraph.graph import StateGraph

graph = StateGraph(State)
graph.add_node("my_node", my_node)
graph.add_edge(START, "my_node")
graph.add_edge("my_node", END)
graph.add_conditional_edges(
    "router",
    route_decision,
    {
        "search": "search_node",
        "chat": "chat_node"
    }
)

# 4. 编译并运行
app = graph.compile()
result = app.invoke({"messages": [], "current_step": "start"})
