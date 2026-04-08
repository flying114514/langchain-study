import aiohttp
from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import asyncio

class SearchInput(BaseModel):
    """搜索工具的参数"""
    query: str = Field(description="搜索关键词")
    max_results: int = Field(default=10, ge=1, le=100, description="最大结果数")
    language: str = Field(default="zh", description="语言代码")

@tool(args_schema=SearchInput)
def advanced_search(query: str, max_results: int = 10, language: str = "zh") -> str:
    """高级搜索工具，支持参数验证"""
    return f"搜索 '{query}'，返回 {max_results} 条 {language} 结果"

@tool
async def async_fetch(url: str) -> str:
    """异步获取网页内容"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

# 使用异步 Agent
async def main():
    agent = create_agent(model=model, tools=[async_fetch])
    response = await agent.ainvoke({"messages": [...]})

@tool
def safe_tool(query: str) -> str:
    """带错误处理的工具"""
    try:
        # 可能失败的操作
        result = risky_operation(query)
        return result
    except ValueError as e:
        return f"参数错误: {e}"
    except Exception as e:
        return f"工具执行失败: {e}"