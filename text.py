from langchain_core.tools import tool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    """搜索工具的参数"""
    query: str = Field(description="搜索关键词")
    max_results: int = Field(default=10, ge=1, le=100, description="最大结果数")
    language: str = Field(default="zh", description="语言代码")

@tool(args_schema=SearchInput)
def advanced_search(query: str, max_results: int = 10, language: str = "zh") -> str:
    """高级搜索工具，支持参数验证"""
    return f"搜索 '{query}'，返回 {max_results} 条 {language} 结果"