from groq import BaseModel
from langchain.chat_models import init_chat_model
from pydantic import Field


# 创建结构化输出对象
class Product(BaseModel):
    name: str = Field(min_length=2)
    price: float = Field(gt=0)


# 创建模型
model = init_chat_model("groq:llama-3.3-70b-versatile")

# 创建主模型
structured_primary = model.with_structured_output(Product)

# 创建备用模型
fallback_model = init_chat_model("groq:llama-3.1-8b-instant")
structured_fallback = fallback_model.with_structured_output(Product)

# 添加重试
primary_with_retry = structured_primary.with_retry(retry_if_exception_type=(ConnectionError, TimeoutError),
                                                   stop_after_attempt=2)

# 添加降级
robust_llm = primary_with_retry.with_fallbacks([structured_fallback])

result = robust_llm.invoke("提取产品信息...")

def create_robust_structured_llm(model_name: str, schema: type[BaseModel]):
    """
    创建鲁棒的结构化 LLM

    正确顺序：structured_output → retry → fallbacks
    """
    # 主模型：先创建结构化输出
    primary_structured = init_chat_model(model_name).with_structured_output(schema)

    # 备用模型：也要先创建结构化输出
    fallback_model = init_chat_model("groq:llama-3.1-8b-instant")
    fallback_structured = fallback_model.with_structured_output(schema)

    # 添加重试和降级
    return (
        primary_structured
        .with_retry(
            retry_if_exception_type=(ConnectionError, TimeoutError),
            stop_after_attempt=3,
            wait_exponential_jitter=True
        )
        .with_fallbacks([fallback_structured])
    )