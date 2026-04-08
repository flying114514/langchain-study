from typing import Optional

from langchain.chat_models import init_chat_model
from pydantic import Field, BaseModel
from groq import BadRequestError  # 导入正确的异常类


class Product(BaseModel):
    name: str = Field(min_length=2)
    price: float = Field(gt=0)


model = init_chat_model("groq:llama-3.3-70b-versatile")


def extract_with_validation(
        text: str,
        max_retries: int = 3
) -> Optional[Product]:
    """带验证的提取函数"""
    structured_llm = model.with_structured_output(Product)
    current_text = text

    for attempt in range(1, max_retries + 1):
        try:
            result = structured_llm.invoke(f"提取: {current_text}")
            # 额外的业务验证
            if result.price < 0:
                raise ValueError("价格必须为正数")
            return result
        except (BadRequestError, ValueError) as e:
            if attempt < max_retries:
                error_msg = str(e)
                current_text = f"{text}\n错误: {error_msg}"
            else:
                return None  # 失败


# 使用
result = extract_with_validation("产品 A 价格 999 元")
if result:
    print(f"成功: {result.name}, {result.price}")
else:
    print("提取失败")