from enum import Enum
from typing import Optional, List

from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model


model = init_chat_model("groq:llama-3.3-70b-versatile")

class Priority(str, Enum):
    LOW = "低"
    MEDIUM = "中"
    HIGH = "高"


class Person(BaseModel):
    name: str
    age: int


class Person(BaseModel):
    """人物信息"""
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")
    occupation: str = Field(description="职业")
    email: Optional[str] = None  # 可以为 None
    money: Priority = Field(description="财富等级")

class PeopleList(BaseModel):
    people: List[Person]  # 多个 Person 对象

class Address(BaseModel):
    city: str
    district: str

class Company(BaseModel):
    name: str
    address: Address  # 嵌套模型

class CustomerInfo(BaseModel):
    name: str = Field(description="客户姓名")
    phone: str = Field(description="电话号码")
    email: Optional[str] = Field(None, description="邮箱")
    issue: str = Field(description="问题描述")

class Review(BaseModel):
    product: str
    rating: int = Field(description="评分 1-5")
    pros: List[str] = Field(description="优点列表")
    cons: List[str] = Field(description="缺点列表")

class Invoice(BaseModel):
    invoice_number: str
    date: str
    total_amount: float
    items: List[str]
structured_llm = model.with_structured_output(Invoice)
review = structured_llm.invoke("""
发票号: INV-2024-001
日期: 2024-01-15
总金额: 1299.00
商品: MacBook Pro, AppleCare+
""")
print(review)

