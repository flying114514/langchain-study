import os

from langchain.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
# from pinecone import Pinecone, ServerlessSpec

# 初始化Pinecone客户端
# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# 创建索引
# pc.create_index(
#     name="my-index",
#     dimension=384,
#     metric="cosine",
#     spec=ServerlessSpec(
#         cloud="aws",
#         region="us-east-1"
#     )
# )

# 加载模型
primary_model = init_chat_model("groq:llama-3.3-70b-versatile")

# 存储文件
# 加载文件
loader = TextLoader("docs.txt", encoding="utf-8")
documents = loader.load()

# 切分文件
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# 文本嵌入
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 存入向量数据库
PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name="my-index")

# 查询数据
# 连接数据库
vectorstore = PineconeVectorStore(index_name="my-index", embedding=embeddings)


# 创建检索工具提供给llm使用
@tool
def search_kb(query: str) -> str:
    """搜索知识库"""
    docs = vectorstore.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in docs])


# 创建Agent
agent = create_agent(model=primary_model,
                     tools=[search_kb])

response = agent.invoke({
    "messages": [{"role": "user", "content": "什么是软件开发"},
                 {"role": "system",
                  "content": "你是一名资深的软件开发工程师,你需要根据向量数据库中的数据如实回答用户提出的问题,如果在数据库中未能找到相关信息,请直接返回'未找到相关信息'."}]
})

print(response)
