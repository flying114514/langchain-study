from langchain.chat_models import init_chat_model
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import TextLoader
from langchain_core.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.agents import create_agent

# 存数据
# 加载文件
loader = TextLoader(file_path="docs.txt", encoding="utf-8")
documents = loader.load()

# 分词
splitter = RecursiveCharacterTextSplitter(chunk_size=200,
                                          chunk_overlap=50)
chunks = splitter.split_documents(documents)

# 文本嵌入
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 存入向量数据库
# vectorStore = PineconeVectorStore.from_documents(
#     documents=chunks,
#     embedding=embeddings,
#     index_name="my-index"
# )

# 连接数据库
vectorStore = PineconeVectorStore(
    index_name="my-index",
    embedding=embeddings
)

# 创建向量检索器
vector_retriever = vectorStore.as_retriever(search_kwargs={"k": 3})

# BM25检索器
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 3

# 创建混合检索器
ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5])

# 查数据(创建agent版)
# 创建查询工具
# @tool
# def search_docs(query: str) -> str:
#     """搜索文档"""
#     docs = ensemble_retriever.invoke(query)
#     return "\n\n".join([doc.page_content for doc in docs])
#
#
# # 创建agent
# model = init_chat_model("groq:llama-3.1-8b-instant")
#
# agent = create_agent(
#     model=model,
#     tools=[search_docs],
#     system_prompt="你是助手。使用 search_docs 搜索信息，然后回答问题。"
# )
#
# # 问答
# response = agent.invoke({
#     "message": [{"role": "user", "content": "LangChain 有什么特性？"}]
# })
# print(response)

# 混合检索
docs = ensemble_retriever.invoke("LangChain 有什么特性？")

# 拼接文档
context = "\n\n".join([doc.page_content for doc in docs])

# 创建主次模型
primary_model = init_chat_model(model="llama-3.1-13b-instant", model_provider="groq")
fallback_model = init_chat_model("llama-3.1-8b-instant", model_provider="groq")

# 添加重试和降级
robust_llm = primary_model.with_retry(retry_if_exception_type=(ConnectionError, TimeoutError),
                                      stop_after_attempt=2).with_fallbacks([fallback_model])

# 提问
response = robust_llm.invoke([{"role": "system", "content": "你是一个专业助手"}, {"role": "user",
                                                                                  "content": f"根据以下内容回答问题：\n\n{context}\n\n问题：LangChain 有什么特性？"}])

print(response.content)
