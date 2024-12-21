# 需要安装：
# pip install langchain docarray tiktoken

import os
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

# 从环境变量获取 API 配置
api_base = os.getenv("API2D_BASE_URL", "https://openai.api2d.net")
api_key = os.getenv("API2D_API_KEY")

# 检查 API 密钥是否存在
if not api_key:
    raise ValueError("请设置环境变量 API2D_API_KEY")

# 创建向量存储，使用文本创建嵌入
vectorstore = DocArrayInMemorySearch.from_texts(
    ["harrison worked at kensho", "bears like to eat honey"],
    embedding=OpenAIEmbeddings(
        openai_api_base=api_base,
        openai_api_key=api_key,
        timeout=30
    ),
)

# 将向量存储转换为检索器
retriever = vectorstore.as_retriever()

# 定义提示模板
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 配置 OpenAI 模型
model = ChatOpenAI(
    openai_api_base=api_base,
    openai_api_key=api_key,
    model_name="gpt-3.5-turbo",
    timeout=30
)

# 定义输出解析器
output_parser = StrOutputParser()

# 设置并行运行的任务，检索上下文和问题
setup_and_retrieval = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)

# 创建处理链：检索 -> 提示 -> 模型 -> 输出解析
chain = setup_and_retrieval | prompt | model | output_parser

try:
    # 调用链以获取回答
    response = chain.invoke("where did harrison work?")
    print(f"回答: {response}")
except Exception as e:
    # 捕获并打印任何错误
    print(f"发生错误: {str(e)}")