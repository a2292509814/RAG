import os
import dspy
from openai import OpenAI

# 从环境变量获取 API 配置
api_base = os.getenv("API2D_BASE_URL", "https://openai.api2d.net")
api_key = os.getenv("API2D_API_KEY")

if not api_key:
    raise ValueError("请设置环境变量 API2D_API_KEY")

# 创建 OpenAI 客户端
client = OpenAI(
    base_url=api_base,
    api_key=api_key
)

# 创建 DSPy 的 LM (Language Model) 配置
lm = dspy.OpenAI(
    model='gpt-3.5-turbo',
    api_base=api_base,
    api_key=api_key
)

# 配置 DSPy 使用这个 LM
dspy.configure(lm=lm)

# 创建简单的知识库
knowledge_base = [
    "harrison worked at kensho",
    "bears like to eat honey"
]

# 定义 RAG 模块
class SimpleRAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.gen = dspy.ChainOfThought("context, question -> answer")
    
    def forward(self, question):
        # 简单的文本匹配作为检索
        context = [text for text in knowledge_base if any(word in text.lower() for word in question.lower().split())]
        context = " ".join(context) if context else "No relevant information found."
        
        # 生成答案
        pred = self.gen(context=context, question=question)
        return pred.answer

# 创建 RAG 实例并测试
rag = SimpleRAG()

try:
    question = "where did harrison work?"
    answer = rag(question)
    print(f"问题: {question}")
    print(f"回答: {answer}")
except Exception as e:
    print(f"发生错误: {str(e)}")
