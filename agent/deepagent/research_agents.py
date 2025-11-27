"""
深度搜索智能体 - 主程序
使用 deepagents 框架实现多智能体协作的深度搜索系统
"""

from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from tools import search_web
import logging
import json
import os
from dotenv import load_dotenv

# === 日志配置 ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# === 加载环境变量 ===
load_dotenv()

# === 配置参数 ===
RECURSION_LIMIT = int(os.getenv("RECURSION_LIMIT", 25))
PORT = int(os.getenv("SERVER_PORT", 8000))

# === 加载核心指令 ===
# 从 instructions.md 文件读取系统提示词
with open("instructions.md", "r", encoding="utf-8") as f:
    CORE_INSTRUCTIONS = f.read()

# === 加载子智能体配置 ===
# 从 subagents.json 文件读取各个子智能体的角色定义
with open("subagents.json", "r", encoding="utf-8") as f:
    subagents_config = json.load(f)

# 提取三个子智能体的配置
planner = subagents_config["planner"]  # 规划师
researcher = subagents_config["researcher"]  # 研究员
analyst = subagents_config["analyst"]  # 分析师

# === 工具列表 ===
# 定义智能体可以使用的工具
tools = [search_web]

# === 初始化语言模型 ===
# 使用通义千问模型,支持中文对话
llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "qwen-plus"),
    temperature=float(os.getenv("MODEL_TEMPERATURE", 0)),
    base_url=os.getenv("MODEL_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    api_key=os.getenv("MODEL_API_KEY", "sk-eab7577236294b8d9fa96814be63e08f"),
    max_tokens=int(os.getenv("MAX_TOKENS", 4000)),
    streaming=os.getenv("STREAMING", "True").lower() == "true",
)

# === 创建深度智能体 ===
# 使用 deepagents 框架创建多智能体协作系统
# 包含监督者和三个子智能体(规划师、研究员、分析师)
agent = create_deep_agent(
    tools=tools,  # 可用工具列表
    system_prompt=CORE_INSTRUCTIONS,  # 系统提示词
    subagents=[planner, researcher, analyst],  # 子智能体配置
    model=llm,  # 语言模型
).with_config(
    {"recursion_limit": RECURSION_LIMIT}
)  # 设置递归限制

logging.info("✅ 深度搜索智能体已成功初始化")
