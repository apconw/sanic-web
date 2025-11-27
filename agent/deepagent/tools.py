"""
深度搜索工具模块
提供网络搜索功能,支持 Tavily 和 Brave Search 两种搜索引擎
"""

from langchain_core.tools import tool
import json
import logging
from langchain_community.tools import BraveSearch
from tavily import TavilyClient
import os

logger = logging.getLogger(__name__)

# 从环境变量获取 API 密钥
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY", "").strip()

# 默认搜索函数为 None
web_search = None

# 优先使用 Brave Search(如果配置了 API 密钥)
# if BRAVE_SEARCH_API_KEY:
#     brave_search = BraveSearch.from_api_key(
#         api_key=BRAVE_SEARCH_API_KEY, search_kwargs={"count": 5}  # 返回 5 条搜索结果
#     )
#
#     def web_search_func(query: str):
#         """使用 Brave Search 执行搜索"""
#         return brave_search.run(query)
#
#     web_search = web_search_func
#     SEARCH_PROVIDER = "brave"
#     logging.info("使用 Brave Search 作为搜索引擎")

# 如果没有 Brave Search,则使用 Tavily Search
# elif TAVILY_API_KEY:
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


def web_search_func(query: str, max_results: int = 5):
    """使用 Tavily Search 执行搜索"""
    return tavily_client.search(query, max_results=max_results)


web_search = web_search_func
SEARCH_PROVIDER = "tavily"
logging.info("使用 Tavily Search 作为搜索引擎")


@tool
def search_web(query: str) -> str:
    """
    网络搜索工具

    参数:
        query: 搜索查询字符串

    返回:
        JSON 格式的搜索结果,包含查询内容和搜索结果列表
    """
    # 检查搜索引擎是否已配置
    if not web_search:
        return json.dumps({"error": "搜索引擎未配置"}, ensure_ascii=False)

    try:
        # 执行搜索
        results = web_search(query)

        # 返回 JSON 格式的结果
        return json.dumps({"query": query, "results": results}, indent=2, ensure_ascii=False)  # 支持中文字符
    except Exception as e:
        # 捕获并返回错误信息
        logging.error(f"搜索失败: {str(e)}")
        return json.dumps({"error": f"搜索失败: {str(e)}"}, ensure_ascii=False)
