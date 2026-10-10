"""
You.com Search API tool for DeepAgent.

API文档: https://you.com/specs/openapi_search_v1.yaml
"""

import json
import logging
import os
from typing import Optional

import requests
from langchain_core.tools import tool

from .tool_call_manager import get_tool_call_manager

logger = logging.getLogger(__name__)

YOU_SEARCH_API_URL = "https://ydc-index.io/v1/search"
YOU_SEARCH_API_KEY = os.getenv("YOU_SEARCH_API_KEY", "").strip()


def _get_session_id() -> str:
    """获取当前会话ID，用于工具调用管理"""
    from .native_sql_tools import _get_session_id as _get_ds_session
    return _get_ds_session()


def _check_tool_call(tool_name: str, query: Optional[str] = None) -> tuple[bool, str]:
    """检查工具调用是否允许（研究工具也纳入会话管理）"""
    session_id = _get_session_id()
    manager = get_tool_call_manager()
    return manager.check_before_call(session_id, tool_name, query)


def _record_tool_call(tool_name: str, success: bool, query: Optional[str] = None) -> None:
    """记录工具调用"""
    session_id = _get_session_id()
    manager = get_tool_call_manager()
    manager.record_call(session_id, tool_name, success, query)
    logger.debug(f"记录工具调用: tool={tool_name}, success={success}, session={session_id}")


@tool
def youcom_search(query: str) -> str:
    """
    通过 You.com Search API 搜索网络信息，返回带有标题、链接、内容摘要的搜索结果。
    适用于需要实时网络信息的研究查询。

    Args:
        query: 搜索关键词或自然语言查询语句
    """
    # 检查是否允许调用
    allowed, reason = _check_tool_call("youcom_search", query)
    if not allowed:
        return reason

    api_key = os.getenv("YOU_SEARCH_API_KEY", "").strip()
    if not api_key:
        _record_tool_call("youcom_search", False, query)
        return "错误: YOU_SEARCH_API_KEY 未设置。请在 .env.dev 中配置 You.com API 密钥。"

    max_results = 10
    url = YOU_SEARCH_API_URL
    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json",
    }
    payload = json.dumps({
        "query": query,
        "count": max_results,
    })

    try:
        response = requests.post(url, headers=headers, data=payload, timeout=30)
        if response.status_code == 200:
            result_data = response.json()
            web_results = result_data.get("results", {}).get("web", [])

            if not web_results:
                _record_tool_call("youcom_search", True, query)
                return "未找到相关搜索结果。"

            formatted = []
            for item in web_results:
                title = item.get("title", "无标题")
                link = item.get("url", "")
                description = item.get("description") or ""
                snippets = item.get("snippets", [])
                snippet = snippets[0] if snippets else description
                page_age = item.get("page_age", "")

                entry = f"标题: {title}\n链接: {link}\n摘要: {snippet}"
                if page_age:
                    entry += f"\n时间: {page_age}"
                formatted.append(entry)

            result_str = "\n---\n".join(formatted)
            _record_tool_call("youcom_search", True, query)
            return result_str

        elif response.status_code == 401:
            _record_tool_call("youcom_search", False, query)
            return f"错误: You.com API 密钥无效或已过期 (HTTP {response.status_code})。"
        elif response.status_code == 403:
            _record_tool_call("youcom_search", False, query)
            return f"错误: You.com API 密钥缺少所需权限 (HTTP {response.status_code})。"
        else:
            _record_tool_call("youcom_search", False, query)
            return f"错误: You.com Search API 请求失败 (HTTP {response.status_code}): {response.text[:200]}"

    except requests.exceptions.Timeout:
        _record_tool_call("youcom_search", False, query)
        return "错误: You.com Search API 请求超时（30秒）。"
    except requests.exceptions.RequestException as e:
        _record_tool_call("youcom_search", False, query)
        return f"错误: You.com Search API 请求失败: {str(e)[:200]}"
    except Exception as e:
        _record_tool_call("youcom_search", False, query)
        return f"错误: {str(e)[:200]}"
