"""
通用问答
"""
import logging
import traceback

from common.exception import MyException
from common.res_decorator import async_json_resp
from constants.code_enum import SysCodeEnum
from sanic import Blueprint, request
from services.search_service import get_bing_first_href

bp = Blueprint("common-chat", url_prefix="/common")


@bp.post("/get_search_url")
@async_json_resp
async def get_bing_search_url(req: request.Request):
    """
    通用问答 获取搜索引擎第一个结果url
    """
    try:
        query_str = req.args.get("query_str")
        return await get_bing_first_href(query_str)
    except Exception as e:
        traceback.print_exception(e)
        logging.error(f"Error processing LLM output: {e}")
        raise MyException(SysCodeEnum.c_9999)
