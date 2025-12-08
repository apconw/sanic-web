import logging

from sanic import Blueprint
from sanic.response import ResponseStream
from sanic_ext import openapi

from common.exception import MyException
from common.res_decorator import async_json_resp
from common.token_decorator import check_token
from constants.code_enum import SysCodeEnum
from services.dify_service import DiFyRequest, query_dify_suggested, stop_dify_chat

bp = Blueprint("fiFyApi", url_prefix="/dify")

dify = DiFyRequest()


@bp.post("/get_answer")
@openapi.summary("获取Dify答案（流式）")
@openapi.description("调用Dify画布获取数据，以流式方式返回结果")
@openapi.tag("Dify服务")
@openapi.body(
    {
        "application/json": {
            "schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "查询内容"},
                    "chat_id": {"type": "string", "description": "聊天ID"},
                },
            }
        }
    },
    description="查询请求体",
    required=True,
)
@openapi.response(
    200,
    {"text/event-stream": {"schema": {"type": "string"}}},
    description="流式返回数据",
)
@check_token
async def get_answer(req):
    """
        调用diFy画布获取数据流式返回
    :param req:
    :return:
    """

    try:
        response = ResponseStream(dify.exec_query, content_type="text/event-stream")
        return response
    except Exception as e:
        logging.error(f"Error Invoke diFy: {e}")
        raise MyException(SysCodeEnum.c_9999)


@bp.post("/get_dify_suggested", name="get_dify_suggested")
@openapi.summary("获取Dify问题建议")
@openapi.description("根据聊天ID获取Dify推荐的问题建议")
@openapi.tag("Dify服务")
@openapi.body(
    {
        "application/json": {
            "schema": {
                "type": "object",
                "properties": {
                    "chat_id": {"type": "string", "description": "聊天ID"},
                },
                "required": ["chat_id"],
            }
        }
    },
    description="请求体",
    required=True,
)
@openapi.response(
    200,
    {
        "application/json": {
            "schema": {
                "type": "object",
                "properties": {
                    "suggestions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "建议问题列表",
                    },
                },
            }
        }
    },
    description="返回建议问题列表",
)
@check_token
@async_json_resp
async def dify_suggested(request):
    """
    dify问题建议
    :param request:
    :return:
    """
    chat_id = request.json.get("chat_id")
    return await query_dify_suggested(chat_id)


@bp.post("/stop_chat", name="stop_chat")
@openapi.summary("停止聊天")
@openapi.description("停止正在进行的聊天任务")
@openapi.tag("Dify服务")
@openapi.body(
    {
        "application/json": {
            "schema": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "任务ID"},
                    "qa_type": {"type": "string", "description": "问答类型"},
                },
                "required": ["task_id", "qa_type"],
            }
        }
    },
    description="停止请求体",
    required=True,
)
@openapi.response(200, description="停止成功")
@check_token
@async_json_resp
async def stop_chat(request):
    """
    👂 停止聊天
    :param request:
    :return:
    """
    task_id = request.json.get("task_id")
    qa_type = request.json.get("qa_type")
    return await stop_dify_chat(request, task_id, qa_type)
