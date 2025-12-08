from sanic import Blueprint
from sanic_ext import openapi

from common.exception import MyException
from common.res_decorator import async_json_resp
from common.token_decorator import check_token
from constants.code_enum import SysCodeEnum
from services.user_service import (
    authenticate_user,
    generate_jwt_token,
    query_user_record,
    get_user_info,
    delete_user_record,
    send_dify_feedback,
)

bp = Blueprint("userService", url_prefix="/user")


@bp.post("/login")
@openapi.summary("用户登录")
@openapi.description("用户登录接口，验证用户名和密码，返回JWT token")
@openapi.tag("用户服务")
@openapi.body(
    {
        "application/json": {
            "schema": {
                "type": "object",
                "properties": {
                    "username": {"type": "string", "description": "用户名"},
                    "password": {"type": "string", "description": "密码"},
                },
                "required": ["username", "password"],
            }
        }
    },
    description="登录请求体",
    required=True,
)
@openapi.response(
    200,
    {
        "application/json": {
            "schema": {
                "type": "object",
                "properties": {"token": {"type": "string", "description": "JWT token"}},
            }
        }
    },
    description="登录成功，返回token",
)
@openapi.response(401, description="登录失败，用户名或密码错误")
@async_json_resp
async def login(request):
    """
    用户登录
    :param request:
    :return:
    """
    username = request.json.get("username")
    password = request.json.get("password")

    # 调用用户服务进行验证
    user = await authenticate_user(username, password)
    if user:
        # 如果验证通过，生成 JWT token
        token = await generate_jwt_token(user["id"], user["userName"])
        return {"token": token}
    else:
        # 如果验证失败，返回错误信息
        raise MyException(SysCodeEnum.c_401)


@bp.post("/query_user_record", name="query_user_record")
@openapi.summary("查询用户聊天记录")
@openapi.description("分页查询当前用户的聊天记录，支持按关键词和聊天ID筛选")
@openapi.tag("用户服务")
@openapi.body(
    {
        "application/json": {
            "schema": {
                "type": "object",
                "properties": {
                    "page": {"type": "integer", "description": "页码，默认1", "default": 1},
                    "limit": {"type": "integer", "description": "每页数量，默认10", "default": 10},
                    "search_text": {"type": "string", "description": "搜索关键词，可选"},
                    "chat_id": {"type": "string", "description": "聊天ID，可选"},
                },
            }
        }
    },
    description="查询参数",
    required=True,
)
@openapi.response(
    200,
    {
        "application/json": {
            "schema": {
                "type": "object",
                "properties": {
                    "data": {"type": "array", "description": "聊天记录列表"},
                    "total": {"type": "integer", "description": "总记录数"},
                },
            }
        }
    },
    description="查询成功",
)
@check_token
@async_json_resp
async def query_user_qa_record(request):
    """
    查询用户聊天记录
    :param request:
    :return:
    """
    page = int(request.json.get("page", 1))
    limit = int(request.json.get("limit", 10))
    search_text = request.json.get("search_text")
    chat_id = request.json.get("chat_id")
    user_info = await get_user_info(request)
    return await query_user_record(user_info["id"], page, limit, search_text, chat_id)


@bp.post("/delete_user_record")
@openapi.summary("删除用户聊天记录")
@openapi.description("批量删除当前用户的聊天记录")
@openapi.tag("用户服务")
@openapi.body(
    {
        "application/json": {
            "schema": {
                "type": "object",
                "properties": {
                    "record_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要删除的记录ID列表",
                    },
                },
                "required": ["record_ids"],
            }
        }
    },
    description="删除请求体",
    required=True,
)
@openapi.response(200, description="删除成功")
@check_token
@async_json_resp
async def delete_user_qa_record(request):
    """
    删除用户聊天记录
    :param request:
    :return:
    """
    record_ids = request.json.get("record_ids")
    user_info = await get_user_info(request)
    return await delete_user_record(user_info["id"], record_ids)


@bp.post("/dify_fead_back", name="dify_fead_back")
@openapi.summary("用户反馈")
@openapi.description("提交对Dify聊天的反馈评分")
@openapi.tag("用户服务")
@openapi.body(
    {
        "application/json": {
            "schema": {
                "type": "object",
                "properties": {
                    "chat_id": {"type": "string", "description": "聊天ID"},
                    "rating": {"type": "string", "description": "评分，如：like/dislike"},
                },
                "required": ["chat_id", "rating"],
            }
        }
    },
    description="反馈请求体",
    required=True,
)
@openapi.response(200, description="反馈提交成功")
@check_token
@async_json_resp
async def fead_back(request):
    """
    用户反馈
    :param request:
    :return:
    """
    chat_id = request.json.get("chat_id")
    rating = request.json.get("rating")
    return await send_dify_feedback(chat_id, rating)
