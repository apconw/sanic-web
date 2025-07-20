from sanic import Blueprint

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
