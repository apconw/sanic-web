"""
测试助手api
已废弃
"""

import asyncio
import logging

from sanic import Blueprint
from sanic.response import ResponseStream
from sanic_ext import openapi

from common.exception import MyException
from common.res_decorator import async_json_resp
from common.token_decorator import check_token
from constants.code_enum import SysCodeEnum
from services.ta_assistant_service import (
    extract_toc_to_markdown,
    insert_demand_manager_to_db,
    query_demand_records,
    delete_demand_records,
    abstract_doc_func,
)
from services.user_service import get_user_info

bp = Blueprint("testAssistant", url_prefix="/ta")


@bp.post("/insert_demand_manager")
@openapi.summary("保存项目信息")
@openapi.description("保存项目信息到数据库（已废弃）")
@openapi.tag("测试助手（已废弃）")
@openapi.body(
    {
        "application/json": {
            "schema": {
                "type": "object",
                "properties": {
                    "project_data": {
                        "type": "object",
                        "properties": {
                            "file_key": {"type": "string", "description": "文件key"},
                            "doc_name": {"type": "string", "description": "文档名称"},
                            "doc_desc": {"type": "string", "description": "文档描述"},
                        },
                        "required": ["file_key", "doc_name"],
                    },
                },
                "required": ["project_data"],
            }
        }
    },
    description="项目数据",
    required=True,
)
@openapi.response(200, description="保存成功")
@async_json_resp
async def insert_demand_manager(request):
    """
    保存项目信息
    :param request:
    :return:
    """
    form_data = request.json.get("project_data")
    file_key = form_data.get("file_key")
    doc_name = form_data.get("doc_name")
    doc_desc = form_data.get("doc_desc")
    user_info = await get_user_info(request)
    result = await insert_demand_manager_to_db(user_info["id"], doc_name, doc_desc, file_key)
    return result


@bp.post("/word_to_md")
@openapi.summary("转换Word文档为Markdown")
@openapi.description("将Word文档转换为Markdown格式，提取目录结构（已废弃）")
@openapi.tag("测试助手（已废弃）")
@openapi.body(
    {
        "application/json": {
            "schema": {
                "type": "object",
                "properties": {
                    "file_key": {"type": "string", "description": "文件key"},
                },
                "required": ["file_key"],
            }
        }
    },
    description="文件信息",
    required=True,
)
@openapi.response(
    200,
    {
        "application/json": {
            "schema": {
                "type": "object",
                "properties": {
                    "markdown": {"type": "string", "description": "转换后的Markdown内容"},
                },
            }
        }
    },
    description="转换成功",
)
@check_token
@async_json_resp
async def convert_doc_to_markdown_api(request):
    """
    转换word文档 markdown
    :param request:
    :return:
    """
    file_key = request.json.get("file_key")
    # result = await convert_word_to_md(file_key)
    user_info = await get_user_info(request)
    result = await extract_toc_to_markdown(user_info["id"], file_key)
    return result


@bp.post("/query_demand_records", name="query_demand_records")
@openapi.summary("查询需求记录")
@openapi.description("查询t_test_assistant表中的记录，支持分页和文件过滤（已废弃）")
@openapi.tag("测试助手（已废弃）")
@openapi.body(
    {
        "application/json": {
            "schema": {
                "type": "object",
                "properties": {
                    "page": {"type": "integer", "description": "页码，默认1", "default": 1},
                    "limit": {"type": "integer", "description": "每页数量，默认10", "default": 10},
                    "file_key": {"type": "string", "description": "文件key，用于过滤，可选"},
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
                    "data": {"type": "array", "description": "记录列表"},
                    "total": {"type": "integer", "description": "总记录数"},
                },
            }
        }
    },
    description="查询成功",
)
@check_token
@async_json_resp
async def query_demand_records_api(request):
    """
    查询t_test_assistant表中的记录，支持分页。

    :param request: 请求对象
    :return: JSON格式的查询结果
    """
    # 解析请求参数
    page = int(request.json.get("page", 1))
    limit = int(request.json.get("limit", 10))
    file_key = request.json.get("file_key")  # 可选参数，用于过滤查询结果

    # 获取用户信息
    user_info = await get_user_info(request)

    # 调用查询函数并返回结果
    return await query_demand_records(user_info["id"], file_key=file_key, page=page, limit=limit)


@bp.post("/delete_demand_records", name="delete_demand_records")
@openapi.summary("删除需求记录")
@openapi.description("删除项目信息记录（已废弃）")
@openapi.tag("测试助手（已废弃）")
@openapi.body(
    {
        "application/json": {
            "schema": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "description": "记录ID"},
                },
                "required": ["id"],
            }
        }
    },
    description="删除请求体",
    required=True,
)
@openapi.response(200, description="删除成功")
@check_token
@async_json_resp
async def delete_demand_records_api(request):
    """
    删除项目信息
    :param request:
    :return:
    """
    return await delete_demand_records(request.json.get("id"))


@bp.route("/abstract_doc_func/<item_id>", name="abstract_doc_func", methods=["GET"])
@openapi.summary("抽取功能点信息")
@openapi.description("从文档中抽取功能点信息，流式返回（已废弃）")
@openapi.tag("测试助手（已废弃）")
@openapi.parameter(
    name="item_id",
    location="path",
    schema={"type": "integer"},
    description="项目ID",
    required=True,
)
@openapi.response(
    200,
    {"text/event-stream": {"schema": {"type": "string"}}},
    description="流式返回功能点信息",
)
async def abstract_doc_func_api(req, item_id):
    """
    抽取功能点信息
    :param req:
    :param item_id
    :return:
    """
    try:
        return ResponseStream(lambda response: abstract_doc_func(response, item_id), content_type="text/event-stream")
    except Exception as e:
        logging.error(f"Error Invoke diFy: {e}")
        raise MyException(SysCodeEnum.c_9999)
