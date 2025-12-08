import logging

from sanic import Blueprint, request
from sanic_ext import openapi

from services.db_qadata_process import select_report_by_title
from services.text2_sql_service import exe_sql_query
from common.exception import MyException
from constants.code_enum import SysCodeEnum
from common.res_decorator import async_json_resp

bp = Blueprint("text2sql", url_prefix="/llm")


@bp.post("/process_llm_out")
@openapi.summary("处理LLM输出的SQL")
@openapi.description("数据问答处理大模型返回的SQL语句并执行查询")
@openapi.tag("数据问答")
@openapi.body(
    {
        "application/x-www-form-urlencoded": {
            "schema": {
                "type": "object",
                "properties": {
                    "llm_text": {"type": "string", "description": "大模型返回的SQL语句"},
                },
                "required": ["llm_text"],
            }
        }
    },
    description="SQL语句",
    required=True,
)
@openapi.response(
    200,
    {
        "application/json": {
            "schema": {
                "type": "object",
                "properties": {
                    "data": {"type": "array", "description": "查询结果"},
                    "columns": {"type": "array", "description": "列名"},
                },
            }
        }
    },
    description="查询成功",
)
@async_json_resp
async def process_llm_out(req: request.Request):
    """
    数据问答处理大模型返回SQL语句
    """
    try:
        # 获取请求体内容
        # body_content = req.body
        # # 将字节流解码为字符串
        # body_str = body_content.decode("utf-8")

        body_str = req.form.get("llm_text")

        # 用户问题
        # question_str = req.args.get("question")
        logging.info(f"query param: {body_str}")

        result = await exe_sql_query(body_str)
        return result
    except Exception as e:
        logging.error(f"Error processing LLM output: {e}")
        raise MyException(SysCodeEnum.c_9999)


@bp.get("/query_guided_report")
@openapi.summary("查询引导报告")
@openapi.description("根据查询字符串查询相关的引导报告")
@openapi.tag("数据问答")
@openapi.parameter(
    name="query_str",
    location="query",
    schema={"type": "string"},
    description="查询字符串",
    required=True,
)
@openapi.response(
    200,
    {
        "application/json": {
            "schema": {
                "type": "object",
                "properties": {
                    "reports": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string", "description": "报告标题"},
                                "content": {"type": "string", "description": "报告内容"},
                            },
                        },
                        "description": "报告列表",
                    },
                },
            }
        }
    },
    description="查询成功",
)
@async_json_resp
async def query_guided_report(req: request.Request):
    """
    查询报告
    """
    try:
        question_str = req.args.get("query_str").strip().replace("\r", "")
        result = await select_report_by_title(question_str)
        return result
    except Exception as e:
        logging.error(f"查询报告失败: {e}")
        raise MyException(SysCodeEnum.c_9999)
