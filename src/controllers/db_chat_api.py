import logging

from sanic import Blueprint, request

from services.db_qadata_process import select_report_by_title
from services.text2_sql_service import exe_sql_query
from common.exception import MyException
from constants.code_enum import SysCodeEnum
from common.res_decorator import async_json_resp

bp = Blueprint("text2sql", url_prefix="/llm")


@bp.post("/process_llm_out")
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
