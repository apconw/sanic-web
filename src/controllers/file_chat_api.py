import logging
from urllib.parse import unquote
from sanic import Blueprint, Request

from common.exception import MyException
from common.minio_util import MinioUtils
from common.res_decorator import async_json_resp
from constants.code_enum import SysCodeEnum
from services.pandas_ai_service import read_excel, query_excel, read_file_columns
from services.text2_sql_service import exe_file_sql_query

bp = Blueprint("fileChatApi", url_prefix="/file")

minio_utils = MinioUtils()


@bp.post("/read_file")
@async_json_resp
async def read_file(req: Request):
    """
    读取excel文件第一行内容
    :param req:
    :return:
    """

    file_key = req.args.get("file_qa_str")
    if not file_key:
        file_key = req.json.get("file_qa_str")

    file_key = file_key.split("|")[0]  # 取文档地址

    file_url = minio_utils.get_file_url_by_key(object_key=file_key)
    result = await read_excel(file_url)
    return result


@bp.post("/read_file_column")
@async_json_resp
async def read_file_column(req: Request):
    """
    读取excel文件第一行内容
    :param req:
    :return:
    """

    file_key = req.args.get("file_qa_str")
    if not file_key:
        file_key = req.json.get("file_qa_str")

    file_key = file_key.split("|")[0]  # 取文档地址

    file_url = minio_utils.get_file_url_by_key(object_key=file_key)
    result = await read_file_columns(file_url)
    return result


@bp.get("/query_excel")
@async_json_resp
async def process_query(req: Request):
    """
     通过object_key获取文件url
    :param req:
    :return:
    """
    query_str = unquote(req.query_string)
    if not query_str:
        return None

    try:
        str_split = query_str.replace("&", "").replace("=", "").split("@@@@")
        query_text = str_split[0]
        file_key = str_split[1]
        file_url = minio_utils.get_file_url_by_key(object_key=file_key)

        result = await query_excel(file_url, query_text)
        return result
    except IndexError:
        raise ValueError("Invalid query string format")
    except Exception as e:
        raise RuntimeError(f"An error occurred: {e}")


@bp.post("/upload_file")
@async_json_resp
async def upload_file(request: Request):
    """
    上传附件
    :param request:
    :return:
    """
    file_url = minio_utils.upload_file_from_request(request=request)
    return file_url


@bp.post("/process_file_llm_out")
@async_json_resp
async def process_file_llm_out(req):
    """
    文件问答处理大模型返回SQL语句
    """
    try:
        # 获取请求体内容
        body_content = req.body
        # # 将字节流解码为字符串
        body_str = body_content.decode("utf-8")

        # 文件key
        file_key = req.args.get("file_key")
        logging.info(f"query param: {body_str}")

        result = await exe_file_sql_query(file_key, body_str)
        return result
    except Exception as e:
        logging.error(f"Error processing LLM output: {e}")
        raise MyException(SysCodeEnum.c_9999)
