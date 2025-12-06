import logging

from common.exception import MyException
from common.minio_util import MinioUtils
from common.res_decorator import async_json_resp
from constants.code_enum import SysCodeEnum
from sanic import Blueprint, Request
from services.file_chat_service import read_excel, read_file_columns
from services.text2_sql_service import exe_file_sql_query
from sanic_ext import openapi

bp = Blueprint("fileChatApi", url_prefix="/file")

minio_utils = MinioUtils()


@bp.post("/read_file")
@openapi.parameter(name="file_qa_str", description="文件地址", required=True)
@openapi.response(
    200,
    content={"application/json": '{"data": 1}'},
    description="文件内容",
)
@async_json_resp
async def read_file(req: Request) -> dict:
    """
    读取excel文件第一行内容
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


@bp.post("/upload_file")
@async_json_resp
async def upload_file(request: Request):
    """
    上传附件
    :param request:
    :return:
    """
    file_key = minio_utils.upload_file_from_request(request=request)
    return file_key


@bp.post("/upload_file_and_parse")
@async_json_resp
async def upload_file_and_parse(request: Request):
    """
    上传附件并解析内容
    :param request:
    :return:
    """
    file_key_dict = minio_utils.upload_file_and_parse_from_request(request=request)
    return file_key_dict


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
