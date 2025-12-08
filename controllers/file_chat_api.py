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
@openapi.summary("读取文件内容")
@openapi.description("读取Excel文件的第一行内容（表头）")
@openapi.tag("文件服务")
@openapi.parameter(
    name="file_qa_str",
    location="query",
    schema={"type": "string"},
    description="文件地址（MinIO对象key）",
    required=True,
)
@openapi.body(
    {
        "application/json": {
            "schema": {
                "type": "object",
                "properties": {
                    "file_qa_str": {"type": "string", "description": "文件地址（MinIO对象key）"},
                },
            }
        }
    },
    description="文件地址（可通过query参数或body传递）",
    required=False,
)
@openapi.response(
    200,
    {
        "application/json": {
            "schema": {
                "type": "object",
                "properties": {
                    "data": {"type": "array", "description": "文件第一行数据"},
                    "columns": {"type": "array", "description": "列名列表"},
                },
            }
        }
    },
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
@openapi.summary("读取文件列信息")
@openapi.description("读取Excel文件的列信息（表头）")
@openapi.tag("文件服务")
@openapi.parameter(
    name="file_qa_str",
    location="query",
    schema={"type": "string"},
    description="文件地址（MinIO对象key）",
    required=True,
)
@openapi.body(
    {
        "application/json": {
            "schema": {
                "type": "object",
                "properties": {
                    "file_qa_str": {"type": "string", "description": "文件地址（MinIO对象key）"},
                },
            }
        }
    },
    description="文件地址（可通过query参数或body传递）",
    required=False,
)
@openapi.response(
    200,
    {
        "application/json": {
            "schema": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "列名列表",
                    },
                },
            }
        }
    },
    description="列信息",
)
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
@openapi.summary("上传文件")
@openapi.description("上传文件到MinIO存储")
@openapi.tag("文件服务")
@openapi.body(
    {
        "multipart/form-data": {
            "schema": {
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "format": "binary",
                        "description": "要上传的文件",
                    },
                },
                "required": ["file"],
            }
        }
    },
    description="文件上传",
    required=True,
)
@openapi.response(
    200,
    {
        "application/json": {
            "schema": {
                "type": "object",
                "properties": {
                    "file_key": {"type": "string", "description": "文件在MinIO中的key"},
                    "file_url": {"type": "string", "description": "文件访问URL"},
                },
            }
        }
    },
    description="上传成功",
)
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
@openapi.summary("上传文件并解析")
@openapi.description("上传文件到MinIO并解析文件内容")
@openapi.tag("文件服务")
@openapi.body(
    {
        "multipart/form-data": {
            "schema": {
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "format": "binary",
                        "description": "要上传的文件",
                    },
                },
                "required": ["file"],
            }
        }
    },
    description="文件上传",
    required=True,
)
@openapi.response(
    200,
    {
        "application/json": {
            "schema": {
                "type": "object",
                "properties": {
                    "file_key": {"type": "string", "description": "文件在MinIO中的key"},
                    "parsed_content": {"type": "object", "description": "解析后的文件内容"},
                },
            }
        }
    },
    description="上传并解析成功",
)
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
@openapi.summary("处理文件问答LLM输出")
@openapi.description("处理文件问答中大模型返回的SQL语句并执行查询")
@openapi.tag("文件服务")
@openapi.parameter(
    name="file_key",
    location="query",
    schema={"type": "string"},
    description="文件key（MinIO对象key）",
    required=True,
)
@openapi.body(
    {
        "application/json": {
            "schema": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "大模型返回的SQL语句"},
                },
            }
        }
    },
    description="SQL语句（通过body传递）",
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
