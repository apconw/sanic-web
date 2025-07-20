import asyncio
import base64
import json
import logging
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import BytesIO
from math import ceil

import mammoth
import markdownify
import requests
from docx import Document

from common.exception import MyException
from common.minio_util import MinioUtils
from common.mysql_util import MysqlUtil
from common.pdf_util import PdfUtil
from common.word_util import WordUtil
from constants.code_enum import SysCodeEnum

logger = logging.getLogger(__name__)


def convert_img(image):
    """
    将Word文档中的图片转换为Markdown中可以使用的链接或嵌入内容。
    :param image:
    :return: 图片标签字典
    """
    with image.open() as image_bytes:
        return {"src": "data:{0};base64,{1}".format(image.content_type, base64.b64encode(image_bytes.read()).decode())}


async def convert_word_to_md(file_key):
    """

    :param file_key:
    :return:
    """
    html_path = file_key + "_to_html.html"
    docx_filename = file_key + ".docx"

    try:
        # 获取文件URL
        file_url = MinioUtils().get_file_url_by_key(object_key=file_key)

        # 从URL下载Word文档并读取其内容
        response = requests.get(file_url)
        response.raise_for_status()  # 检查请求是否成功

        # 使用BytesIO来处理内存中的二进制数据流，避免创建临时文件
        docx_content = BytesIO(response.content)

        # 转化Word文档为HTML和Markdown
        result = mammoth.convert_to_html(docx_content, convert_image=mammoth.images.img_element(convert_img))
        html = result.value
        md = markdownify.markdownify(html, heading_style="ATX")

        messages = result.messages
        if messages:
            for message in messages:
                print(f"警告或错误信息: {message}")

        return md  # 直接返回Markdown文本

    except requests.RequestException as e:
        print(f"请求文件时发生错误：{e}")
        raise MyException(SysCodeEnum.c_9999)
    except Exception as e:
        print(f"转换过程中发生错误：{e}")
        raise MyException(SysCodeEnum.c_9999)
    finally:
        # 清理临时文件
        for path in [html_path, docx_filename]:
            if os.path.exists(path):
                os.remove(path)


async def extract_toc_to_markdown(user_id, file_key):
    """
    从Word文档中提取目录信息并转成Markdown格式。
    :param user_id
    :param file_key
    :return: Markdown格式的目录字符串
    """
    # 获取文件URL
    file_url = MinioUtils().get_file_url_by_key(object_key=file_key)

    # 从URL下载Word文档并读取其内容
    response = requests.get(file_url)
    response.raise_for_status()  # 检查请求是否成功

    # 使用BytesIO来处理内存中的二进制数据流，避免创建临时文件
    docx_content = BytesIO(response.content)

    toc_md = []
    document = Document(docx_content)

    for paragraph in document.paragraphs:
        if paragraph.style and paragraph.style.name.startswith("Heading"):
            level = int(paragraph.style.name[-1])  # 获取标题级别
            markdown_heading = "#" * level + " " + paragraph.text
            toc_md.append(markdown_heading)

    md = "\n".join(toc_md)

    # 转换word to pdf 并上传至minio
    file_key = PdfUtil().convert_document_to_pdf_from_minio(file_key)
    file_url = MinioUtils().get_file_url_by_key(object_key=file_key)
    # insert_markdown_to_db(user_id=user_id, file_key=file_key, file_url=file_url, markdown=md)

    return md


mysql_client = MysqlUtil()


async def insert_demand_manager_to_db(user_id, doc_name, doc_desc, file_key) -> int:
    """
    将Markdown内容插入到数据库表t_test_assistant中。
    :param user_id
    :param file_key: 文件的MinIO key
    :param doc_name: 需求文档名称
    :param doc_desc: 需求文档描述
    """
    try:
        # 插入数据
        sql = "INSERT INTO t_demand_manager (user_id,doc_name,doc_desc,file_key,create_time, update_time) VALUES (%s,%s,%s, %s, %s, %s)"
        current_time = datetime.now()
        data = (user_id, doc_name, doc_desc, file_key, current_time, current_time)
        record_id = mysql_client.insert(sql, data)

        # 保存文档元信息
        await insert_demand_doc_meta(user_id, record_id, file_key)

        return record_id
    except Exception as e:
        traceback.print_exception(e)
        logger.error(f"保存测试助手记录失败: {e}")
        return False


async def insert_demand_doc_meta(user_id, demand_id, file_key) -> bool:
    """
        保存文档元信息
    :param user_id:
    :param demand_id:
    :param file_key:
    :return:
    """
    try:
        outline_with_content = WordUtil().read_target_content(file_key)
        outline_dict_list = [{"功能模块": heading, "功能点详息说明": [f"  {line}" for line in content]} for heading, content in outline_with_content]

        for outline_dict in outline_dict_list:
            page_title = outline_dict["功能模块"]
            page_content = json.dumps(outline_dict["功能点详息说明"])
            sql = "INSERT INTO t_demand_doc_meta (user_id,demand_id,page_title,page_content,create_time, update_time) VALUES (%s,%s,%s,%s, %s, %s)"
            current_time = datetime.now()
            data = (user_id, demand_id, page_title, page_content, current_time, current_time)
            mysql_client.insert(sql, data)

    except Exception as e:
        traceback.print_exception(e)
        logger.error(f"保存文档元信息失败: {e}")
        raise Exception(f"保存文档元信息失败: {e}")


async def query_demand_records(user_id, file_key=None, page=1, limit=10):
    """
    根据文件key查询t_test_assistant表中的记录，并支持分页。

    :param user_id
    :param file_key: 文件的MinIO key，用于过滤查询结果。如果为None，则不应用此过滤条件。
    :param page: 当前页码，默认为第一页。
    :param limit: 每页显示的记录数，默认为10条。
    :return: 包含分页信息和记录列表的字典。
    """
    # 构建SQL查询语句的基础部分
    base_sql = "SELECT * FROM t_demand_manager"
    where_clause = f" WHERE 1=1 and user_id={user_id} "
    params = []

    # 如果提供了file_key，则添加到WHERE子句中
    if file_key:
        where_clause = " AND file_key=%s"
        params.append(file_key)

    # 获取总记录数
    count_sql = f"SELECT COUNT(1) AS count FROM t_demand_manager{where_clause}"
    total_count = mysql_client.query_mysql_dict_params(count_sql, params)[0]["count"]
    total_pages = ceil(total_count / limit)  # 计算总页数

    # 计算偏移量
    offset = (page - 1) * limit

    # 添加LIMIT和OFFSET子句
    fetch_sql = f"{base_sql}{where_clause} ORDER BY id DESC LIMIT %s OFFSET %s"
    params.extend([limit, offset])

    # 执行查询并获取结果
    records = mysql_client.query_mysql_dict_params(fetch_sql, params)

    return {"records": records, "current_page": page, "total_pages": total_pages, "total_count": total_count}


async def delete_demand_records(record_id):
    """
    删除t_test_assistant表中的记录。
    :param record_id: 要删除的记录的ID。
    """
    # 构建SQL删除语句
    delete_sql = f"DELETE FROM t_demand_manager WHERE id={record_id}"
    mysql_client.execute_mysql(delete_sql)

    delete_sql = f"DELETE FROM t_demand_doc_meta WHERE demand_id={record_id}"
    mysql_client.execute_mysql(delete_sql)


executor = ThreadPoolExecutor()


async def abstract_doc_func(response, doc_id):
    """
    抽取功能点信息
    :param response
    :param doc_id
    :return:
    """
    try:
        logging.info(f"query param: {doc_id}")

        sql = f"select * from t_demand_doc_meta where demand_id='{doc_id}'"
        meta_dict = mysql_client.query_mysql_dict(sql)
        total_steps = len(meta_dict)

        # 确定每个步骤应该增加的百分比
        step_percentage = 10 if total_steps <= 10 else (100 / total_steps)

        function_array = []
        for step, item in enumerate(meta_dict):
            # 计算当前进度，如果是最后一步，则直接设为100%
            current_progress = min(100, int((step + 1) * step_percentage))
            await response.write(f'data: {{"type": "progress", "progress": {current_progress}, "total": 100}}\n\n')
            await response.write(f'data: {{"type": "log", "message": "{"#### 模块: " + item["page_title"]}"}}\n\n')

            # 这里避免阻塞主线程
            loop = asyncio.get_running_loop()
            answer = await loop.run_in_executor(executor, extract_function, item["page_content"])

            case_info = {
                "demand_id": item["demand_id"],
                "section_id": item["id"],
                "section_name": item["page_title"],
                "fun_names": [],
            }

            logger.info(answer)
            answer_arr = json.loads(answer.strip("```json\n").strip("\n```"))
            case_info["fun_names"].extend(answer_arr)
            function_array.append(case_info)

            for index, func in enumerate(answer_arr):
                await response.write(
                    "data:"
                    + json.dumps(
                        {"type": "log", "message": f"- {func} </br>"},
                        ensure_ascii=False,
                    )
                    + "\n\n"
                )

            await response.write(
                "data:"
                + json.dumps(
                    {"type": "log", "message": "---"},
                    ensure_ascii=False,
                )
                + "\n\n"
            )
        # 完成后发送完成标志
        await response.write('data: {"type": "complete"}\n\n')
        await response.write("\n\n")

        logger.info(function_array)
        update_functions(doc_id, function_array)
        insert_demand_case(doc_id, function_array)
    except Exception as e:
        logging.error(f"Error Invoke diFy: {e}")
        traceback.print_exception(e)


def update_functions(doc_id, function_array):
    """
    更新功能点数量
    :param doc_id:
    :param function_array:
    :return:
    """
    functions = sum(len(item["fun_names"]) for item in function_array)
    sql = f"""update t_demand_manager set fun_num={functions},update_time='{datetime.now()}' where id={doc_id}"""
    mysql_client.update(sql)


def insert_demand_case(doc_id, function_array):
    """
        添加功能点信息
    :param doc_id:
    :param function_array:
    :return:
    """

    # 先删除数据
    delete_sql = f"""delete from t_demand_case where demand_id={doc_id}"""
    mysql_client.execute_mysql(delete_sql)

    # 插入数据的SQL语句
    insert_query = """
    INSERT INTO t_demand_case (demand_id, section_id, section_name, fun_name, create_time, update_time)
    VALUES (%s, %s, %s, %s, %s, %s)
    """

    # 准备批量插入的数据
    values_to_insert = []
    for item in function_array:
        for fun_name in item["fun_names"]:
            values_to_insert.append((doc_id, item["section_id"], item["section_name"], fun_name, datetime.now(), datetime.now()))

    mysql_client.batch_insert(insert_query, values_to_insert)


result_format = """["功能点1","功能点2"]"""


def build_prompt(doc_content) -> str:
    """
    构建提示词

    :return:
    """
    prompt_content = f"""
    # system: 你是一个测试专家精通从需求文档内容中抽取具体功能点
    # 任务: 从需求文档内容中抽取具体功能点
    -----------
    # 需求文档内容: {doc_content}
    -----------

    # 约束:
    - 严格依据需求文档内容回答不要虚构
    - 每个功能点信息字数限制在30字以内
    - 根据需求文档内容尽量列举出所有功能点信息
    - 不要输出思考过程信息
    # 返回格式
    请一步步思考并按照以下JSON格式回复：{result_format}
    确保返回正确的json并且可以被Python json.loads方法解析.
    """
    return prompt_content


def extract_function(doc_content):
    """

    :return:
    """
    # Ollama 服务器地址
    url = "http://127.0.0.1:11434/api/generate"

    # 构建请求体
    payload = {
        "prompt": build_prompt(doc_content),
        "model": "qwen2.5",
        "stream": False,
        "think_output": False,
        "max_tokens": 8192,
        "temperature": 0,
        "top_k": 1,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
    }

    headers = {"Content-Type": "application/json"}

    response = requests.post(url, data=json.dumps(payload), headers=headers)
    if response.status_code == 200:
        # 解析响应中的结果
        result = response.text
        return json.loads(result)["response"]
    else:
        logger.error(f"Error: {response.status_code}")
