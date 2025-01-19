import logging
import os
import traceback
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
from constants.code_enum import SysCodeEnum

logger = logging.getLogger(__name__***REMOVED***


def convert_img(image***REMOVED***:
    """
    将Word文档中的图片转换为Markdown中可以使用的链接或嵌入内容。
    :param image:
    :return: 图片标签字典
    """
    with image.open(***REMOVED*** as image_bytes:
    ***REMOVED***"src": ""***REMOVED***
        # return {"src": "data:{0***REMOVED***;base64,{1***REMOVED***".format(image.content_type, base64.b64encode(image_bytes.read(***REMOVED******REMOVED***.decode(***REMOVED******REMOVED******REMOVED***


async def convert_word_to_md(file_key***REMOVED***:
    """

    :param file_key:
    :return:
    """
    html_path = file_key + "_to_html.html"
    docx_filename = file_key + ".docx"

    try:
        # 获取文件URL
        file_url = MinioUtils(***REMOVED***.get_file_url_by_key(object_key=file_key***REMOVED***

        # 从URL下载Word文档并读取其内容
        response = requests.get(file_url***REMOVED***
        response.raise_for_status(***REMOVED***  # 检查请求是否成功

        # 使用BytesIO来处理内存中的二进制数据流，避免创建临时文件
        docx_content = BytesIO(response.content***REMOVED***

        # 转化Word文档为HTML和Markdown
        result = mammoth.convert_to_html(docx_content, convert_image=mammoth.images.img_element(convert_img***REMOVED******REMOVED***
        html = result.value
        md = markdownify.markdownify(html, heading_style="ATX"***REMOVED***

        messages = result.messages
        if messages:
            for message in messages:
                print(f"警告或错误信息: {message***REMOVED***"***REMOVED***

        return md  # 直接返回Markdown文本

    except requests.RequestException as e:
        print(f"请求文件时发生错误：{e***REMOVED***"***REMOVED***
        raise MyException(SysCodeEnum.c_9999***REMOVED***
    except Exception as e:
        print(f"转换过程中发生错误：{e***REMOVED***"***REMOVED***
        raise MyException(SysCodeEnum.c_9999***REMOVED***
    finally:
        # 清理临时文件
        for path in [html_path, docx_filename]:
            if os.path.exists(path***REMOVED***:
                os.remove(path***REMOVED***


async def extract_toc_to_markdown(user_id, file_key***REMOVED***:
    """
    从Word文档中提取目录信息并转成Markdown格式。
    :param user_id
    :param file_key
    :return: Markdown格式的目录字符串
    """
    # 获取文件URL
    file_url = MinioUtils(***REMOVED***.get_file_url_by_key(object_key=file_key***REMOVED***

    # 从URL下载Word文档并读取其内容
    response = requests.get(file_url***REMOVED***
    response.raise_for_status(***REMOVED***  # 检查请求是否成功

    # 使用BytesIO来处理内存中的二进制数据流，避免创建临时文件
    docx_content = BytesIO(response.content***REMOVED***

    toc_md = []
    document = Document(docx_content***REMOVED***

    for paragraph in document.paragraphs:
        if paragraph.style and paragraph.style.name.startswith("Heading"***REMOVED***:
            level = int(paragraph.style.name[-1]***REMOVED***  # 获取标题级别
            markdown_heading = "#" * level + " " + paragraph.text
            toc_md.append(markdown_heading***REMOVED***

    md = "\n".join(toc_md***REMOVED***
    insert_markdown_to_db(user_id=user_id, file_key=file_key, markdown=md***REMOVED***
    return md


mysql_client = MysqlUtil(***REMOVED***


def insert_markdown_to_db(user_id, file_key, markdown***REMOVED***:
    """
    将Markdown内容插入到数据库表t_test_assistant中。
    :param user_id
    :param file_key: 文件的MinIO key
    :param markdown: Markdown格式的文本
    """
    try:
        # 插入数据
        sql = "INSERT INTO t_test_assistant (user_id,file_key, markdown, create_time, update_time***REMOVED*** VALUES (%s,%s, %s, %s, %s***REMOVED***"
        current_time = datetime.now(***REMOVED***
        data = (user_id, file_key, markdown, current_time, current_time***REMOVED***

        mysql_client.insert(sql, data***REMOVED***

    except Exception as e:
        traceback.print_exception(e***REMOVED***
        logger.error(f"保存测试助手记录失败: {e***REMOVED***"***REMOVED***


async def query_test_assistant_records(file_key=None, page=1, limit=10***REMOVED***:
    """
    根据文件key查询t_test_assistant表中的记录，并支持分页。

    :param file_key: 文件的MinIO key，用于过滤查询结果。如果为None，则不应用此过滤条件。
    :param page: 当前页码，默认为第一页。
    :param limit: 每页显示的记录数，默认为10条。
    :return: 包含分页信息和记录列表的字典。
    """
    # 构建SQL查询语句的基础部分
    base_sql = "SELECT * FROM t_test_assistant"
    where_clause = " WHERE 1=1 "
    params = []

    # 如果提供了file_key，则添加到WHERE子句中
    if file_key:
        where_clause = " AND file_key=%s"
        params.append(file_key***REMOVED***

    # 获取总记录数
    count_sql = f"SELECT COUNT(1***REMOVED*** AS count FROM t_test_assistant{where_clause***REMOVED***"
    total_count = mysql_client.query_mysql_dict_params(count_sql, params***REMOVED***[0]["count"]
    total_pages = ceil(total_count / limit***REMOVED***  # 计算总页数

    # 计算偏移量
    offset = (page - 1***REMOVED*** * limit

    # 添加LIMIT和OFFSET子句
    fetch_sql = f"{base_sql***REMOVED***{where_clause***REMOVED*** ORDER BY id DESC LIMIT %s OFFSET %s"
    params.extend([limit, offset]***REMOVED***

    # 执行查询并获取结果
    records = mysql_client.query_mysql_dict_params(fetch_sql, params***REMOVED***

***REMOVED***"records": records, "current_page": page, "total_pages": total_pages, "total_count": total_count***REMOVED***
