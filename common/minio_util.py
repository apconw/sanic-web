import csv
import io
import json
import logging
import mimetypes
import os
import traceback
from datetime import timedelta
from uuid import uuid4
import requests
import pandas as pd
import pymupdf
import pymupdf4llm
from docx import Document
from minio import Minio, S3Error
from sanic import Request

from common.exception import MyException
from constants.code_enum import SysCodeEnum as SysCode

logger = logging.getLogger(__name__)


class MinioUtils:
    """
    上传文件工具类
    """

    def __init__(self):
        self.client = self._build_client()
        # self.executor = ThreadPoolExecutor(max_workers=5)  # 多线程上传控制最大并发数

    @staticmethod
    def _build_client():
        """初始化MinIO客户端"""
        minio_endpoint = os.getenv("MINIO_ENDPOINT")
        access_key = os.getenv("MINIO_ACCESS_KEY")
        secret_key = os.getenv("MiNIO_SECRET_KEY")
        if not all([minio_endpoint, access_key, secret_key]):
            raise MyException(SysCode.c_9999, "MinIO环境变量未正确配置")
        return Minio(endpoint=minio_endpoint, access_key=access_key, secret_key=secret_key, secure=False)

    def ensure_bucket(self, bucket_name: str) -> None:
        """确保bucket存在，不存在则创建"""
        try:
            found = self.client.bucket_exists(bucket_name)
            if not found:
                self.client.make_bucket(bucket_name)
                logger.info(f"Bucket '{bucket_name}' created.")
            else:
                logger.info(f"Bucket '{bucket_name}' already exists.")
        except S3Error as err:
            logger.error(f"Error checking or creating bucket {bucket_name}: {err}")
            raise MyException(SysCode.c_9999)

    def upload_file_from_request(self, request: Request, bucket_name: str = "filedata",
                                 object_name: str = None) -> dict:
        """
        从请求中读取文件数据并上传到MinIO服务器，返回预签名URL。

        参数:
        - request: Sanic请求对象
        - bucket_name: 存储桶名称
        返回:
        - 包含object_key的字典
        """
        try:
            file_data = request.files.get("file")
            if not file_data:
                raise MyException(SysCode.c_9999, "未找到文件数据")

            file_stream = io.BytesIO(file_data.body)
            file_length = len(file_data.body)
            if object_name is None:
                # uuid可以避免不同用户上传同名文件 导致minio的文件被覆盖
                object_name = f"{uuid4()}__{file_data.name}"

            self.ensure_bucket(bucket_name)
            self.client.put_object(bucket_name, object_name, file_stream, file_length, content_type=file_data.type)
            logger.info(f"File successfully uploaded as {object_name}.")

            return {"object_key": object_name}
        except Exception as err:
            logger.error(f"Error uploading file from request: {err}")
            traceback.print_exception(err)
            raise MyException(SysCode.c_9999)

    def upload_to_minio_form_stream(
        self, file_stream: io.BytesIO, bucket_name: str = "filedata", file_name: str | None = None
    ) -> str | None:
        """
        将给定的字节流上传到MinIO，并返回上传文件的键（key）。

        :param file_stream: 文件的字节流 (BytesIO)
        :param bucket_name: MinIO存储桶名称
        :param file_name: 上传文件的名称（可选）
        :return: 上传文件的键（key）或None如果上传失败
        """
        try:
            self.ensure_bucket(bucket_name)

            if not file_name:
                file_extension = mimetypes.guess_extension(mimetypes.guess_type(file_stream.getvalue())[0]) or ""
                file_name = f"{uuid4()}{file_extension}"

            file_stream.seek(0)
            file_length = len(file_stream.getvalue())
            content_type, _ = mimetypes.guess_type(file_name) or ("application/octet-stream", None)

            self.client.put_object(bucket_name, file_name, file_stream, file_length, content_type=content_type)
            logger.info(f"File uploaded successfully with key: {file_name}")
            return file_name
        except Exception as e:
            logger.error(f"An error occurred while uploading to MinIO: {e}")
            return None

    def get_file_url_by_key(self, bucket_name: str = "filedata", object_key: str | None = None) -> str:
        """
        通过object_key获取文件url
        """
        try:
            if not object_key:
                raise MyException(SysCode.c_9999, "object_key不能为空")
            return self.client.presigned_get_object(bucket_name, object_key, expires=timedelta(days=7))
        except Exception as err:
            logger.error(f"Error getting file URL by key: {err}")
            traceback.print_exception(err)
            raise MyException(SysCode.c_9999)

    def upload_file_and_parse_from_request(self, request: Request, bucket_name: str = "filedata") -> dict:
        """
        上传文件并解析文件内容，返回文件内容key。

        参数:
        - request: Sanic请求对象
        - bucket_name: 存储桶名称
        返回:
        - 文件内容key
        """

        try:
            file_data = request.files.get("file")
            if not file_data:
                raise MyException(SysCode.c_9999, "未找到文件数据")

            content = io.BytesIO(file_data.body)
            # uuid可以避免不同用户上传同名文件 导致minio的文件被覆盖
            object_name = f"{uuid4()}__{file_data.name}"
            mime_type = file_data.type
            file_suffix = ".txt"
            # 可选：添加文件大小限制（例如 50MB）
            if len(file_data.body) > 50 * 1024 * 1024:
                raise MyException(SysCode.c_9999, "文件大小超出限制")

            source_file_key = self.upload_file_from_request(request, bucket_name, object_name)

            # 校验 MIME 类型是否支持（增强安全性）
            allowed_mimes = {
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
                "application/msword",  # .doc
                "text/plain",  # .txt
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # .xlsx
                "application/vnd.ms-excel",  # .xls
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # .pptx
                "application/vnd.ms-powerpoint",  # .ppt
                "application/pdf",  # .pdf
                "text/csv",  # .csv
            }

            # 获取文件大小
            file_size = len(file_data.body)

            if mime_type not in allowed_mimes:
                raise ValueError("不支持的文件格式")

            # 根据文件类型选择不同的方式读取内容
            if mime_type in (
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/msword",
            ):
                doc = Document(content)
                full_text = "\n".join([para.text for para in doc.paragraphs])
            elif mime_type == "text/plain":
                content.seek(0)
                full_text = content.read().decode("utf-8")
            elif mime_type == "text/csv":
                content.seek(0)
                full_text = self._parse_csv(content)
            elif mime_type in (
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "application/vnd.ms-excel",
            ):
                content.seek(0)
                full_text = self._parse_excel(content, mime_type)
            elif mime_type in (
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                "application/vnd.ms-powerpoint",
            ):
                content.seek(0)
                full_text = self.read_pdf_text_from_bytes(content.getvalue())
            elif mime_type == "application/pdf":
                # todo 如果pdf文件中包含图片，则需要使用OCR处理图片 私有化部署minerU支持
                content.seek(0)
                full_text = self.read_pdf_text_from_bytes(content.getvalue())
            else:
                raise ValueError("不支持的文件格式")

            # 创建一个txt文件并上传
            parse_file_key = self.upload_to_minio_form_stream(
                io.BytesIO(full_text.encode("utf-8")), bucket_name, object_name + file_suffix
            )
            return {
                "source_file_key": source_file_key["object_key"],
                "parse_file_key": parse_file_key,
                "file_size": self._format_file_size(file_size),
            }
        except Exception as err:
            logger.error(f"Error uploading file and parsing from request: {err}")
            traceback.print_exception(type(err), err, err.__traceback__)
            raise MyException(SysCode.c_9999) from err

    @staticmethod
    def _format_file_size(size_bytes: int) -> str:
        """
        将字节大小转换为人类可读的格式 (如: 12KB, 1MB)

        :param size_bytes: 文件大小（字节）
        :return: 格式化后的文件大小字符串
        """
        if size_bytes == 0:
            return "0B"

        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024.0 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1

        if i == 0:
            # 对于字节，不使用小数点
            return f"{int(size_bytes)}{size_names[i]}"
        else:
            # 对于KB及以上，保留一位小数
            return f"{size_bytes:.1f}{size_names[i]}"

    @staticmethod
    def _parse_csv(content):
        """
        解析CSV文件内容
        :param content: CSV文件内容
        :return: 解析后的文本
        """
        try:
            content.seek(0)
            # 尝试不同的编码
            encodings = ["utf-8", "gbk", "gb2312"]
            lines = None
            for encoding in encodings:
                try:
                    content.seek(0)
                    text = content.read().decode(encoding)
                    lines = text.splitlines()
                    break
                except UnicodeDecodeError:
                    continue

            if lines is None:
                raise ValueError("无法解码CSV文件")

            full_text = ""
            reader = csv.reader(lines)
            for row in reader:
                full_text += "\t".join(row) + "\n"
            return full_text
        except Exception as e:
            logger.error(f"读取CSV文件时出错: {e}")
            raise MyException(SysCode.c_9999, "CSV解析失败") from e

    @staticmethod
    def _parse_excel(content, mime_type):
        """
        解析Excel文件内容，输出结构化JSON，便于大模型识别
        自动根据 MIME 类型选择正确引擎，避免 xlrd 读取 .xlsx
        :param content: Excel文件内容（BytesIO 或 bytes）
        :param mime_type: 文件MIME类型
        :return: JSON 字符串，含结构化表格数据
        """
        try:
            if isinstance(content, bytes):
                content = io.BytesIO(content)

            # 根据 MIME 类型智能选择引擎，避免 xlrd 读取 .xlsx
            if mime_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":  # .xlsx
                engine = "openpyxl"
            elif mime_type == "application/vnd.ms-excel":  # .xls
                engine = "xlrd"
            else:
                # 默认尝试 openpyxl（更安全）
                engine = "openpyxl"
                logger.warning(f"未知 MIME 类型: {mime_type}，默认使用 openpyxl 引擎")

            xls = pd.ExcelFile(content, engine=engine)

            result = {"file_type": "excel", "mime_type": mime_type, "engine_used": engine, "sheets": []}

            for sheet_name in xls.sheet_names:
                # 读取时不自动推断 header，保留原始行列结构
                df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
                sheet_data = {
                    "sheet_name": sheet_name,
                    "nrows": len(df),
                    "ncols": len(df.columns) if not df.empty else 0,
                    "rows": [],
                }

                for row_idx, row in df.iterrows():
                    row_cells = []
                    for col_idx, cell in enumerate(row):
                        # 保留原始值类型（int/float/datetime 等），转字符串用于显示
                        raw_value = cell
                        display_value = str(cell) if pd.notna(cell) else None
                        row_cells.append(
                            {
                                "row": row_idx + 1,  # 从 1 开始，符合人类习惯
                                "col": col_idx + 1,
                                "value": display_value,  # 用于显示/LLM 理解
                                "raw_value": raw_value,  # 保留原始数据类型
                            }
                        )
                    sheet_data["rows"].append(row_cells)

                result["sheets"].append(sheet_data)

            return json.dumps(result, ensure_ascii=False, indent=2, default=str)

        except Exception as e:
            logger.error(f"读取Excel文件时出错: {e}")
            raise MyException(SysCode.c_9999, "Excel解析失败") from e

    def read_pdf_text_from_bytes(self, file_bytes):
        """
        从字节数据中读取文件返回markdown文本 缺点不支持图片解析 如果开启需要走公网服务
        :param file_bytes: bytes, PDF 文件的二进制内容
        :return: str, 提取的文本内容
        """
        try:
            with pymupdf.open(stream=file_bytes) as doc:
                has_text = False
                has_images = False

                # 合并一次遍历判断文本和图像
                for page in doc:
                    if not has_text and len(page.get_text("text").strip()) > 0:
                        has_text = True
                    if not has_images and len(page.get_images(full=True)) > 0:
                        has_images = True
                    if has_text and has_images:
                        break  # 提前退出，避免多余遍历

                if has_text and has_images:
                    # 场景：文字/纯图片/扫描件 PDF → 调用 MinerU OCR 服务
                    logger.info("检测到扫描件PDF，调用私有化MinerU服务进行OCR...")
                    full_text = self._call_mineru_ocr_service(file_bytes)
                else:
                    # 场景：普通可读PDF → 使用 pymupdf 提取 Markdown
                    full_text = pymupdf4llm.to_markdown(doc=doc, ignore_images=True)
                return full_text
        except Exception as e:
            logger.error(f"读取文本时出错: {e}")
            raise MyException(SysCode.c_9999, "PDF 解析失败") from e

    def get_files_content_as_markdown(self, file_info_list: list, bucket_name: str = "filedata") -> str:
        """
        根据文件信息列表获取文件内容并拼接成Markdown格式

        参数:
        - file_info_list: 文件信息列表，格式如 [{"source_file_key": "销售数据.xlsx", "parse_file_key": "销售数据.xlsx.txt", "file_size": "11.0KB"}]
        - bucket_name: 存储桶名称

        返回:
        - 拼接后的Markdown格式文本
        """
        result_parts = []

        for file_info in file_info_list:
            source_file_key = file_info.get("source_file_key")
            parse_file_key = file_info.get("parse_file_key")

            if not parse_file_key:
                continue

            try:
                # 获取文件内容
                response = self.client.get_object(bucket_name, parse_file_key)
                content = response.data.decode("utf-8")
                response.close()
                response.release_conn()

                # 获取文件扩展名
                _, ext = os.path.splitext(source_file_key or parse_file_key)

                # 构建Markdown格式文本
                file_part = f"- 文件名称: {source_file_key}\n- 文件格式: {ext}\n- 文件内容: {content}"
                result_parts.append(file_part)

            except Exception as e:
                logger.error(f"读取文件 {parse_file_key} 内容时出错: {e}")
                # 即使某个文件读取出错也继续处理其他文件
                continue

        # 使用分隔线连接各部分
        return "\n----------\n".join(result_parts) if result_parts else ""

    @staticmethod
    def _call_mineru_ocr_service(pdf_bytes: bytes) -> str:
        """
        调用私有化部署的 MinerU 服务进行 PDF OCR 解析。

        参数:
            pdf_bytes (bytes): PDF 文件的二进制数据

        返回:
            str: OCR 解析后的文本内容

        异常:
            MyException: 当 MinerU 服务调用失败时抛出
        """
        try:
            # 🔧 配置 MinerU 服务地址（私有化部署）
            mineru_api_url = os.getenv("MINERU_API_RUL")
            headers = {
                "Authorization": "Bearer your-secret-token",  # 可选认证
            }

            files = {"file": ("document.pdf", pdf_bytes, "application/pdf")}

            response = requests.post(mineru_api_url, files=files, headers=headers, timeout=300)  # 支持大文件，超时5分钟

            if response.status_code != 200:
                raise MyException(SysCode.c_9999, f"MinerU服务返回错误: {response.status_code}")

            result = response.json()
            return result.get("text", "") or result.get("content", "")

        except requests.exceptions.RequestException as e:
            logger.error(f"调用MinerU服务失败: {e}")
            raise MyException(SysCode.c_9999, "OCR服务不可用，请检查网络或服务状态") from e
        except Exception as e:
            logger.error(f"解析MinerU返回结果失败: {e}")
            raise MyException(SysCode.c_9999, "OCR解析结果异常") from e
