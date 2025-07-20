import logging
import re
from io import BytesIO

import requests

from common.minio_util import MinioUtils


class PdfUtil:
    """
    pdf工具类
    """

    @classmethod
    def convert_document_to_pdf_from_minio(cls, file_key):
        """
        在内存中转换word文档为pdf，不涉及本地文件系统。

        :param file_key: 文件在Minio中的键名
        :return: 转换后PDF的BytesIO对象或None
        """
        # 获取文件URL
        url = MinioUtils().get_file_url_by_key(object_key=file_key)

        try:
            # 下载文件内容到BytesIO对象
            response = requests.get(url)
            response.raise_for_status()
            file_content = BytesIO(response.content)

            conversion_url = "http://localhost:3000/forms/libreoffice/convert"

            # 发送POST请求，将文件内容作为multipart/form-data上传
            files = {"files": (file_key, file_content)}  # 假定是Word文档，根据实际情况调整

            conversion_response = requests.post(conversion_url, files=files)
            conversion_response.raise_for_status()

            # 将转换后的PDF存储在BytesIO对象中
            pdf_bytes_io = BytesIO(conversion_response.content)

            new_file_key = cls.change_extension_direct(file_key, "pdf")

            # 直接上传到Minio，或者进行其他内存中处理
            if MinioUtils().upload_to_minio_form_stream(file_stream=pdf_bytes_io, file_name=new_file_key):
                logging.info(f"Document converted and uploaded successfully.")
                return new_file_key  # 或者根据需求返回其他内容
            else:
                logging.error("Conversion succeeded but upload failed.")

        except requests.exceptions.RequestException as e:
            logging.error(f"An error occurred during the process: {e}")
            return None

    @classmethod
    def change_extension_direct(cls, filename, new_ext):
        """
        :param filename
        :param new_ext:
        :return:
        """
        # 直接使用正则表达式替换扩展名
        return re.sub(r"\.[^.]*$", "." + new_ext, filename)
