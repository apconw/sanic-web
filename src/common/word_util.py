from io import BytesIO

import requests
from docx import Document

from common.minio_util import MinioUtils


class WordUtil:
    """
    word文档工具类
    """

    @classmethod
    def read_target_content(cls, file_key, target_heading_text="详细功能需求"):
        """
        读取位于指定一级标题下的所有内容。

        参数:
        - doc_path: Word文档的路径。
        - target_heading_text: 目标一级标题的文本，默认为"详细功能需求"。

        返回:
        - 包含目标大纲项及其下所有具体内容的列表。
        """

        # 获取文件URL
        file_url = MinioUtils().get_file_url_by_key(object_key=file_key)

        # 从URL下载Word文档并读取其内容
        response = requests.get(file_url)
        response.raise_for_status()  # 检查请求是否成功

        # 使用BytesIO来处理内存中的二进制数据流，避免创建临时文件
        docx_content = BytesIO(response.content)

        document = Document(docx_content)
        in_target_section = False
        outline_items = []
        current_heading = None
        content_for_current_heading = []

        for para in document.paragraphs:
            # 检查是否为一级标题
            if para.style.name.lower().startswith("heading 1"):
                if para.text == target_heading_text:
                    in_target_section = True
                else:
                    # 如果找到了另一个一级标题且已经在目标部分内，则停止收集
                    if in_target_section:
                        break
                    in_target_section = False

            # 只有在目标部分内时才收集内容
            if in_target_section:
                if para.style.name.lower().startswith("heading"):  # 如果是任何级别的标题
                    if current_heading is not None:
                        outline_items.append((current_heading, content_for_current_heading))
                    current_heading = para.text
                    content_for_current_heading = []
                else:
                    if current_heading is not None:
                        content_for_current_heading.append(para.text)

        # 添加最后一个大纲项及其内容（如果有的话）
        if current_heading is not None:
            outline_items.append((current_heading, content_for_current_heading))

        return outline_items
