import json
import logging
import os
from datetime import datetime
from typing import Dict, List
import traceback

import pandas as pd

from agent.excel.excel_agent_state import ExcelAgentState, FileInfo
from agent.excel.excel_duckdb_manager import get_duckdb_manager
from common.minio_util import MinioUtils

minio_utils = MinioUtils()

# 日志配置
logger = logging.getLogger(__name__)

# 支持的文件扩展名
SUPPORTED_EXTENSIONS = {"xlsx", "xls", "csv"}


def read_excel_columns(state: ExcelAgentState) -> ExcelAgentState:
    """
    读取多个Excel文件的所有sheet，生成表结构信息并注册到 DuckDB
    支持多文件、多Sheet的统一分析

    :param state: ExcelAgentState对象，包含file_list等信息
    :return: 更新后的ExcelAgentState
    """
    file_list = state["file_list"]

    try:
        # 检查文件列表是否为空
        if not file_list or len(file_list) == 0:
            raise ValueError("文件列表为空")

        # 初始化元数据存储
        file_metadata = {}
        sheet_metadata = {}
        catalog_info = {}
        all_db_info = []

        # 获取chat_id，优先从state中获取
        chat_id = state.get("chat_id")

        # 获取DuckDB管理器实例
        duckdb_manager = get_duckdb_manager(chat_id=chat_id)

        logger.info(f"开始处理 {len(file_list)} 个文件")

        # 处理每个文件
        for file_idx, file_info in enumerate(file_list):
            try:
                source_file_key = file_info.get("source_file_key")
                if not source_file_key:
                    logger.warning(f"文件 {file_idx} 缺少 source_file_key 字段，跳过")
                    continue

                # 获取文件信息
                file_name = os.path.basename(source_file_key)
                file_url = minio_utils.get_file_url_by_key(object_key=source_file_key)

                # 解析文件扩展名
                path_parts = source_file_key.split(".")
                extension = path_parts[-1].lower() if len(path_parts) > 1 else ""

                # 验证文件扩展名
                if extension not in SUPPORTED_EXTENSIONS:
                    logger.warning(f"文件 {file_name} 扩展名不支持: {extension}，跳过")
                    continue

                # 创建文件信息
                file_info_obj = FileInfo(
                    file_name=file_name,
                    file_path=file_url,
                    catalog_name="",  # 将在注册后填充
                    sheet_count=0,  # 将在注册后填充
                    upload_time=datetime.now().isoformat(),
                )

                registered_tables = {}
                catalog_name = ""
                if extension in ["xlsx", "xls"]:
                    # 注册到 DuckDB 管理器
                    catalog_name, registered_tables = duckdb_manager.register_excel_file(file_url, file_name)

                elif extension == "csv":
                    # 注册到 DuckDB 管理器
                    catalog_name, registered_tables = duckdb_manager.register_csv_file(file_url, file_name)

                # 更新文件信息
                file_info_obj.catalog_name = catalog_name
                file_info_obj.sheet_count = len(registered_tables)

                # 合并表元数据
                sheet_metadata.update(registered_tables)

                # 生成表结构信息
                for table_name, sheet_info in registered_tables.items():

                    table_schema = {
                        "table_name": table_name,
                        "catalog_name": catalog_name,
                        "columns": sheet_info.columns_info,
                        "foreign_keys": [],
                        "table_comment": f"{file_name} - {sheet_info.sheet_name}",
                        "sample_data": sheet_info.sample_data,
                    }
                    all_db_info.append(table_schema)

                # 保存元数据
                file_metadata[source_file_key] = file_info_obj
                catalog_info[catalog_name] = source_file_key

                logger.info(f"成功处理文件 {file_name}: catalog={catalog_name}, sheets={file_info_obj.sheet_count}")

            except Exception as e:
                logger.error(f"处理文件 {file_idx} 失败: {str(e)}")
                continue

        # 更新状态
        state["file_metadata"] = file_metadata
        state["sheet_metadata"] = sheet_metadata
        state["catalog_info"] = catalog_info
        state["db_info"] = all_db_info

        logger.info(f"处理完成: {len(file_metadata)} 个文件, {len(sheet_metadata)} 个表")
        logger.info(f"生成的表结构: {json.dumps(all_db_info, default=json_serializer, ensure_ascii=False, indent=2)}")
    except Exception as e:
        traceback.print_exception(e)
        logger.error(f"读取Excel表列信息出错: {str(e)}", exc_info=True)
        raise ValueError(f"读取文件列信息时发生错误: {str(e)}") from e

    return state


def json_serializer(obj):
    """处理不可直接JSON序列化的对象"""
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif hasattr(obj, "isoformat"):
        # 处理其他可能的时间类型对象
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
