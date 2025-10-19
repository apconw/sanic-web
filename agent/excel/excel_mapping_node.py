import json
import logging
import os
import re
import traceback
from datetime import datetime
from typing import Dict, List

import duckdb
import pandas as pd

from agent.excel.excel_agent_state import ExcelAgentState, FileInfo, SheetInfo
from common.minio_util import MinioUtils

minio_utils = MinioUtils()

# 日志配置
logger = logging.getLogger(__name__)

# 支持的文件扩展名
SUPPORTED_EXTENSIONS = {"xlsx", "xls", "csv"}


# 创建 DuckDB 连接
conn = duckdb.connect(':memory:')

# 安装并加载 Excel 扩展
conn.execute("INSTALL httpfs")
conn.execute("LOAD httpfs")




# 数据类型映射
def map_pandas_dtype_to_sql(dtype: str) -> str:
    """
    将 pandas 数据类型映射到 SQL 数据类型

    :param dtype: pandas 数据类型
    :return: SQL 数据类型
    """
    dtype_mapping = {
        "object": "VARCHAR(255)",
        "int64": "BIGINT",
        "int32": "INTEGER",
        "float64": "FLOAT",
        "float32": "FLOAT",
        "bool": "BOOLEAN",
        "datetime64[ns]": "DATETIME",
        "timedelta64[ns]": "VARCHAR(50)",
    }

    # 处理字符串类型
    if dtype.startswith("object"):
        return "VARCHAR(255)"
    # 处理整数类型
    elif dtype.startswith("int"):
        return dtype_mapping.get(dtype, "BIGINT")
    # 处理浮点数类型
    elif dtype.startswith("float"):
        return dtype_mapping.get(dtype, "FLOAT")
    # 处理日期时间类型
    elif dtype.startswith("datetime"):
        return "DATETIME"
    else:
        return "VARCHAR(255)"


def sanitize_catalog_name(file_name: str) -> str:
    """
    清理文件名，生成合法的 DuckDB catalog 名称
    """
    # 移除文件扩展名
    name_without_ext = os.path.splitext(file_name)[0]
    # 替换非法字符
    catalog_name = re.sub(r'[^\w\u4e00-\u9fa5]', '_', name_without_ext)
    # 移除开头和结尾的下划线
    catalog_name = catalog_name.strip('_')
    # 确保不以数字开头
    if catalog_name and catalog_name[0].isdigit():
        catalog_name = f'catalog_{catalog_name}'
    return catalog_name or 'unknown_catalog'


def sanitize_table_or_column_name(origin_name: str, is_column_name  = False) -> str:
    """
    清理 Sheet 或者列 名称，生成合法的名称
    """
    # 替换非法字符
    legal_name = re.sub(r'[^\w\u4e00-\u9fa5]', '_', origin_name)
    # 移除开头和结尾的下划线
    legal_name = legal_name.strip('_')
    # 确保不以数字开头
    if legal_name and legal_name[0].isdigit():
        legal_name = f'{"column" if is_column_name else "table" }_{legal_name}'
    return legal_name or 'unknown_sheet'


def register_excel_to_duckdb(file_path: str, catalog_name: str, sheet_names: List[str]) -> Dict[str, SheetInfo]:
    """
    将 Excel 文件的所有 Sheet 注册到 DuckDB catalog 中
    """
    sheet_metadata = {}

    try:


        for sheet_name in sheet_names:
            try:
                # 生成表名
                table_name = sanitize_table_or_column_name(sheet_name)

                # 读取 Sheet 数据
                df = pd.read_excel(file_path, sheet_name=sheet_name)

                if df.empty:
                    logger.warning(f"Sheet '{sheet_name}' 为空，跳过注册")
                    continue

                # 清理列名
                df.columns = [sanitize_table_or_column_name(str(col)) for col in df.columns]

                # 创建表并插入数据
                conn.execute(f"CREATE TABLE {catalog_name}.{table_name} AS SELECT * FROM df")

                # 获取表信息
                row_count = len(df)
                column_count = len(df.columns)

                # 获取列信息
                columns_info = {}
                for col in df.columns:
                    dtype = str(df[col].dtype)
                    sql_type = map_pandas_dtype_to_sql(dtype)
                    columns_info[col] = {
                        "comment": col,
                        "type": sql_type
                    }

                # 获取样本数据（前5行）
                sample_data = df.head(5).to_dict('records')

                # 创建 SheetInfo
                sheet_info = SheetInfo(
                    sheet_name=sheet_name,
                    table_name=table_name,
                    catalog_name=catalog_name,
                    row_count=row_count,
                    column_count=column_count,
                    columns_info=columns_info,
                    sample_data=sample_data
                )

                sheet_metadata[f"{catalog_name}.{table_name}"] = sheet_info
                logger.info(f"成功注册表: {catalog_name}.{table_name} ({row_count} 行, {column_count} 列)")

            except Exception as e:
                logger.error(f"注册 Sheet '{sheet_name}' 失败: {str(e)}")
                continue

        conn.close()

    except Exception as e:
        logger.error(f"注册文件到 DuckDB 失败: {str(e)}")
        raise

    return sheet_metadata


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

                # 生成 catalog 名称
                catalog_name = sanitize_catalog_name(file_name)

                # 确保 catalog 名称唯一
                original_catalog_name = catalog_name
                counter = 1
                while catalog_name in catalog_info:
                    catalog_name = f"{original_catalog_name}_{counter}"
                    counter += 1


                # 创建文件信息
                file_info_obj = FileInfo(
                    file_name=file_name,
                    file_path=file_url,
                    catalog_name=catalog_name,
                    sheet_count=0,
                    upload_time=datetime.now().isoformat()
                )

                if extension in ["xlsx", "xls"]:
                    # 处理 Excel 文件
                    excel_file_data = pd.ExcelFile(file_url)
                    sheet_names = excel_file_data.sheet_names

                    # 注册到 DuckDB
                    sheet_data = register_excel_to_duckdb(file_url, catalog_name, sheet_names)
                    sheet_metadata.update(sheet_data)

                    # 更新文件信息
                    file_info_obj.sheet_count = len(sheet_names)

                    # 生成表结构信息
                    for sheet_name in sheet_names:
                        table_name = sanitize_table_or_column_name(sheet_name)
                        if table_name in sheet_metadata:
                            sheet_info = sheet_metadata[table_name]

                            # 生成表结构信息
                            table_schema = {
                                "table_name": f"{catalog_name}.{table_name}",
                                "columns": sheet_info.columns_info,
                                "foreign_keys": [],
                                "table_comment": f"{file_name} - {sheet_name}",
                                "catalog_name": catalog_name
                            }
                            all_db_info.append(table_schema)

                elif extension == "csv":
                    # 处理 CSV 文件
                    df = pd.read_csv(file_url, nrows=5)
                    table_name = sanitize_table_or_column_name(os.path.splitext(file_name)[0])

                    # 注册到 DuckDB（将 CSV 作为单表处理）
                    sheet_data = register_excel_to_duckdb(file_url, catalog_name, [table_name])
                    sheet_metadata.update(sheet_data)

                    # 更新文件信息
                    file_info_obj.sheet_count = 1

                    # 生成表结构信息
                    if table_name in sheet_metadata:
                        sheet_info = sheet_metadata[table_name]
                        table_schema = {
                            "table_name": f"{catalog_name}.{table_name}",
                            "columns": sheet_info.columns_info,
                            "foreign_keys": [],
                            "table_comment": file_name,
                            "catalog_name": catalog_name
                        }
                        all_db_info.append(table_schema)

                # 保存元数据
                file_metadata[source_file_key] = file_info_obj
                catalog_info[catalog_name] = source_file_key

                logger.info(f"成功处理文件 {file_name}: catalog={catalog_name}, sheets={file_info_obj.sheet_count}")

            except Exception as e:
                logger.error(f"处理文件 {file_idx} 失败: {str(e)}")
                traceback.print_exception(e)
                continue

        # 更新状态
        state["file_metadata"] = file_metadata
        state["sheet_metadata"] = sheet_metadata
        state["catalog_info"] = catalog_info
        state["db_info"] = all_db_info

        logger.info(f"处理完成: {len(file_metadata)} 个文件, {len(sheet_metadata)} 个表")
        logger.info(f"生成的表结构: {json.dumps(all_db_info, ensure_ascii=False, indent=2)}")

    except Exception as e:
        traceback.print_exception(e)
        logger.error(f"读取Excel表列信息出错: {str(e)}", exc_info=True)
        raise ValueError(f"读取文件列信息时发生错误: {str(e)}") from e

    return state
