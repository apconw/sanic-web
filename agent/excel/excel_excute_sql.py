import io
import logging
import traceback
import duckdb
import requests
import pandas as pd
from typing import Dict, List

from agent.excel.excel_agent_state import ExecutionResult
from common.minio_util import MinioUtils

logger = logging.getLogger(__name__)

minio_util = MinioUtils()


def setup_multi_catalog_duckdb(file_metadata: Dict, catalog_info: Dict) -> duckdb.DuckDBPyConnection:
    """
    设置多catalog DuckDB连接，注册所有文件和Sheet
    """
    # 创建DuckDB连接
    con = duckdb.connect(database=":memory:")

    # 安装并加载必要的扩展
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")

    logger.info(f"开始注册 {len(file_metadata)} 个文件到DuckDB catalogs")

    # 为每个文件创建catalog并注册所有Sheet
    for file_key, file_info in file_metadata.items():
        catalog_name = file_info.catalog_name
        file_url = file_info.file_path

        try:
            # 获取文件扩展名
            extension = file_key.split(".")[-1].lower()

            if extension in ["xlsx", "xls"]:
                # 处理Excel文件
                excel_file_data = pd.ExcelFile(file_url)

                for sheet_name in excel_file_data.sheet_names:
                    try:
                        # 读取Sheet数据
                        df = pd.read_excel(file_url, sheet_name=sheet_name)

                        if df.empty:
                            continue

                        # 清理列名
                        df.columns = [str(col).replace(" ", "_").replace("-", "_") for col in df.columns]

                        # 生成表名
                        table_name = sheet_name.replace(" ", "_").replace("-", "_")

                        # 创建catalog中的表
                        con.execute(f"CREATE TABLE {catalog_name}.{table_name} AS SELECT * FROM df")

                        logger.info(f"成功注册表: {catalog_name}.{table_name} ({len(df)} 行)")

                    except Exception as e:
                        logger.error(f"注册Sheet '{sheet_name}' 到catalog '{catalog_name}' 失败: {str(e)}")
                        continue

            elif extension == "csv":
                # 处理CSV文件
                try:
                    df = pd.read_csv(file_url)

                    if df.empty:
                        continue

                    # 清理列名
                    df.columns = [str(col).replace(" ", "_").replace("-", "_") for col in df.columns]

                    # 生成表名
                    table_name = file_key.split("/")[-1].split(".")[0].replace(" ", "_").replace("-", "_")

                    # 创建catalog中的表
                    con.execute(f"CREATE TABLE {catalog_name}.{table_name} AS SELECT * FROM df")

                    logger.info(f"成功注册表: {catalog_name}.{table_name} ({len(df)} 行)")

                except Exception as e:
                    logger.error(f"注册CSV文件到catalog '{catalog_name}' 失败: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"处理文件 '{file_key}' 失败: {str(e)}")
            continue

    logger.info("DuckDB多catalog环境设置完成")
    return con


def exe_sql_excel_query(state):
    """
    执行SQL查询 - 支持多文件多Sheet的跨catalog查询
    :param state: ExcelAgentState
    :return: 更新后的ExcelAgentState
    """
    try:
        # 检查必要的字段
        if "generated_sql" not in state or not state["generated_sql"]:
            raise ValueError("没有找到有效的SQL查询语句")

        if state["generated_sql"] == "No SQL query generated":
            raise ValueError("SQL生成失败，无法执行查询")

        # 获取元数据信息
        file_metadata = state.get("file_metadata", {})
        catalog_info = state.get("catalog_info", {})

        if not file_metadata:
            raise ValueError("文件元数据为空，无法设置DuckDB环境")

        # 设置多catalog DuckDB环境
        con = setup_multi_catalog_duckdb(file_metadata, catalog_info)

        # 获取SQL查询语句
        sql = state["generated_sql"].replace("`", "")  # 移除反引号以避免SQL语法错误

        logger.info(f"执行SQL查询: {sql}")

        # 执行SQL查询
        cursor = con.execute(sql)

        # 获取列名称和查询结果的数据行
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()

        # 构建结果字典
        result = [dict(zip(columns, row)) for row in rows]

        # 成功情况
        state["execution_result"] = ExecutionResult(
            success=True,
            columns=columns,
            data=result
        )

        logger.info(f"查询执行成功: 返回 {len(result)} 行, {len(columns)} 列")

        # 关闭连接
        con.close()

    except Exception as e:
        error_msg = str(e)
        logger.error(f"执行SQL查询失败: {error_msg}")
        traceback.print_exception(e)

        state["execution_result"] = ExecutionResult(
            success=False,
            columns=[],
            data=[],
            error=error_msg
        )

    return state
