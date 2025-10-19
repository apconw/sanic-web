"""
DuckDB 连接管理器
统一管理 Excel Agent 中的 DuckDB 连接和数据注册，避免重复创建和注册
"""

import logging
import re
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import duckdb
import pandas as pd

from agent.excel.excel_agent_state import FileInfo, SheetInfo

logger = logging.getLogger(__name__)


class ExcelDuckDBManager:
    """
    Excel DuckDB 连接管理器
    - 统一管理 DuckDB 连接
    - 避免重复创建连接和注册数据
    - 支持多文件、多Sheet的数据管理
    """

    def __init__(self):
        self._connection: Optional[duckdb.DuckDBPyConnection] = None
        self._registered_catalogs: Dict[str, str] = {}  # {catalog_name: file_path}
        self._registered_tables: Dict[str, SheetInfo] = {}  # {table_name: SheetInfo}
        self._session_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """
        获取 DuckDB 连接，延迟初始化
        """
        if self._connection is None:
            logger.info("创建新的 DuckDB 连接")
            self._connection = duckdb.connect(database=":memory:")

            # 安装并加载必要的扩展
            self._connection.execute("INSTALL httpfs")
            self._connection.execute("LOAD httpfs")

            logger.info("DuckDB 连接创建完成，已加载 httpfs 扩展")

        return self._connection

    def _sanitize_catalog_name(self, file_name: str) -> str:
        """
        清理文件名，生成合法的 DuckDB catalog 名称
        """
        # 移除文件扩展名
        name_without_ext = file_name.split('/')[-1]
        name_without_ext = name_without_ext.rsplit('.',1)[0]

        # 替换非法字符
        catalog_name = re.sub(r'[^\w\u4e00-\u9fa5]', '_', name_without_ext)
        # 移除开头和结尾的下划线
        catalog_name = catalog_name.strip('_')
        # 确保不以数字开头
        if catalog_name and catalog_name[0].isdigit():
            catalog_name = f'catalog_{catalog_name}'
        return catalog_name or 'unknown_catalog'

    def _sanitize_table_name(self, sheet_name: str) -> str:
        """
        清理 Sheet 名称，生成合法的表名
        """
        # 替换非法字符
        table_name = re.sub(r'[^\w\u4e00-\u9fa5]', '_', sheet_name)
        # 移除开头和结尾的下划线
        table_name = table_name.strip('_')
        # 确保不以数字开头
        if table_name and table_name[0].isdigit():
            table_name = f'table_{table_name}'
        return table_name or 'unknown_sheet'

    def _sanitize_column_name(self, column_name: str) -> str:
        """
        清理列名，生成合法的列名
        """
        # 替换非法字符
        col_name = re.sub(r'[^\w\u4e00-\u9fa5]', '_', str(column_name))
        # 移除开头和结尾的下划线
        col_name = col_name.strip('_')
        # 确保不以数字开头
        if col_name and col_name[0].isdigit():
            col_name = f'column_{col_name}'
        return col_name or 'unknown_column'

    def _register_dataframes_to_catalog(self,
                                    dataframes: List[Tuple[str, pd.DataFrame]],
                                    catalog_name: str,
                                    file_name: str) -> Dict[str, SheetInfo]:
        """
        将多个 DataFrame 注册到指定的 catalog 中

        :param dataframes: List[(sheet_name, DataFrame)] - 表名和数据框的列表
        :param catalog_name: 目标 catalog 名称
        :param file_name: 源文件名（用于日志）
        :return: {table_name: SheetInfo}
        """
        conn = self._get_connection()
        registered_tables = {}

        # 创建schema（如果不存在）
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}")

        for sheet_name, df in dataframes:
            try:
                # 生成表名
                table_name = self._sanitize_table_name(sheet_name)
                full_table_name = f'"{catalog_name}"."{table_name}"'

                # 检查表是否已注册
                if full_table_name in self._registered_tables:
                    logger.warning(f"表 '{full_table_name}' 已存在，跳过注册")
                    continue

                if df.empty:
                    logger.warning(f"表 '{sheet_name}' 为空，跳过注册")
                    continue

                # 清理列名
                df.columns = [self._sanitize_column_name(col) for col in df.columns]

                # 创建表并插入数据
                create_sql = f"CREATE TABLE {full_table_name} AS SELECT * FROM df"
                conn.execute(create_sql)

                # 获取表信息
                row_count = len(df)
                column_count = len(df.columns)

                # 获取列信息
                columns_info = {}
                for col in df.columns:
                    dtype = str(df[col].dtype)
                    sql_type = self._map_pandas_dtype_to_sql(dtype)
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

                registered_tables[table_name] = sheet_info
                self._registered_tables[full_table_name] = sheet_info

                logger.info(f"成功注册表: {full_table_name} ({row_count} 行, {column_count} 列)")

            except Exception as e:
                logger.error(f"注册表 '{sheet_name}' 失败: {str(e)}")
                traceback.print_exception(e)
                continue

        return registered_tables

    def _get_unique_catalog_name(self, file_name: str) -> str:
        """
        获取唯一的 catalog 名称

        :param file_name: 文件名
        :return: 唯一的 catalog 名称
        """
        catalog_name = self._sanitize_catalog_name(file_name)

        # 确保 catalog 名称唯一
        original_catalog_name = catalog_name
        counter = 1
        while catalog_name in self._registered_catalogs:
            catalog_name = f"{original_catalog_name}_{counter}"
            counter += 1

        return catalog_name

    def register_excel_file(self, file_path: str, file_name: str) -> Tuple[str, Dict[str, SheetInfo]]:
        """
        注册 Excel 文件到 DuckDB，返回 catalog 名称和表信息

        :param file_path: 文件路径或URL
        :param file_name: 文件名
        :return: (catalog_name, {table_name: SheetInfo})
        """
        catalog_name = self._get_unique_catalog_name(file_name)
        logger.info(f"开始注册Excel文件到 catalog '{catalog_name}': {file_name}")

        try:
            # 读取 Excel 文件的所有 sheet
            excel_file_data = pd.ExcelFile(file_path)
            sheet_names = excel_file_data.sheet_names

            # 构建数据框列表
            dataframes = []
            for sheet_name in sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                dataframes.append((sheet_name, df))

            # 注册到 catalog
            registered_tables = self._register_dataframes_to_catalog(dataframes, catalog_name, file_name)

            # 记录 catalog 信息
            self._registered_catalogs[catalog_name] = file_path
            logger.info(f"成功注册Excel文件 '{file_name}' 到 catalog '{catalog_name}'，共 {len(registered_tables)} 个表")

        except Exception as e:
            logger.error(f"注册Excel文件 '{file_name}' 失败: {str(e)}")
            traceback.print_exception(e)
            raise

        return catalog_name, registered_tables

    def register_csv_file(self, file_path: str, file_name: str) -> Tuple[str, Dict[str, SheetInfo]]:
        """
        注册 CSV 文件到 DuckDB

        :param file_path: 文件路径或URL
        :param file_name: 文件名
        :return: (catalog_name, {table_name: SheetInfo})
        """
        catalog_name = self._get_unique_catalog_name(file_name)
        logger.info(f"开始注册CSV文件到 catalog '{catalog_name}': {file_name}")

        try:
            # 读取 CSV 文件
            df = pd.read_csv(file_path)

            if df.empty:
                logger.warning(f"CSV文件 '{file_name}' 为空")
                return catalog_name, {}

            # 生成表名（使用文件名去掉扩展名）
            table_name = self._sanitize_table_name(file_name.rsplit('.', 1)[0])

            # 构建数据框列表
            dataframes = [(table_name, df)]

            # 注册到 catalog
            registered_tables = self._register_dataframes_to_catalog(dataframes, catalog_name, file_name)

            # 记录 catalog 信息
            self._registered_catalogs[catalog_name] = file_path
            logger.info(f"成功注册CSV文件 '{file_name}' 到 catalog '{catalog_name}'，共 {len(registered_tables)} 个表")

        except Exception as e:
            logger.error(f"注册CSV文件 '{file_name}' 失败: {str(e)}")
            traceback.print_exception(e)
            raise

        return catalog_name, registered_tables

    def execute_sql(self, sql: str) -> Tuple[List[str], List[Dict]]:
        """
        执行 SQL 查询

        :param sql: SQL 查询语句
        :return: (columns, data)
        """
        conn = self._get_connection()

        try:
            logger.info(f"执行SQL查询: {sql}")
            cursor = conn.execute(sql)

            # 获取列名称和查询结果的数据行
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()

            # 构建结果字典
            result = [dict(zip(columns, row)) for row in rows]

            logger.info(f"查询执行成功: 返回 {len(result)} 行, {len(columns)} 列")
            return columns, result

        except Exception as e:
            logger.error(f"SQL查询执行失败: {str(e)}")
            raise

    def get_registered_catalogs(self) -> Dict[str, str]:
        """
        获取已注册的 catalog 信息
        """
        return self._registered_catalogs.copy()

    def get_registered_tables(self) -> Dict[str, SheetInfo]:
        """
        获取已注册的表信息
        """
        return self._registered_tables.copy()

    def get_table_schema_info(self) -> List[Dict]:
        """
        获取所有表的架构信息，用于SQL生成
        """
        schema_info = []

        for full_table_name, sheet_info in self._registered_tables.items():
            catalog_name, table_name = full_table_name.split('.', 1)

            table_schema = {
                "table_name": full_table_name,
                "columns": sheet_info.columns_info,
                "foreign_keys": [],
                "table_comment": f"{catalog_name} - {sheet_info.sheet_name}",
                "catalog_name": catalog_name
            }
            schema_info.append(table_schema)

        return schema_info

    def close(self):
        """
        关闭 DuckDB 连接
        """
        if self._connection is not None:
            self._connection.close()
            self._connection = None
            logger.info("DuckDB 连接已关闭")

    def clear_session(self):
        """
        清理当前会话数据
        """
        self.close()
        self._registered_catalogs.clear()
        self._registered_tables.clear()
        logger.info(f"会话 {self._session_id} 数据已清理")

    def _map_pandas_dtype_to_sql(self, dtype: str) -> str:
        """
        将 pandas 数据类型映射到 SQL 数据类型
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


class ChatDuckDBManager:
    """
    聊天级别的DuckDB管理器
    为每个chat_id维护独立的ExcelDuckDBManager实例
    """

    def __init__(self):
        # {chat_id: ExcelDuckDBManager}
        self._chat_managers: Dict[str, ExcelDuckDBManager] = {}
        # 会话清理时间配置（秒）
        self._session_timeout = 36000  # 10小时
        # {chat_id: 最后访问时间}
        self._last_access: Dict[str, float] = {}
        logger.info("初始化聊天级别DuckDB管理器")

    def get_manager(self, chat_id: str) -> ExcelDuckDBManager:
        """
        获取指定chat_id的DuckDB管理器实例

        :param chat_id: 聊天ID
        :return: ExcelDuckDBManager实例
        """
        # 检查是否已存在该chat_id的管理器
        if chat_id not in self._chat_managers:
            self._chat_managers[chat_id] = ExcelDuckDBManager()
            logger.info(f"为chat_id '{chat_id}' 创建新的DuckDB管理器实例")

        # 更新最后访问时间
        self._last_access[chat_id] = time.time()

        return self._chat_managers[chat_id]

    def close_manager(self, chat_id: str) -> bool:
        """
        关闭指定chat_id的DuckDB管理器

        :param chat_id: 聊天ID
        :return: 是否成功关闭
        """
        if chat_id in self._chat_managers:
            try:
                self._chat_managers[chat_id].close()
                del self._chat_managers[chat_id]
                if chat_id in self._last_access:
                    del self._last_access[chat_id]
                logger.info(f"已关闭chat_id '{chat_id}' 的DuckDB管理器实例")
                return True
            except Exception as e:
                logger.error(f"关闭chat_id '{chat_id}' 的DuckDB管理器失败: {str(e)}")
                return False
        return False

    def cleanup_expired_sessions(self):
        """
        清理过期的会话
        """
        current_time = time.time()
        expired_chats = []

        for chat_id, last_access in self._last_access.items():
            if current_time - last_access > self._session_timeout:
                expired_chats.append(chat_id)

        for chat_id in expired_chats:
            logger.info(f"清理过期会话: {chat_id}")
            self.close_manager(chat_id)

    def get_active_chat_count(self) -> int:
        """
        获取活跃的聊天数量

        :return: 活跃聊天数量
        """
        return len(self._chat_managers)

    def get_chat_list(self) -> List[str]:
        """
        获取所有活跃的chat_id列表

        :return: chat_id列表
        """
        return list(self._chat_managers.keys())

    def close_all(self):
        """
        关闭所有聊天会话的DuckDB管理器
        """
        for chat_id in list(self._chat_managers.keys()):
            self.close_manager(chat_id)
        logger.info("已关闭所有聊天会话的DuckDB管理器实例")


# 全局聊天级别管理器实例
_chat_duckdb_manager: Optional[ChatDuckDBManager] = None


def get_chat_duckdb_manager() -> ChatDuckDBManager:
    """
    获取全局聊天级别DuckDB管理器实例（单例模式）

    :return: ChatDuckDBManager实例
    """
    global _chat_duckdb_manager
    if _chat_duckdb_manager is None:
        _chat_duckdb_manager = ChatDuckDBManager()
        logger.info("创建全局聊天级别DuckDB管理器实例")
    return _chat_duckdb_manager


def get_duckdb_manager(chat_id: str = None) -> ExcelDuckDBManager:
    """
    获取DuckDB管理器实例

    :param chat_id: 聊天ID，如果提供则获取chat_id级别的管理器，否则使用全局默认管理器
    :return: ExcelDuckDBManager实例
    """
    if chat_id:
        # 使用chat_id级别的管理器
        chat_manager = get_chat_duckdb_manager()
        return chat_manager.get_manager(chat_id)
    else:
        # 向后兼容：如果没有提供chat_id，使用默认的全局管理器
        return get_default_duckdb_manager()


def get_default_duckdb_manager() -> ExcelDuckDBManager:
    """
    获取默认的全局DuckDB管理器实例（向后兼容）

    :return: ExcelDuckDBManager实例
    """
    # 创建一个默认的管理器实例
    if not hasattr(get_default_duckdb_manager, '_default_manager'):
        get_default_duckdb_manager._default_manager = ExcelDuckDBManager()
        logger.info("创建默认全局DuckDB管理器实例")
    return get_default_duckdb_manager._default_manager


def close_duckdb_manager(chat_id: str = None):
    """
    关闭DuckDB管理器

    :param chat_id: 聊天ID，如果提供则关闭指定chat_id的管理器，否则关闭默认管理器
    """
    if chat_id:
        # 关闭指定chat_id的管理器
        chat_manager = get_chat_duckdb_manager()
        chat_manager.close_manager(chat_id)
    else:
        # 向后兼容：关闭默认管理器
        if hasattr(get_default_duckdb_manager, '_default_manager'):
            get_default_duckdb_manager._default_manager.close()
            get_default_duckdb_manager._default_manager = None
            logger.info("默认全局DuckDB管理器已关闭")