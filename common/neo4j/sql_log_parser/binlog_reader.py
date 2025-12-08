#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MySQL Binlog读取器
实时读取MySQL binlog，提取SQL语句
"""

import pymysql
from typing import List, Dict, Optional, Callable
import logging

logger = logging.getLogger(__name__)

try:
    from pymysqlreplication import BinLogStreamReader
    from pymysqlreplication.row_event import (
        DeleteRowsEvent,
        UpdateRowsEvent,
        WriteRowsEvent,
    )
    from pymysqlreplication.event import QueryEvent
    BINLOG_AVAILABLE = True
except ImportError:
    BINLOG_AVAILABLE = False
    logger.warning("pymysql-replication not installed. Install with: pip install pymysql-replication")


class BinlogReader:
    """MySQL Binlog读取器"""

    def __init__(
        self,
        mysql_config: Dict,
        server_id: int = 100,
        only_events: Optional[List] = None,
        only_tables: Optional[List] = None,
        only_schemas: Optional[List] = None,
    ):
        """
        初始化Binlog读取器
        :param mysql_config: MySQL配置字典，包含host, port, user, password
        :param server_id: 服务器ID（用于binlog复制）
        :param only_events: 只监听的事件类型列表，如[QueryEvent, WriteRowsEvent]
        :param only_tables: 只监听的表列表
        :param only_schemas: 只监听的数据库列表
        """
        if not BINLOG_AVAILABLE:
            raise ImportError(
                "pymysql-replication is required. Install with: pip install pymysql-replication"
            )

        self.mysql_config = mysql_config
        self.server_id = server_id
        self.only_events = only_events
        self.only_tables = only_tables
        self.only_schemas = only_schemas
        self.stream = None
        self.sql_statements = []
        self.is_running = False

    def get_binlog_position(self) -> Dict[str, str]:
        """
        获取当前binlog位置
        :return: 包含file和position的字典
        """
        try:
            conn = pymysql.connect(
                host=self.mysql_config.get("host", "localhost"),
                port=self.mysql_config.get("port", 3306),
                user=self.mysql_config.get("user", "root"),
                password=self.mysql_config.get("password", ""),
                charset="utf8mb4",
            )

            with conn.cursor() as cursor:
                cursor.execute("SHOW MASTER STATUS")
                result = cursor.fetchone()

                if result:
                    return {"file": result[0], "position": result[1]}
                else:
                    # 如果没有主从复制，尝试获取binlog文件列表
                    cursor.execute("SHOW BINARY LOGS")
                    logs = cursor.fetchall()
                    if logs:
                        return {"file": logs[-1][0], "position": 4}  # 从最后一个文件开始
                    return {"file": None, "position": 4}

            conn.close()
        except Exception as e:
            logger.error(f"获取binlog位置失败: {str(e)}")
            return {"file": None, "position": 4}

    def _extract_sql_from_query_event(self, event: QueryEvent) -> Optional[str]:
        """
        从QueryEvent中提取SQL语句
        :param event: QueryEvent对象
        :return: SQL语句
        """
        sql = event.query
        # 过滤掉非业务SQL
        if sql and self._is_valid_sql(sql):
            return sql
        return None

    def _extract_sql_from_row_event(
        self, event, event_type: str
    ) -> List[Dict[str, str]]:
        """
        从行事件中生成SQL语句
        :param event: 行事件对象（WriteRowsEvent/UpdateRowsEvent/DeleteRowsEvent）
        :param event_type: 事件类型（INSERT/UPDATE/DELETE）
        :return: SQL语句列表（包含表名和SQL）
        """
        sql_statements = []
        table_name = event.table

        for row in event.rows:
            if event_type == "INSERT":
                # 生成INSERT语句
                if hasattr(row, "values"):
                    values = row["values"]
                    columns = list(values.keys())
                    values_list = [self._format_value(v) for v in values.values()]
                    sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({', '.join(values_list)})"
                    sql_statements.append({"sql": sql, "table": table_name, "type": "INSERT"})

            elif event_type == "UPDATE":
                # 生成UPDATE语句
                if hasattr(row, "before_values") and hasattr(row, "after_values"):
                    before = row["before_values"]
                    after = row["after_values"]
                    set_clause = ", ".join(
                        [f"{k} = {self._format_value(v)}" for k, v in after.items()]
                    )
                    where_clause = " AND ".join(
                        [f"{k} = {self._format_value(before.get(k))}" for k in before.keys()]
                    )
                    sql = f"UPDATE {table_name} SET {set_clause} WHERE {where_clause}"
                    sql_statements.append({"sql": sql, "table": table_name, "type": "UPDATE"})

            elif event_type == "DELETE":
                # 生成DELETE语句
                if hasattr(row, "values"):
                    values = row["values"]
                    where_clause = " AND ".join(
                        [f"{k} = {self._format_value(v)}" for k, v in values.items()]
                    )
                    sql = f"DELETE FROM {table_name} WHERE {where_clause}"
                    sql_statements.append({"sql": sql, "table": table_name, "type": "DELETE"})

        return sql_statements

    def _format_value(self, value) -> str:
        """格式化SQL值"""
        if value is None:
            return "NULL"
        elif isinstance(value, str):
            # 转义单引号
            value = value.replace("'", "''")
            return f"'{value}'"
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, bool):
            return "1" if value else "0"
        else:
            return f"'{str(value)}'"

    def _is_valid_sql(self, sql: str) -> bool:
        """判断是否为有效的业务SQL"""
        if not sql or len(sql.strip()) < 10:
            return False

        sql_upper = sql.upper().strip()
        skip_patterns = [
            "SHOW ",
            "DESCRIBE ",
            "DESC ",
            "EXPLAIN ",
            "USE ",
            "SET ",
            "FLUSH ",
            "RESET ",
            "GRANT ",
            "REVOKE ",
            "CREATE USER",
            "SELECT @@",
            "SELECT DATABASE()",
            "SELECT VERSION()",
            "SELECT NOW()",
            "BEGIN",
            "COMMIT",
            "ROLLBACK",
        ]

        for pattern in skip_patterns:
            if sql_upper.startswith(pattern):
                return False

        return True

    def read_binlog_realtime(
        self,
        callback: Optional[Callable] = None,
        log_file: Optional[str] = None,
        log_pos: Optional[int] = None,
        stop_after_seconds: Optional[int] = None,
    ) -> List[Dict]:
        """
        实时读取binlog
        :param callback: 回调函数，每读取到SQL时调用 callback(sql_data)
        :param log_file: 起始binlog文件名，如果为None则从当前位置开始
        :param log_pos: 起始binlog位置，如果为None则从当前位置开始
        :param stop_after_seconds: 读取多少秒后停止，如果为None则持续读取
        :return: 读取到的SQL语句列表
        """
        if not BINLOG_AVAILABLE:
            raise ImportError("pymysql-replication is required")

        import time

        # 获取binlog位置
        if log_file is None or log_pos is None:
            position = self.get_binlog_position()
            log_file = log_file or position["file"]
            log_pos = log_pos or position["position"]

        if log_file is None:
            raise ValueError("无法获取binlog文件位置，请手动指定log_file和log_pos")

        print(f"📖 开始读取binlog: {log_file}@{log_pos}")
        print("💡 提示: 按Ctrl+C停止读取")

        self.sql_statements = []
        self.is_running = True
        start_time = time.time()

        try:
            # 创建binlog流
            self.stream = BinLogStreamReader(
                connection_settings={
                    "host": self.mysql_config.get("host", "localhost"),
                    "port": self.mysql_config.get("port", 3306),
                    "user": self.mysql_config.get("user", "root"),
                    "passwd": self.mysql_config.get("password", ""),
                },
                server_id=self.server_id,
                log_file=log_file,
                log_pos=log_pos,
                only_events=self.only_events,
                only_tables=self.only_tables,
                only_schemas=self.only_schemas,
                resume_stream=True,  # 支持断点续传
            )

            sql_count = 0

            for binlogevent in self.stream:
                if not self.is_running:
                    break

                # 检查是否超时
                if stop_after_seconds and (time.time() - start_time) > stop_after_seconds:
                    print(f"\n⏰ 已读取 {stop_after_seconds} 秒，停止读取")
                    break

                sql_data = None

                # 处理QueryEvent（DDL和部分DML）
                if isinstance(binlogevent, QueryEvent):
                    sql = self._extract_sql_from_query_event(binlogevent)
                    if sql:
                        sql_data = {
                            "sql": sql,
                            "table": None,
                            "type": "QUERY",
                            "timestamp": binlogevent.timestamp,
                            "log_file": binlogevent.log_file,
                            "log_pos": binlogevent.log_pos,
                        }
                        sql_count += 1
                        print(f"  📝 [{sql_count}] {sql[:100]}...")

                # 处理WriteRowsEvent（INSERT）
                elif isinstance(binlogevent, WriteRowsEvent):
                    sqls = self._extract_sql_from_row_event(binlogevent, "INSERT")
                    for sql_info in sqls:
                        sql_data = {
                            "sql": sql_info["sql"],
                            "table": sql_info["table"],
                            "type": "INSERT",
                            "timestamp": binlogevent.timestamp,
                            "log_file": binlogevent.log_file,
                            "log_pos": binlogevent.log_pos,
                        }
                        sql_count += 1
                        print(f"  📝 [{sql_count}] INSERT {sql_info['table']}")

                # 处理UpdateRowsEvent（UPDATE）
                elif isinstance(binlogevent, UpdateRowsEvent):
                    sqls = self._extract_sql_from_row_event(binlogevent, "UPDATE")
                    for sql_info in sqls:
                        sql_data = {
                            "sql": sql_info["sql"],
                            "table": sql_info["table"],
                            "type": "UPDATE",
                            "timestamp": binlogevent.timestamp,
                            "log_file": binlogevent.log_file,
                            "log_pos": binlogevent.log_pos,
                        }
                        sql_count += 1
                        print(f"  📝 [{sql_count}] UPDATE {sql_info['table']}")

                # 处理DeleteRowsEvent（DELETE）
                elif isinstance(binlogevent, DeleteRowsEvent):
                    sqls = self._extract_sql_from_row_event(binlogevent, "DELETE")
                    for sql_info in sqls:
                        sql_data = {
                            "sql": sql_info["sql"],
                            "table": sql_info["table"],
                            "type": "DELETE",
                            "timestamp": binlogevent.timestamp,
                            "log_file": binlogevent.log_file,
                            "log_pos": binlogevent.log_pos,
                        }
                        sql_count += 1
                        print(f"  📝 [{sql_count}] DELETE {sql_info['table']}")

                # 如果有SQL数据，保存并调用回调
                if sql_data:
                    self.sql_statements.append(sql_data)
                    if callback:
                        callback(sql_data)

        except KeyboardInterrupt:
            print("\n\n⚠️  用户中断读取")
        except Exception as e:
            logger.error(f"读取binlog失败: {str(e)}")
            raise
        finally:
            self.is_running = False
            if self.stream:
                self.stream.close()

        print(f"\n✅ 读取完成，共提取 {len(self.sql_statements)} 条SQL语句")
        return self.sql_statements

    def stop(self):
        """停止读取"""
        self.is_running = False
        if self.stream:
            self.stream.close()

    def get_sql_statements(self) -> List[Dict]:
        """获取所有收集的SQL语句"""
        return self.sql_statements

    def clear_statements(self):
        """清空已收集的SQL语句"""
        self.sql_statements = []

