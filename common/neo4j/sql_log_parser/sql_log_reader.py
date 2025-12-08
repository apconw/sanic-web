#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MySQL SQL日志读取器
支持从以下来源读取SQL语句：
1. MySQL General Log 文件
2. MySQL Slow Query Log 文件
3. 自定义SQL日志文件
4. MySQL performance_schema.events_statements_history 表
"""

import os
import re
import pymysql
from typing import List, Dict, Optional
from datetime import datetime


class SQLLogReader:
    """SQL日志读取器"""

    def __init__(self, mysql_config: Optional[Dict] = None):
        """
        初始化SQL日志读取器
        :param mysql_config: MySQL配置字典，包含host, port, user, password, database
        """
        self.mysql_config = mysql_config or {}
        self.sql_statements = []

    def read_from_general_log_file(self, log_file_path: str) -> List[str]:
        """
        从MySQL General Log文件读取SQL语句
        :param log_file_path: General Log文件路径
        :return: SQL语句列表
        """
        sql_statements = []

        if not os.path.exists(log_file_path):
            print(f"❌ 日志文件不存在: {log_file_path}")
            return sql_statements

        print(f"📖 正在读取General Log文件: {log_file_path}")

        try:
            with open(log_file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # General Log格式示例：
            # 2025-01-01T10:00:00.000000Z    1 Query    SELECT * FROM users WHERE id = 1
            # 匹配Query行中的SQL语句
            pattern = r"\d{4}-\d{2}-\d{2}T[\d:\.]+Z\s+\d+\s+Query\s+(.+?)(?=\n\d{4}-\d{2}-\d{2}T|\Z)"
            matches = re.findall(pattern, content, re.DOTALL | re.MULTILINE)

            for match in matches:
                sql = match.strip()
                if sql and self._is_valid_sql(sql):
                    sql_statements.append(sql)

            print(f"✅ 从General Log读取到 {len(sql_statements)} 条SQL语句")

        except Exception as e:
            print(f"❌ 读取General Log文件失败: {str(e)}")

        return sql_statements

    def read_from_slow_query_log_file(self, log_file_path: str) -> List[str]:
        """
        从MySQL Slow Query Log文件读取SQL语句
        :param log_file_path: Slow Query Log文件路径
        :return: SQL语句列表
        """
        sql_statements = []

        if not os.path.exists(log_file_path):
            print(f"❌ 日志文件不存在: {log_file_path}")
            return sql_statements

        print(f"📖 正在读取Slow Query Log文件: {log_file_path}")

        try:
            with open(log_file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            current_sql = []
            in_sql = False

            for line in lines:
                # Slow Query Log格式：
                # # Time: 2025-01-01T10:00:00.000000Z
                # # User@Host: root[root] @ localhost []
                # # Query_time: 1.234  Lock_time: 0.000  Rows_sent: 10  Rows_examined: 100
                # SET timestamp=1234567890;
                # SELECT * FROM users WHERE id = 1;

                if line.startswith("# Time:") or line.startswith("# User@Host:"):
                    # 如果之前有SQL，先保存
                    if current_sql:
                        sql = " ".join(current_sql).strip()
                        if self._is_valid_sql(sql):
                            sql_statements.append(sql)
                        current_sql = []
                    in_sql = False
                elif line.startswith("SET timestamp="):
                    in_sql = True
                elif in_sql and not line.startswith("#"):
                    current_sql.append(line.strip())

            # 处理最后一条SQL
            if current_sql:
                sql = " ".join(current_sql).strip()
                if self._is_valid_sql(sql):
                    sql_statements.append(sql)

            print(f"✅ 从Slow Query Log读取到 {len(sql_statements)} 条SQL语句")

        except Exception as e:
            print(f"❌ 读取Slow Query Log文件失败: {str(e)}")

        return sql_statements

    def read_from_custom_log_file(self, log_file_path: str, sql_pattern: Optional[str] = None) -> List[str]:
        """
        从自定义日志文件读取SQL语句
        :param log_file_path: 日志文件路径
        :param sql_pattern: 自定义SQL匹配正则表达式，如果为None则使用默认模式
        :return: SQL语句列表
        """
        sql_statements = []

        if not os.path.exists(log_file_path):
            print(f"❌ 日志文件不存在: {log_file_path}")
            return sql_statements

        print(f"📖 正在读取自定义日志文件: {log_file_path}")

        try:
            with open(log_file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # 默认模式：匹配常见的SQL语句
            if sql_pattern is None:
                # 匹配 SELECT, INSERT, UPDATE, DELETE, CREATE, ALTER 等开头的SQL
                sql_pattern = r"(?i)(?:SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|TRUNCATE|MERGE|REPLACE)\s+[^;]+;?"

            matches = re.findall(sql_pattern, content, re.DOTALL | re.MULTILINE)

            for match in matches:
                sql = match.strip().rstrip(";")
                if self._is_valid_sql(sql):
                    sql_statements.append(sql)

            print(f"✅ 从自定义日志文件读取到 {len(sql_statements)} 条SQL语句")

        except Exception as e:
            print(f"❌ 读取自定义日志文件失败: {str(e)}")

        return sql_statements

    def read_from_performance_schema(self, limit: int = 1000) -> List[Dict[str, str]]:
        """
        从MySQL performance_schema.events_statements_history表读取SQL语句
        :param limit: 读取记录数限制
        :return: SQL语句列表（包含元数据）
        """
        sql_statements = []

        if not self.mysql_config:
            print("❌ MySQL配置未设置，无法从performance_schema读取")
            return sql_statements

        print(f"📖 正在从performance_schema读取SQL语句（限制: {limit}条）")

        try:
            conn = pymysql.connect(
                host=self.mysql_config.get("host", "localhost"),
                port=self.mysql_config.get("port", 13006),
                user=self.mysql_config.get("user", "root"),
                password=self.mysql_config.get("password", 1),
                database=self.mysql_config.get("database", "performance_schema"),
                charset="utf8mb4",
            )

            with conn.cursor() as cursor:
                # 查询最近的SQL语句
                sql_query = f"""
                SELECT 
                    sql_text,
                    db,
                    exec_count,
                    sum_timer_wait / 1000000000000 as exec_time_sec
                FROM performance_schema.events_statements_history_long
                WHERE sql_text IS NOT NULL 
                    AND sql_text NOT LIKE 'SHOW%'
                    AND sql_text NOT LIKE 'SELECT%performance_schema%'
                    AND sql_text NOT LIKE 'SELECT%information_schema%'
                ORDER BY timer_start DESC
                LIMIT {limit}
                """

                cursor.execute(sql_query)
                results = cursor.fetchall()

                for row in results:
                    sql_text, db, exec_count, exec_time = row
                    if sql_text and self._is_valid_sql(sql_text):
                        sql_statements.append(
                            {
                                "sql": sql_text,
                                "database": db or "",
                                "exec_count": exec_count or 0,
                                "exec_time": exec_time or 0,
                                "source": "performance_schema",
                            }
                        )

            conn.close()
            print(f"✅ 从performance_schema读取到 {len(sql_statements)} 条SQL语句")

        except Exception as e:
            print(f"❌ 从performance_schema读取失败: {str(e)}")

        return sql_statements

    def _is_valid_sql(self, sql: str) -> bool:
        """
        判断是否为有效的SQL语句
        :param sql: SQL语句
        :return: 是否为有效SQL
        """
        if not sql or len(sql.strip()) < 10:
            return False

        # 过滤掉系统查询
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
        ]

        for pattern in skip_patterns:
            if sql_upper.startswith(pattern):
                return False

        return True

    def get_sql_statements(self) -> List[str]:
        """
        获取所有收集的SQL语句
        :return: SQL语句列表
        """
        return self.sql_statements

    def clear_statements(self):
        """清空已收集的SQL语句"""
        self.sql_statements = []
