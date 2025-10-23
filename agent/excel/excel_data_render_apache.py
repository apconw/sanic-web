import json
import logging
import traceback
from decimal import Decimal

from agent.text2sql.state.agent_state import AgentState, ExecutionResult
import sqlglot
from sqlglot import parse
from datetime import datetime, date
import pandas as pd

from services.db_qadata_process import process

"""
AntV MCP默认没有提供表格组件 这里使用
Apache EChart数据渲染节点来支撑表格的渲染
前期使用硬编码方式后面使用MCP方式
"""

logger = logging.getLogger(__name__)


def excel_data_render_apache(state: AgentState) -> dict:
    """
    渲染Apache ECharts数据（支持表格数据结构）

    :param state: Agent状态对象
    :return: 包含图表数据的字典
    """
    table_schema_info = state.get("db_info", {})
    # sql_reasoning = state.get("sql_reasoning", "")
    generated_sql = state.get("generated_sql", "")
    data_result: ExecutionResult = state.get("execution_result")

    # 构建基础表格数据结构
    table_data = {"llm": {"type": "response_table"}, "data": {"column": [], "result": []}}

    # 获取生成的SQL中的表名
    generated_table_names = extract_table_names_sqlglot(generated_sql)
    if not generated_table_names:
        logger.info("未从SQL中提取到表名")
        return table_data

    # 检查是否为 SELECT * 查询
    is_select_all = check_if_select_all(generated_sql)

    # 获取列标题（中文注释或别名）
    if is_select_all:
        # 对于 SELECT * 查询，从表 schema 中获取所有列信息
        column_comments = get_all_column_comments_for_tables(generated_table_names, table_schema_info)
        # 获取实际的列名
        actual_columns = get_actual_columns_for_select_all(generated_table_names, table_schema_info)
    else:
        # 正常解析 SQL 获取列信息
        column_comments = extract_select_columns_with_comments(generated_sql, table_schema_info)
        # 获取实际的列名
        actual_columns = extract_actual_column_names(generated_sql)

    table_data["data"]["column"] = column_comments

    # 填充 result 数据
    if data_result and data_result.data:
        for row in data_result.data:
            # 确保 row 是 dict 类型
            if isinstance(row, dict):
                # 按 actual_columns 顺序取出值，对应 column_comments 的顺序
                row_values = [convert_value(row.get(col, None)) for col in actual_columns]
                # 修改此处：将 row_values 和 column_comments 组合成对象
                table_data["data"]["result"].append(dict(zip(column_comments, row_values)))
            else:
                # 兼容非 dict 格式（如元组或列表）
                logger.info(f"数据行格式异常，跳过: {row}")

    processed_data = process(json.dumps(table_data, ensure_ascii=False))
    state["apache_chart_data"] = processed_data

    return state


def convert_value(v):
    if isinstance(v, Decimal):
        return float(v)  # 或者 str(v)
    elif isinstance(v, (datetime, pd.Timestamp)):
        return v.strftime("%Y-%m-%d %H:%M:%S")
    elif isinstance(v, date):
        return v.strftime("%Y-%m-%d")
    else:
        return v


def check_if_select_all(sql: str) -> bool:
    """
    检查 SQL 是否为 SELECT * 查询

    :param sql: SQL 语句
    :return: 是否为 SELECT * 查询
    """
    try:
        expressions = parse(sql)
        for expression in expressions:
            if expression:
                selects = expression.find_all(sqlglot.exp.Select)
                for select in selects:
                    projections = select.expressions if hasattr(select, "expressions") else []
                    for proj in projections:
                        # 检查是否有星号表达式
                        if isinstance(proj, sqlglot.exp.Star):
                            return True
        return False
    except Exception as e:
        logger.info(f"检查 SELECT * 失败: {e}")
        return False


def find_table_in_list(table_name, schema_inspector_list):
    """
    在 list[dict] 类型的 schema 信息中查找指定表的信息

    :param table_name: 表名
    :param schema_inspector_list: schema 信息列表
    :return: 表信息字典或空字典
    """
    for table_info in schema_inspector_list:
        if isinstance(table_info, dict) and table_info.get("table_name") == table_name:
            return table_info
    return {}


def get_all_column_comments_for_tables(table_names: list, schema_inspector: list) -> list:
    """
    为指定表获取所有列的注释信息（用于 SELECT * 查询）

    :param table_names: 表名列表
    :param schema_inspector: schema 信息 (list[dict] 类型)
    :return: 所有列的注释列表
    """
    comments = []
    try:
        for table_name in table_names:
            table_info = find_table_in_list(table_name, schema_inspector)
            if not table_info:
                table_info = schema_inspector[0] if schema_inspector and isinstance(schema_inspector, list) else {}

            columns = table_info.get("columns", {}) if table_info else {}
            for col_name, col_info in columns.items():
                if isinstance(col_info, dict):
                    comment = col_info.get("comment")
                    if comment and comment != "None":
                        comments.append(comment)
                    else:
                        comments.append(col_name)  # 使用列名作为默认值
                else:
                    comments.append(col_name)
    except Exception as e:
        logger.error(f"获取所有列注释失败: {e}")
        return []

    return comments


def get_actual_columns_for_select_all(table_names: list, schema_inspector: list) -> list:
    """
    为 SELECT * 查询获取实际的列名列表

    :param table_names: 表名列表
    :param schema_inspector: schema 信息 (list[dict] 类型)
    :return: 所有列的列名列表
    """
    columns = []
    try:
        for table_name in table_names:
            table_info = find_table_in_list(table_name, schema_inspector)
            if not table_info:
                table_info = schema_inspector[0] if schema_inspector and isinstance(schema_inspector, list) else {}

            table_columns = table_info.get("columns", {}) if table_info else {}
            columns.extend(list(table_columns.keys()))
    except Exception as e:
        logger.info(f"获取 SELECT * 实际列名失败: {e}")
        return []

    return columns


def extract_table_names_sqlglot(sql: str) -> list:
    """
    使用 sqlglot 提取 SQL 中的所有表名（支持复杂语法、多表、子查询、CTE 等）

    :param sql: SQL 语句
    :return: 表名列表（去重）
    """
    try:
        expression = parse(sql)[0]
        tables = set()
        for table in expression.find_all(sqlglot.exp.Table):
            # 取表名（去掉 schema）
            tables.add(table.name)
        return list(tables)
    except Exception as e:
        logger.info(f"SQL 解析错误: {e}")
        return []


def get_column_comments(schema_inspector: list, table_name: str) -> list:
    """
    从 schema_inspector 中提取指定表的所有字段的 comment，放入 list 中。

    :param schema_inspector: 包含数据库 schema 信息的列表 (list[dict])
    :param table_name: 要提取 comment 的表名
    :return: 包含所有字段 comment 的 list
    """
    try:
        table_info = find_table_in_list(table_name, schema_inspector)
        if not table_info:
            table_info = schema_inspector[0] if schema_inspector and isinstance(schema_inspector, list) else {}

        if not table_info:
            logger.info(f"未找到表 {table_name} 的信息")
            return []

        columns = table_info.get("columns", {})
        if not columns:
            logger.info(f"表 {table_name} 没有列信息")
            return []

        comments = []
        for col_name, col_info in columns.items():
            # 确保col_info是字典类型
            if isinstance(col_info, dict):
                comment = col_info.get("comment")
                # 如果 comment 为 'None' 字符串或 None，用空字符串代替
                if comment and comment != "None":
                    comments.append(comment)
                else:
                    comments.append("")  # 使用空字符串作为默认值
            else:
                comments.append("")  # 如果列信息格式不正确，使用空字符串
        return comments
    except Exception as e:
        traceback.print_exception(e)
        logger.info(f"处理表 {table_name} 的列注释时出错: {e}")
        return []


def extract_actual_column_names(sql: str) -> list:
    """
    使用 sqlglot 提取 SQL 中 SELECT 子句的实际列名

    :param sql: SQL 语句
    :return: 实际列名列表
    """
    try:
        expressions = parse(sql)
        columns = []

        for expression in expressions:
            if expression:
                # 查找所有的 SELECT 表达式
                select_expressions = expression.find_all(sqlglot.exp.Select)

                for select_expr in select_expressions:
                    # 获取 projections (SELECT 后面的列)
                    projections = select_expr.expressions if hasattr(select_expr, "expressions") else []

                    for proj in projections:
                        # 处理普通列名
                        if isinstance(proj, sqlglot.exp.Column):
                            # 构造完整的列名（如果有表前缀）
                            if hasattr(proj, "table") and proj.table:
                                columns.append(f"{proj.table}.{proj.name}")
                            else:
                                columns.append(proj.name)
                        # 处理聚合函数等表达式（使用别名）
                        elif isinstance(proj, sqlglot.exp.Alias):
                            columns.append(proj.alias)
                        elif isinstance(
                            proj,
                            (
                                sqlglot.exp.Sum,
                                sqlglot.exp.Count,
                                sqlglot.exp.Avg,
                                sqlglot.exp.Max,
                                sqlglot.exp.Min,
                                sqlglot.exp.Star,
                            ),
                        ):
                            # 对于聚合函数等，如果没有别名，使用生成的名称
                            alias = getattr(proj, "alias", None)
                            name = getattr(proj, "name", None)
                            output_name = getattr(proj, "output_name", None)
                            columns.append(alias or name or output_name or "unknown")
                        # 处理其他情况
                        elif hasattr(proj, "alias") and proj.alias:
                            columns.append(proj.alias)
                        elif hasattr(proj, "name"):
                            columns.append(proj.name)
                        elif hasattr(proj, "output_name"):
                            columns.append(proj.output_name)

        return columns
    except Exception as e:
        logger.info(f"SQL 实际列名解析错误: {e}")
        return []


def extract_column_names_sqlglot(sql: str) -> list:
    """
    使用 sqlglot 提取 SQL 中 SELECT 子句的所有列名

    :param sql: SQL 语句
    :return: 列名列表
    """
    try:
        expressions = parse(sql)
        columns = []

        for expression in expressions:
            if expression:
                # 查找所有的 SELECT 表达式
                select_expressions = expression.find_all(sqlglot.exp.Select)

                for select_expr in select_expressions:
                    # 获取 projections (SELECT 后面的列)
                    projections = select_expr.expressions if hasattr(select_expr, "expressions") else []

                    for proj in projections:
                        # 处理别名情况
                        if isinstance(proj, sqlglot.exp.Alias):
                            columns.append(proj.alias)
                        # 处理普通列名
                        elif hasattr(proj, "name"):
                            columns.append(proj.name)
                        # 处理其他情况
                        elif hasattr(proj, "output_name"):
                            columns.append(proj.output_name)
                        # 处理字符串形式的表达式
                        elif isinstance(proj, sqlglot.exp.Column):
                            columns.append(proj.name)
                        # 处理星号
                        elif isinstance(proj, sqlglot.exp.Star):
                            columns.append("*")

        # 去重但保持顺序
        unique_columns = []
        for col in columns:
            if col not in unique_columns:
                unique_columns.append(col)

        return unique_columns
    except Exception as e:
        logger.info(f"SQL 列名解析错误: {e}")
        return []


def extract_select_columns_with_comments(sql: str, schema_inspector: list) -> list:
    """
    使用 sqlglot 提取 SQL 中 SELECT 子句的列名，并尝试获取对应的中文注释
    对于带有别名的统计类列（如聚合函数），保留别名不转换为中文

    :param sql: SQL 语句
    :param schema_inspector: 包含数据库 schema 信息的列表 (list[dict])
    :return: 列标题列表（优先使用中文注释，其次使用别名或列名）
    """
    try:
        expressions = parse(sql)
        column_info_list = []

        # 提取表别名映射
        table_alias_mapping = extract_table_alias_mapping(sql)

        for expression in expressions:
            if expression:
                # 查找 SELECT 表达式
                selects = expression.find_all(sqlglot.exp.Select)

                for select in selects:
                    # 获取 projections (SELECT 后面的列)
                    projections = select.expressions if hasattr(select, "expressions") else []

                    for proj in projections:
                        column_info = {"name": None, "alias": None, "is_aggregate": False, "table": None}

                        # 处理带别名的列
                        if isinstance(proj, sqlglot.exp.Alias):
                            column_info["alias"] = proj.alias
                            column_info["name"] = proj.this.name if hasattr(proj.this, "name") else None
                            # 检查是否为聚合函数
                            column_info["is_aggregate"] = isinstance(
                                proj.this,
                                (sqlglot.exp.Sum, sqlglot.exp.Count, sqlglot.exp.Avg, sqlglot.exp.Max, sqlglot.exp.Min),
                            )
                            # 获取表名（如果有）
                            if hasattr(proj.this, "table"):
                                table_ref = proj.this.table
                                # 使用别名映射查找真实表名
                                column_info["table"] = table_alias_mapping.get(table_ref, table_ref)
                        # 处理普通列
                        elif isinstance(proj, sqlglot.exp.Column):
                            column_info["name"] = proj.name
                            if hasattr(proj, "table"):
                                table_ref = proj.table
                                # 使用别名映射查找真实表名
                                column_info["table"] = table_alias_mapping.get(table_ref, table_ref)
                        # 处理聚合函数等表达式
                        elif isinstance(
                            proj,
                            (sqlglot.exp.Sum, sqlglot.exp.Count, sqlglot.exp.Avg, sqlglot.exp.Max, sqlglot.exp.Min),
                        ):
                            column_info["is_aggregate"] = True
                            column_info["alias"] = getattr(proj, "alias", None) or getattr(proj, "name", None)
                        # 处理星号（SELECT *）
                        elif isinstance(proj, sqlglot.exp.Star):
                            # 对于星号，我们稍后会特殊处理
                            column_info["name"] = "*"
                            column_info["alias"] = "*"

                        # 获取列的标准名称
                        if not column_info["name"] and hasattr(proj, "name"):
                            column_info["name"] = proj.name

                        column_info_list.append(column_info)

        # 获取表名用于查找列注释
        table_names = extract_table_names_sqlglot(sql)

        # 生成最终的列标题
        result_columns = []
        for col_info in column_info_list:
            # 处理 SELECT * 的情况
            if col_info["name"] == "*":
                # 这种情况已经在上层函数中特殊处理了，这里只是兜底
                continue

            # 对于聚合函数或带别名的列，优先使用别名
            if col_info["is_aggregate"] or col_info["alias"]:
                result_columns.append(col_info["alias"] or col_info["name"])
            # 对于普通列，尝试获取中文注释
            elif col_info["name"]:
                # 优先使用列中指定的表名查找注释
                comment_found = False

                # 如果列信息中指定了表名
                if col_info["table"]:
                    # 首先尝试使用解析出的真实表名
                    real_table_name = col_info["table"]
                    if real_table_name in table_names:
                        table_info = find_table_in_list(real_table_name, schema_inspector)
                        if not table_info:
                            table_info = (
                                schema_inspector[0] if schema_inspector and isinstance(schema_inspector, list) else {}
                            )

                        columns = table_info.get("columns", {}) if table_info else {}
                        col_schema = columns.get(col_info["name"], {})

                        if isinstance(col_schema, dict):
                            comment = col_schema.get("comment")
                            if comment and comment != "None":
                                result_columns.append(comment)
                                comment_found = True

                    # 如果上面没找到，再遍历所有表查找
                    if not comment_found:
                        for table_name in table_names:
                            table_info = find_table_in_list(table_name, schema_inspector)
                            if not table_info:
                                table_info = (
                                    schema_inspector[0]
                                    if schema_inspector and isinstance(schema_inspector, list)
                                    else {}
                                )

                            columns = table_info.get("columns", {}) if table_info else {}
                            col_schema = columns.get(col_info["name"], {})

                            if isinstance(col_schema, dict):
                                comment = col_schema.get("comment")
                                if comment and comment != "None":
                                    result_columns.append(comment)
                                    comment_found = True
                                    break

                # 如果没找到注释，则使用列名
                if not comment_found:
                    result_columns.append(col_info["name"])
            else:
                # 兜底处理
                result_columns.append(col_info["alias"] or col_info["name"] or "未知列")

        return result_columns
    except Exception as e:
        logger.info(f"SQL 列名及注释解析错误: {e}")
        # 出错时回退到原来的简单实现
        return extract_column_names_sqlglot(sql)


def extract_table_alias_mapping(sql: str) -> dict:
    """
    提取SQL中的表别名映射关系

    :param sql: SQL语句
    :return: 别名到真实表名的映射字典
    """
    alias_mapping = {}
    try:
        expressions = parse(sql)

        # 方法1: 直接从表表达式中提取
        for expression in expressions:
            if expression:
                for table_exp in expression.find_all(sqlglot.exp.Table):
                    if hasattr(table_exp, "alias") and table_exp.alias:
                        alias_mapping[table_exp.alias] = table_exp.name

        # 方法2: 从FROM和JOIN子句中提取
        for expression in expressions:
            if expression:
                # 处理 FROM 子句
                for from_exp in expression.find_all(sqlglot.exp.From):
                    if hasattr(from_exp, "expressions"):
                        for expr in from_exp.expressions:
                            if isinstance(expr, sqlglot.exp.Alias) and hasattr(expr.this, "name"):
                                alias_mapping[expr.alias] = expr.this.name
                            elif hasattr(expr, "name") and hasattr(expr, "alias"):
                                alias_mapping[expr.alias] = expr.name

                # 处理 JOIN 子句
                for join_exp in expression.find_all(sqlglot.exp.Join):
                    if hasattr(join_exp, "this"):
                        join_table = join_exp.this
                        if isinstance(join_table, sqlglot.exp.Alias) and hasattr(join_table.this, "name"):
                            alias_mapping[join_table.alias] = join_table.this.name
                        elif hasattr(join_table, "name") and hasattr(join_table, "alias"):
                            alias_mapping[join_table.alias] = join_table.name
    except Exception as e:
        logger.info(f"提取表别名映射失败: {e}")

    logger.info(f"表别名映射: {alias_mapping}")
    return alias_mapping
