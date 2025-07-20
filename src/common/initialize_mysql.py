import pymysql
import os

from pymysql import MySQLError

"""
Mysql 初始化脚本工具类
"""

# 配置信息
MYSQL_ROOT_PASSWORD = "1"  # MySQL root 用户的密码
SQL_FILE = "../docker/init_sql.sql"  # SQL 文件路径
HOST = "localhost"  # MySQL 服务器地址
PORT = 13006  # MySQL 服务器端口


def check_sql_file(file_path):
    """
    检查 SQL 文件是否存在
    :param file_path:
    :return:
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Error: SQL file {file_path} not found.")


def execute_sql_file(file_path):
    """
    执行 SQL 文件
    :param file_path:
    :return:
    """
    try:
        # 创建数据库连接
        connection = pymysql.connect(
            host=HOST,
            user="root",
            password=MYSQL_ROOT_PASSWORD,
            port=PORT,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
        )
        with connection.cursor() as cursor:
            print(f"Initializing MySQL with {file_path} on port {PORT}...")

            # 读取 SQL 文件
            with open(file_path, "r", encoding="utf-8") as file:
                sql_script = file.read()

            # 分割 SQL 命令并执行
            commands = sql_script.split(";")
            for command in commands:
                if command.strip():  # 忽略空命令
                    cursor.execute(command)

            # 提交事务
            connection.commit()
            print("MySQL initialization completed successfully.")
    except pymysql.MySQLError as e:
        print(f"Error: {e}")
    finally:
        if connection.open:
            connection.close()


def execute_user_qa_record_sql():
    """
    使用 pymysql 连接数据库并执行 SQL 语句。
    初始化特殊结构json数据
    """

    sql_insert_data = """
    INSERT INTO chat_db.t_user_qa_record (user_id,uuid,conversation_id,message_id,task_id,chat_id,question,to2_answer,
    to4_answer,qa_type,file_key,create_time) VALUES(%s, %s, %s,%s, %s, %s, %s, %s, %s, %s,%s,%s)
    """
    # 紧凑格式数据，去掉换行
    data_to_insert = [
        (
            1,
            "82766aca-146d-44ac-9280-fc829991848f",
            "ed0ea22a-4403-4e9d-8947-92899b8f1b73",
            "eafb17a3-0fa3-4f99-b9db-196d4de12df3",
            "09e16efa-dedf-421a-b944-93537a8ff9cf",
            "3b542378-c492-4063-8c25-2e20e2d9428d",
            "统计案件数据按分局分组饼图",
            r'{"data": {"messageType": "continue", "content": "## 数据趋势概述  \n不同分局事件数量存在明显差异  \n\n**关键发现**  \n- '
            r'徐汇分局事件数最多，达10起  \n- 浦东分局次之（7起），朝阳、天河、海淀分局并列第三（各4起）  \n- 武侯、徐汇分局呈现极端值特征  \n\n**注意**  \n数据反映的是静态总量分布，未包含时间维度变化信息"}, "dataType": "t02"}',
            r'{"data": {"chart_type": "饼图", "template_code": "temp02", "data": [{"name": "浦东分局", "value": "7", '
            r'"percent": false}, {"name": "朝阳分局", "value": "4", "percent": false}, {"name": "天河分局", "value": "4", "percent": false}, {"name": "南山分局", "value": "3", "percent": false}, {"name": "武侯分局", "value": "1", "percent": false}, {"name": "锦江分局", "value": "2", "percent": false}, {"name": "徐汇分局", "value": "10", "percent": false}, {"name": "海淀分局", "value": "4", "percent": false}, {"name": "白云分局", "value": "3", "percent": false}, {"name": "福田分局", "value": "2", "percent": false}], "note": "数据来源: xxx数据库，以上数据仅供参考，具体情况可能会根据xx进一步调查和统计而有所变化"}, "dataType": "t04"}',
            "DATABASE_QA",
            "",
            "2025-07-09 14:35:42",
        ),
        (
            1,
            "db2d6f8c-f990-43a9-ba21-79dfcbb43c17",
            "38ad9103-d37b-4652-a9bf-cfebcfe3c9d6",
            "f3536814-65f6-4848-83a8-26429f5c71f9",
            "d4b18b90-8085-4d42-8fd5-b57f6b61b7a4",
            "3b542378-c492-4063-8c25-2e20e2d9428d",
            "统计案件数据按分局分组柱状图",
            r'{"data": {"messageType": "continue", "content": "## 分局案件数量分布趋势  '
            r'\n\n从数据来看，各分局的案件总数呈现出明显的区域差异。**徐汇分局以10起案件位居首位**，其次是**浦东分局（7起）和朝阳、天河、海淀分局（均为4起）**，其他分局案件数均低于3起。  \n\n**关键发现**  \n- 徐汇分局案件数量最多，是最低值的10倍  \n- 三个分局并列第二梯队，案件数为4起  \n- 超过半数分局案件数≤2起  \n\n**注意**  \n该数据未体现时间维度变化，仅反映静态总量分布情况，建议补充时间序列数据以分析动态趋势。"}, "dataType": "t02"}',
            r'{"data": {"chart_type": "柱状图", "template_code": "temp03", "data": [["product", "总数"], ["浦东分局", "7"], '
            r'["朝阳分局", "4"], ["天河分局", "4"], ["南山分局", "3"], ["武侯分局", "1"], ["锦江分局", "2"], ["徐汇分局", "10"], ["海淀分局", "4"], ["白云分局", "3"], ["福田分局", "2"]], "note": "数据来源: xxx数据库，以上数据仅供参考，具体情况可能会根据xx进一步调查和统计而有所变化"}, "dataType": "t04"}',
            "DATABASE_QA",
            "",
            "2025-07-09 14:36:41",
        ),
    ]

    # 创建数据库连接
    connection = pymysql.connect(
        host=HOST,
        user="root",
        password=MYSQL_ROOT_PASSWORD,
        port=PORT,
        db="chat_db",
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )

    try:
        with connection.cursor() as cursor:
            # 使用 executemany 方法插入多条数据
            cursor.executemany(sql_insert_data, data_to_insert)
        # 提交事务
        connection.commit()
    except MySQLError as e:
        print(f"Error executing query: {e}")
        # 如果出现错误，则回滚事务
        if connection.open:
            connection.rollback()
    finally:
        # 关闭数据库连接
        connection.close()


if __name__ == "__main__":
    check_sql_file(SQL_FILE)
    execute_sql_file(SQL_FILE)
    execute_user_qa_record_sql()
