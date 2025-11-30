#!/bin/bash


SQL_FILE="init_sql.sql"        # SQL 文件路径

# 检查 SQL 文件是否存在
if [ ! -f "$SQL_FILE" ]; then
    echo "Error: SQL file $SQL_FILE not found."
    exit 1
fi

# 调用 Python 脚本 初始化mysql数据表
python3 ../common/initialize_mysql.py

# 调用 Python 脚本 初始化表关系
python3 ../common/initialize_neo4j.py