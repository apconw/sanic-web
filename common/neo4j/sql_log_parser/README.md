# SQL日志解析器使用说明

## 功能概述

SQL日志解析器用于从MySQL执行日志中自动提取表关系，并写入Neo4j图数据库。适用于无法直接访问业务代码的场景。

## 代码目录结构

```
common/neo4j/sql_log_parser/
├── __init__.py                          # 模块初始化文件
├── sql_log_reader.py                    # SQL日志读取器（General Log、Slow Query Log、Performance Schema）
├── binlog_reader.py                     # Binlog实时读取器（推荐）
├── sql_relationship_extractor.py        # SQL关系提取器（从SQL中提取表关系）
├── sql_log_to_neo4j.py                 # 主流程脚本（整合所有功能，提供完整流程）
├── README.md                            # 使用说明文档
└── 技术文章：从SQL执行日志自动构建数据库表关系图谱.md  # 技术文章
```

### 核心模块说明

| 文件 | 功能 | 主要类/函数 |
|------|------|------------|
| `sql_log_reader.py` | 从多种日志文件读取SQL | `SQLLogReader` |
| `binlog_reader.py` | 实时读取MySQL binlog | `BinlogReader` |
| `sql_relationship_extractor.py` | 从SQL语句中提取表关系 | `SQLRelationshipExtractor` |
| `sql_log_to_neo4j.py` | 完整流程编排 | `SQLLogToNeo4jPipeline` |

### 工作流程

```
┌─────────────────────┐
│  sql_log_reader.py  │  → 从日志文件读取SQL
│  binlog_reader.py   │  → 实时读取binlog
└──────────┬──────────┘
           │
           ▼
┌──────────────────────────────┐
│ sql_relationship_extractor.py│  → 解析SQL，提取表关系
└──────────┬───────────────────┘
           │
           ▼
┌─────────────────────┐
│ sql_log_to_neo4j.py │  → 写入Neo4j，导出JSON
└─────────────────────┘
```

## 支持的数据源

1. **MySQL Binlog（实时读取）** ⭐推荐
2. MySQL General Log 文件
3. MySQL Slow Query Log 文件
4. 自定义SQL日志文件
5. MySQL performance_schema.events_statements_history 表

## 快速开始

### 1. 安装依赖

```bash
# 基础依赖
pip install pymysql py2neo

# Binlog实时读取（推荐）
pip install pymysql-replication
```

### 2. 配置MySQL

#### 启用Binlog（推荐）

```sql
-- 检查binlog是否启用
SHOW VARIABLES LIKE 'log_bin';

-- 如果未启用，在my.cnf中配置
[mysqld]
log-bin=mysql-bin
binlog-format=ROW
server-id=1
```

#### 启用General Log（备选）

```sql
SET GLOBAL general_log = 'ON';
SET GLOBAL general_log_file = '/var/log/mysql/general.log';
```

#### 启用Slow Query Log（备选）

```sql
SET GLOBAL slow_query_log = 'ON';
SET GLOBAL slow_query_log_file = '/var/log/mysql/slow.log';
SET GLOBAL long_query_time = 0;  -- 记录所有查询
```

### 3. 运行脚本

```bash
python sql_log_to_neo4j.py
```

按照提示选择数据源并输入相应路径即可。

## 使用示例

### 从General Log提取

```python
from sql_log_to_neo4j import SQLLogToNeo4jPipeline

pipeline = SQLLogToNeo4jPipeline()
pipeline.run_from_general_log(
    log_file_path="/var/log/mysql/general.log",
    clear_existing=False,
    export_json=True
)
```

### 从performance_schema提取

```python
pipeline = SQLLogToNeo4jPipeline()
pipeline.run_from_performance_schema(
    limit=1000,
    clear_existing=False
)
```

### 从Binlog实时读取（推荐）

```python
pipeline = SQLLogToNeo4jPipeline()

# 实时读取，增量更新到Neo4j
pipeline.run_from_binlog_realtime(
    log_file=None,              # 从当前位置开始
    log_pos=None,               # 从当前位置开始
    stop_after_seconds=3600,    # 读取1小时后停止（None则持续读取）
    clear_existing=False,
    export_json=True,
    incremental_update=True     # 实时增量更新到Neo4j
)
```

## 输出文件

- `sql_log_relationships.json`: 提取的表关系JSON文件

## 注意事项

1. 确保Neo4j服务已启动
2. 确保有读取MySQL日志文件的权限
3. 大量SQL语句处理可能需要较长时间
4. 建议先在小规模数据上测试

