#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SQL日志到Neo4j自动化工具
功能：
1. 从MySQL执行日志中读取SQL语句
2. 解析SQL语句，提取表关系
3. 自动写入Neo4j图数据库
"""

import os
import sys
import json
from typing import List, Dict, Optional
from py2neo import Graph

# 支持作为独立脚本运行
try:
    from .sql_log_reader import SQLLogReader
    from .sql_relationship_extractor import SQLRelationshipExtractor
    from .binlog_reader import BinlogReader
except ImportError:
    from sql_log_reader import SQLLogReader
    from sql_relationship_extractor import SQLRelationshipExtractor
    from binlog_reader import BinlogReader

# ==================== 配置 ====================
# Neo4j 配置
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "neo4j123"

# MySQL 配置（用于从performance_schema读取）
MYSQL_CONFIG = {"host": "localhost", "port": 13006, "user": "root", "password": "1", "database": "performance_schema"}


class SQLLogToNeo4jPipeline:
    """SQL日志到Neo4j的完整流程"""

    def __init__(
        self,
        neo4j_uri: str = NEO4J_URI,
        neo4j_user: str = NEO4J_USER,
        neo4j_password: str = NEO4J_PASSWORD,
        mysql_config: Optional[Dict] = None,
    ):
        """
        初始化流程
        :param neo4j_uri: Neo4j连接URI
        :param neo4j_user: Neo4j用户名
        :param neo4j_password: Neo4j密码
        :param mysql_config: MySQL配置（可选）
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.mysql_config = mysql_config or MYSQL_CONFIG

        self.graph = None
        self.log_reader = SQLLogReader(self.mysql_config)
        self.binlog_reader = None
        self.extractor = SQLRelationshipExtractor()
        self.relationships = []
        self.tables = set()

    def connect_neo4j(self):
        """连接Neo4j数据库"""
        try:
            self.graph = Graph(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
            print("✅ Neo4j连接成功")
            return True
        except Exception as e:
            print(f"❌ Neo4j连接失败: {str(e)}")
            return False

    def read_sql_from_general_log(self, log_file_path: str):
        """从General Log文件读取SQL"""
        print("\n" + "=" * 60)
        print("步骤1: 从General Log读取SQL语句")
        print("=" * 60)

        sql_statements = self.log_reader.read_from_general_log_file(log_file_path)
        return sql_statements

    def read_sql_from_slow_query_log(self, log_file_path: str):
        """从Slow Query Log文件读取SQL"""
        print("\n" + "=" * 60)
        print("步骤1: 从Slow Query Log读取SQL语句")
        print("=" * 60)

        sql_statements = self.log_reader.read_from_slow_query_log_file(log_file_path)
        return sql_statements

    def read_sql_from_custom_log(self, log_file_path: str, sql_pattern: Optional[str] = None):
        """从自定义日志文件读取SQL"""
        print("\n" + "=" * 60)
        print("步骤1: 从自定义日志文件读取SQL语句")
        print("=" * 60)

        sql_statements = self.log_reader.read_from_custom_log_file(log_file_path, sql_pattern)
        return sql_statements

    def read_sql_from_performance_schema(self, limit: int = 1000):
        """从performance_schema读取SQL"""
        print("\n" + "=" * 60)
        print("步骤1: 从performance_schema读取SQL语句")
        print("=" * 60)

        sql_data = self.log_reader.read_from_performance_schema(limit)
        return sql_data

    def extract_relationships(self, sql_statements: List, source_name: str = "sql_log"):
        """
        从SQL语句中提取表关系
        :param sql_statements: SQL语句列表（可以是字符串列表或字典列表）
        :param source_name: 来源名称
        """
        print("\n" + "=" * 60)
        print("步骤2: 提取表关系")
        print("=" * 60)

        all_relationships = []

        for idx, sql_item in enumerate(sql_statements):
            if isinstance(sql_item, dict):
                sql = sql_item.get("sql", "")
                sql_id = sql_item.get("sql_id", f"{source_name}_{idx}")
                source = sql_item.get("source", source_name)
            else:
                sql = sql_item
                sql_id = f"{source_name}_{idx}"
                source = source_name

            if not sql:
                continue

            # 提取关系
            rels = self.extractor.extract_from_sql(sql, source=source, sql_id=sql_id)
            all_relationships.extend(rels)

            if (idx + 1) % 100 == 0:
                print(f"  已处理 {idx + 1}/{len(sql_statements)} 条SQL语句")

        # 去重
        self.relationships = self.extractor.deduplicate_relationships(all_relationships)
        self.tables = self.extractor.get_tables()

        print(f"\n📊 提取统计:")
        print(f"  - 处理SQL语句: {len(sql_statements)} 条")
        print(f"  - 发现表: {len(self.tables)} 个")
        print(f"  - 发现关系: {len(self.relationships)} 个")

        return True

    def export_relationships(self, output_dir: Optional[str] = None):
        """
        导出关系文件
        :param output_dir: 输出目录，如果为None则使用当前脚本目录
        """
        print("\n" + "=" * 60)
        print("步骤3: 导出关系文件")
        print("=" * 60)

        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(__file__))

        # 导出JSON格式
        json_output = os.path.join(output_dir, "sql_log_relationships.json")
        data = {
            "tables": list(self.tables),
            "relationships": self.relationships,
            "total_tables": len(self.tables),
            "total_relationships": len(self.relationships),
        }

        with open(json_output, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"✅ 关系已导出到: {json_output}")
        return True

    def create_neo4j_constraints(self):
        """创建Neo4j约束"""
        try:
            self.graph.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Table) REQUIRE t.name IS UNIQUE")
            print("✅ Neo4j约束创建完成")
            return True
        except Exception as e:
            print(f"❌ 创建约束失败: {str(e)}")
            return False

    def create_table_nodes(self):
        """创建表节点"""
        print("\n📦 正在创建表节点...")

        for table_name in self.tables:
            try:
                self.graph.run(
                    "MERGE (t:Table {name: $name}) " "SET t.label = $label, " "    t.source = 'sql_log'",
                    name=table_name,
                    label=table_name,
                )
                print(f"  ✅ 创建表节点: {table_name}")
            except Exception as e:
                print(f"  ❌ 创建表节点失败 {table_name}: {str(e)}")

        print(f"✅ 共创建 {len(self.tables)} 个表节点")
        return True

    def create_table_relationships(self):
        """创建表关系"""
        print("\n🔗 正在创建表关系...")

        success_count = 0
        for rel in self.relationships:
            try:
                cypher = """
                MATCH (from_table:Table {name: $from_table})
                MATCH (to_table:Table {name: $to_table})
                MERGE (from_table)-[r:REFERENCES {
                    description: $description,
                    field_relation: $field_relation,
                    join_type: $join_type,
                    relation_type: $relation_type,
                    source: $source,
                    sql_id: $sql_id
                }]->(to_table)
                """

                self.graph.run(
                    cypher,
                    from_table=rel["from_table"],
                    to_table=rel["to_table"],
                    description=rel.get("description", ""),
                    field_relation=rel.get("field_relation", ""),
                    join_type=rel.get("join_type", "UNKNOWN"),
                    relation_type=rel.get("relation_type", "UNKNOWN"),
                    source=rel.get("source", ""),
                    sql_id=rel.get("sql_id", ""),
                )

                print(f"  ✅ {rel['from_table']} -> {rel['to_table']} ({rel.get('field_relation', '')})")
                success_count += 1

            except Exception as e:
                print(f"  ❌ 创建关系失败: {str(e)}")

        print(f"✅ 共创建 {success_count} 个表关系")
        return True

    def write_to_neo4j(self, clear_existing: bool = False):
        """写入Neo4j数据库"""
        print("\n" + "=" * 60)
        print("步骤4: 写入Neo4j图数据库")
        print("=" * 60)

        # 连接Neo4j
        if not self.connect_neo4j():
            return False

        # 是否清空现有数据
        if clear_existing:
            print("🗑️  清空现有Neo4j数据...")
            self.graph.delete_all()

        # 创建约束
        self.create_neo4j_constraints()

        # 创建表节点
        self.create_table_nodes()

        # 创建表关系
        self.create_table_relationships()

        return True

    def run_from_general_log(self, log_file_path: str, clear_existing: bool = False, export_json: bool = True):
        """
        从General Log运行完整流程
        :param log_file_path: General Log文件路径
        :param clear_existing: 是否清空现有Neo4j数据
        :param export_json: 是否导出JSON文件
        """
        print("\n" + "=" * 60)
        print("🚀 SQL日志到Neo4j自动化流程 (General Log)")
        print("=" * 60)
        print(f"日志文件: {log_file_path}")
        print(f"Neo4j URI: {self.neo4j_uri}")
        print("=" * 60)

        try:
            # 步骤1: 读取SQL
            sql_statements = self.read_sql_from_general_log(log_file_path)
            if not sql_statements:
                print("❌ 未读取到任何SQL语句")
                return False

            # 步骤2: 提取关系
            self.extract_relationships(sql_statements, source_name="general_log")

            # 步骤3: 导出JSON（可选）
            if export_json:
                self.export_relationships()

            # 步骤4: 写入Neo4j
            self.write_to_neo4j(clear_existing)

            # 完成
            self._print_summary()
            return True

        except Exception as e:
            print(f"\n❌ 流程执行失败: {str(e)}")
            import traceback

            traceback.print_exc()
            return False

    def run_from_slow_query_log(self, log_file_path: str, clear_existing: bool = False, export_json: bool = True):
        """从Slow Query Log运行完整流程"""
        print("\n" + "=" * 60)
        print("🚀 SQL日志到Neo4j自动化流程 (Slow Query Log)")
        print("=" * 60)

        try:
            sql_statements = self.read_sql_from_slow_query_log(log_file_path)
            if not sql_statements:
                return False

            self.extract_relationships(sql_statements, source_name="slow_query_log")

            if export_json:
                self.export_relationships()

            self.write_to_neo4j(clear_existing)
            self._print_summary()
            return True

        except Exception as e:
            print(f"\n❌ 流程执行失败: {str(e)}")
            import traceback

            traceback.print_exc()
            return False

    def run_from_performance_schema(self, limit: int = 1000, clear_existing: bool = False, export_json: bool = True):
        """从performance_schema运行完整流程"""
        print("\n" + "=" * 60)
        print("🚀 SQL日志到Neo4j自动化流程 (Performance Schema)")
        print("=" * 60)

        try:
            sql_data = self.read_sql_from_performance_schema(limit)
            if not sql_data:
                return False

            self.extract_relationships(sql_data, source_name="performance_schema")

            if export_json:
                self.export_relationships()

            self.write_to_neo4j(clear_existing)
            self._print_summary()
            return True

        except Exception as e:
            print(f"\n❌ 流程执行失败: {str(e)}")
            import traceback

            traceback.print_exc()
            return False

    def _print_summary(self):
        """打印统计信息"""
        print("\n" + "=" * 60)
        print("🎉 流程执行完成！")
        print("=" * 60)
        print(f"📊 统计信息:")
        print(f"  - 发现表: {len(self.tables)} 个")
        print(f"  - 发现关系: {len(self.relationships)} 个")
        print("=" * 60)


def main():
    """主函数"""
    print("=" * 60)
    print("SQL日志到Neo4j自动化工具")
    print("=" * 60)

    # 选择数据源
    print("\n请选择数据源:")
    print("1. MySQL General Log 文件")
    print("2. MySQL Slow Query Log 文件")
    print("3. 自定义日志文件")
    print("4. MySQL performance_schema")
    print("5. MySQL Binlog (实时读取) ⭐推荐")
    
    choice = input("\n请输入选项 (1-5): ").strip()

    pipeline = SQLLogToNeo4jPipeline()

    if choice == "1":
        log_file = input("请输入General Log文件路径: ").strip()
        if not os.path.exists(log_file):
            print(f"❌ 文件不存在: {log_file}")
            return
        clear = input("是否清空Neo4j现有数据? (y/N): ").strip().lower() in ["y", "yes"]
        pipeline.run_from_general_log(log_file, clear_existing=clear)

    elif choice == "2":
        log_file = input("请输入Slow Query Log文件路径: ").strip()
        if not os.path.exists(log_file):
            print(f"❌ 文件不存在: {log_file}")
            return
        clear = input("是否清空Neo4j现有数据? (y/N): ").strip().lower() in ["y", "yes"]
        pipeline.run_from_slow_query_log(log_file, clear_existing=clear)

    elif choice == "3":
        log_file = input("请输入自定义日志文件路径: ").strip()
        if not os.path.exists(log_file):
            print(f"❌ 文件不存在: {log_file}")
            return
        sql_pattern = input("请输入SQL匹配正则表达式（可选，直接回车使用默认）: ").strip() or None
        clear = input("是否清空Neo4j现有数据? (y/N): ").strip().lower() in ["y", "yes"]

        sql_statements = pipeline.read_sql_from_custom_log(log_file, sql_pattern)
        if sql_statements:
            pipeline.extract_relationships(sql_statements, source_name="custom_log")
            pipeline.export_relationships()
            pipeline.write_to_neo4j(clear)
            pipeline._print_summary()

    elif choice == "4":
        limit = input("请输入读取记录数限制 (默认1000): ").strip()
        limit = int(limit) if limit.isdigit() else 1000
        clear = input("是否清空Neo4j现有数据? (y/N): ").strip().lower() in ["y", "yes"]
        pipeline.run_from_performance_schema(limit=limit, clear_existing=clear)
        
    elif choice == "5":
        print("\n📖 Binlog实时读取配置:")
        log_file = input("起始binlog文件名（可选，直接回车从当前位置开始）: ").strip() or None
        log_pos_str = input("起始binlog位置（可选，直接回车从当前位置开始）: ").strip()
        log_pos = int(log_pos_str) if log_pos_str.isdigit() else None
        
        duration_str = input("读取时长（秒，直接回车则持续读取）: ").strip()
        duration = int(duration_str) if duration_str.isdigit() else None
        
        incremental = input("是否增量更新（实时写入Neo4j）? (Y/n): ").strip().lower() not in ["n", "no"]
        clear = input("是否清空Neo4j现有数据? (y/N): ").strip().lower() in ["y", "yes"]
        
        pipeline.run_from_binlog_realtime(
            log_file=log_file,
            log_pos=log_pos,
            stop_after_seconds=duration,
            clear_existing=clear,
            incremental_update=incremental
        )
        
    else:
        print("❌ 无效选项")
        return

    print("\n✅ 所有操作已成功完成！")
    print("\n💡 提示:")
    print("  - 关系JSON文件: common/neo4j/sql_log_parser/sql_log_relationships.json")
    print("  - 可以在Neo4j Browser中查看图谱")


if __name__ == "__main__":
    main()
