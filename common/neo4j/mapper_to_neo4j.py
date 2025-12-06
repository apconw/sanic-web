#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MyBatis Mapper 到 Neo4j 自动化工具
功能：
1. 扫描Java Spring Boot项目中的MyBatis mapper.xml文件
2. 解析SQL语句，提取表关系
3. 自动写入Neo4j图数据库
"""

import os
import sys

from py2neo import Graph

from mybatis_mapper_parser import MyBatisMapperParser

# ==================== 配置 ====================
# Neo4j 配置
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "neo4j123"


class MapperToNeo4jPipeline:
    """MyBatis Mapper 到 Neo4j 的完整流程"""

    def __init__(
        self,
        java_project_path: str,
        neo4j_uri: str = NEO4J_URI,
        neo4j_user: str = NEO4J_USER,
        neo4j_password: str = NEO4J_PASSWORD,
    ):
        """
        初始化流程
        :param java_project_path: Java项目路径
        :param neo4j_uri: Neo4j连接URI
        :param neo4j_user: Neo4j用户名
        :param neo4j_password: Neo4j密码
        """
        self.java_project_path = java_project_path
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.parser = None
        self.graph = None
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

    def parse_mappers(self):
        """解析Mapper文件"""
        print("\n" + "=" * 60)
        print("步骤1: 解析MyBatis Mapper文件")
        print("=" * 60)

        self.parser = MyBatisMapperParser(self.java_project_path)

        # 扫描mapper文件
        self.parser.scan_mapper_files()

        if not self.parser.mapper_files:
            print("❌ 未找到任何mapper.xml文件")
            return False

        # 解析所有mapper
        self.relationships = self.parser.parse_all_mappers()
        self.tables = self.parser.tables

        return True

    def export_relationships(self):
        """导出关系文件"""
        print("\n" + "=" * 60)
        print("步骤2: 导出关系文件")
        print("=" * 60)

        output_dir = os.path.dirname(os.path.abspath(__file__))

        # 导出JSON格式
        json_output = os.path.join(output_dir, "mapper_relationships.json")
        self.parser.export_to_json(json_output)

        # 导出Python格式
        python_output = os.path.join(output_dir, "generated_relationships.py")
        self.parser.export_to_python_format(python_output)

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
                # 创建或更新表节点
                self.graph.run(
                    "MERGE (t:Table {name: $name}) " "SET t.label = $label, " "    t.source = 'mybatis_mapper'",
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
                # 创建关系
                cypher = """
                MATCH (from_table:Table {name: $from_table})
                MATCH (to_table:Table {name: $to_table})
                MERGE (from_table)-[r:REFERENCES {
                    description: $description,
                    field_relation: $field_relation,
                    join_type: $join_type,
                    source_file: $source_file,
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
                    source_file=rel.get("source_file", ""),
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
        print("步骤3: 写入Neo4j图数据库")
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

    def run(self, clear_existing: bool = False):
        """运行完整流程"""
        print("\n" + "=" * 60)
        print("🚀 MyBatis Mapper 到 Neo4j 自动化流程")
        print("=" * 60)
        print(f"Java项目路径: {self.java_project_path}")
        print(f"Neo4j URI: {self.neo4j_uri}")
        print("=" * 60)

        try:
            # 步骤1: 解析Mapper
            if not self.parse_mappers():
                return False

            # 步骤2: 导出关系文件
            if not self.export_relationships():
                return False

            # 步骤3: 写入Neo4j
            if not self.write_to_neo4j(clear_existing):
                return False

            # 完成
            print("\n" + "=" * 60)
            print("🎉 流程执行完成！")
            print("=" * 60)
            print(f"📊 统计信息:")
            print(f"  - 扫描Mapper文件: {len(self.parser.mapper_files)} 个")
            print(f"  - 发现表: {len(self.tables)} 个")
            print(f"  - 发现关系: {len(self.relationships)} 个")
            print("=" * 60)

            return True

        except Exception as e:
            print(f"\n❌ 流程执行失败: {str(e)}")
            import traceback

            traceback.print_exc()
            return False


def main():
    """主函数"""
    print("=" * 60)
    print("MyBatis Mapper 到 Neo4j 自动化工具")
    print("=" * 60)

    # 获取配置
    java_project_path = input("\n请输入Java Spring Boot项目路径: ").strip()

    if not java_project_path:
        print("❌ 项目路径不能为空")
        return

    if not os.path.exists(java_project_path):
        print(f"❌ 路径不存在: {java_project_path}")
        return

    # Neo4j配置（可选）
    use_default_neo4j = input("\n使用默认Neo4j配置? (Y/n): ").strip().lower()

    if use_default_neo4j in ["n", "no"]:
        neo4j_uri = input(f"Neo4j URI (默认: {NEO4J_URI}): ").strip() or NEO4J_URI
        neo4j_user = input(f"Neo4j用户名 (默认: {NEO4J_USER}): ").strip() or NEO4J_USER
        neo4j_password = input(f"Neo4j密码 (默认: {NEO4J_PASSWORD}): ").strip() or NEO4J_PASSWORD
    else:
        neo4j_uri = NEO4J_URI
        neo4j_user = NEO4J_USER
        neo4j_password = NEO4J_PASSWORD

    # 是否清空现有数据
    clear_existing = input("\n是否清空Neo4j现有数据? (y/N): ").strip().lower() in ["y", "yes"]

    # 创建并运行流程
    pipeline = MapperToNeo4jPipeline(
        java_project_path=java_project_path, neo4j_uri=neo4j_uri, neo4j_user=neo4j_user, neo4j_password=neo4j_password
    )

    success = pipeline.run(clear_existing=clear_existing)

    if success:
        print("\n✅ 所有操作已成功完成！")
        print("\n💡 提示:")
        print("  - 关系JSON文件: common/mapper_relationships.json")
        print("  - 关系Python文件: common/generated_relationships.py")
        print("  - 可以在Neo4j Browser中查看图谱")
    else:
        print("\n❌ 操作失败，请检查错误信息")
        sys.exit(1)


if __name__ == "__main__":
    main()
