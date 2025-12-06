import os
import re
import xml.etree.ElementTree as ET
from typing import List, Dict, Set, Tuple
import json

"""
MyBatis Mapper.xml 解析器
用于从Java Spring Boot项目中扫描和解析MyBatis的mapper.xml文件
提取SQL语句中的表关系信息
"""


class MyBatisMapperParser:
    """MyBatis Mapper XML 解析器"""

    def __init__(self, project_path: str):
        """
        初始化解析器
        :param project_path: Java Spring Boot项目根目录路径
        """
        self.project_path = project_path
        self.mapper_files = []
        self.relationships = []
        self.tables = set()

    def scan_mapper_files(self) -> List[str]:
        """
        扫描 Spring Boot 项目中所有 MyBatis Mapper XML 文件。
        """
        mapper_files = []
        # 要跳过的目录
        skip_dirs = {"target", "build", ".git", "node_modules", ".idea", ".vscode", "__pycache__", "dist", "out"}

        for root, dirs, files in os.walk(self.project_path):
            # 过滤掉不需要的目录
            dirs[:] = [d for d in dirs if d not in skip_dirs]

            for file in files:
                # 检查是否为XML文件且文件名包含'mapper'(不区分大小写)
                if file.lower().endswith(".xml") and "mapper" in file.lower():
                    full_path = os.path.join(root, file)
                    mapper_files.append(full_path)
                    print(f"找到Mapper文件: {full_path}")

        # 重要：将找到的文件保存到实例变量中
        self.mapper_files = mapper_files
        print(f"总共找到 {len(mapper_files)} 个Mapper文件")
        return mapper_files

    def parse_mapper_file(self, mapper_file: str) -> List[Dict]:
        """
        解析单个mapper.xml文件
        :param mapper_file: mapper文件路径
        :return: 从该文件中提取的关系列表
        """
        relationships = []

        try:
            tree = ET.parse(mapper_file)
            root = tree.getroot()

            # 获取namespace
            namespace = root.get("namespace", "")
            print(f"\n📄 解析文件: {os.path.basename(mapper_file)}")
            print(f"   Namespace: {namespace}")

            # 遍历所有SQL语句节点
            sql_nodes = root.findall(".//*")
            for node in sql_nodes:
                if node.tag in ["select", "insert", "update", "delete"]:
                    sql_id = node.get("id", "unknown")
                    sql_text = self._extract_sql_text(node)

                    if sql_text:
                        # 从SQL中提取表关系
                        rels = self._extract_relationships_from_sql(sql_text, sql_id, mapper_file)
                        relationships.extend(rels)

        except Exception as e:
            print(f"❌ 解析文件失败 {mapper_file}: {str(e)}")

        return relationships

    def _extract_sql_text(self, node: ET.Element) -> str:
        """
        提取SQL文本（包括子节点）
        :param node: XML节点
        :return: SQL文本
        """
        sql_parts = []

        # 获取节点文本
        if node.text:
            sql_parts.append(node.text.strip())

        # 递归获取子节点文本
        for child in node:
            child_text = self._extract_sql_text(child)
            if child_text:
                sql_parts.append(child_text)
            if child.tail:
                sql_parts.append(child.tail.strip())

        return " ".join(sql_parts)

    def _extract_relationships_from_sql(self, sql: str, sql_id: str, source_file: str) -> List[Dict]:
        """
        从SQL语句中提取表关系
        :param sql: SQL语句
        :param sql_id: SQL语句ID
        :param source_file: 来源文件
        :return: 关系列表
        """
        relationships = []

        # 清理SQL（移除注释和多余空格）
        sql = self._clean_sql(sql)

        # 提取所有表名
        tables = self._extract_tables_from_sql(sql)
        self.tables.update(tables)

        # 提取JOIN关系
        join_relationships = self._extract_join_relationships(sql, tables)
        relationships.extend(join_relationships)

        # 提取外键关系（从WHERE子句）
        fk_relationships = self._extract_foreign_key_relationships(sql, tables)
        relationships.extend(fk_relationships)

        # 为每个关系添加元数据
        for rel in relationships:
            rel["source_file"] = os.path.basename(source_file)
            rel["sql_id"] = sql_id

        return relationships

    def _clean_sql(self, sql: str) -> str:
        """
        清理SQL语句
        :param sql: 原始SQL
        :return: 清理后的SQL
        """
        # 移除单行注释
        sql = re.sub(r"--.*?$", "", sql, flags=re.MULTILINE)
        # 移除多行注释
        sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
        # 移除多余空格
        sql = re.sub(r"\s+", " ", sql)
        return sql.strip()

    def _extract_tables_from_sql(self, sql: str) -> Set[str]:
        """
        从SQL中提取所有表名
        :param sql: SQL语句
        :return: 表名集合
        """
        tables = set()

        # 匹配FROM子句中的表名
        from_pattern = r"FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)"
        from_matches = re.findall(from_pattern, sql, re.IGNORECASE)
        tables.update([t.lower() for t in from_matches])

        # 匹配JOIN子句中的表名
        join_pattern = r"JOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)"
        join_matches = re.findall(join_pattern, sql, re.IGNORECASE)
        tables.update([t.lower() for t in join_matches])

        # 匹配INSERT INTO中的表名
        insert_pattern = r"INSERT\s+INTO\s+([a-zA-Z_][a-zA-Z0-9_]*)"
        insert_matches = re.findall(insert_pattern, sql, re.IGNORECASE)
        tables.update([t.lower() for t in insert_matches])

        # 匹配UPDATE中的表名
        update_pattern = r"UPDATE\s+([a-zA-Z_][a-zA-Z0-9_]*)"
        update_matches = re.findall(update_pattern, sql, re.IGNORECASE)
        tables.update([t.lower() for t in update_matches])

        return tables

    def _extract_join_relationships(self, sql: str, tables: Set[str]) -> List[Dict]:
        """
        从JOIN语句中提取表关系
        :param sql: SQL语句
        :param tables: 表名集合
        :return: 关系列表
        """
        relationships = []

        # 匹配各种JOIN语句
        # 格式: JOIN table2 ON table1.field1 = table2.field2
        join_pattern = r"(LEFT\s+JOIN|RIGHT\s+JOIN|INNER\s+JOIN|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*)?\s+ON\s+([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*)"

        matches = re.findall(join_pattern, sql, re.IGNORECASE)

        for match in matches:
            join_type, table2, alias, left_field, right_field = match

            # 解析字段
            left_parts = left_field.split(".")
            right_parts = right_field.split(".")

            if len(left_parts) == 2 and len(right_parts) == 2:
                table1_ref = left_parts[0].lower()
                field1 = left_parts[1].lower()
                table2_ref = right_parts[0].lower()
                field2 = right_parts[1].lower()

                # 确定实际表名（可能是别名）
                table2_name = table2.lower()

                # 尝试匹配实际表名
                from_table = self._resolve_table_name(table1_ref, tables)
                to_table = self._resolve_table_name(table2_ref, tables) or table2_name

                if from_table and to_table:
                    relationship = {
                        "from_table": from_table,
                        "to_table": to_table,
                        "description": f"{from_table} {join_type.lower()} {to_table}",
                        "field_relation": f"{field1} references {field2}",
                        "join_type": join_type.strip().upper(),
                    }
                    relationships.append(relationship)
                    print(f"   🔗 发现JOIN关系: {from_table}.{field1} -> {to_table}.{field2}")

        return relationships

    def _extract_foreign_key_relationships(self, sql: str, tables: Set[str]) -> List[Dict]:
        """
        从WHERE子句中提取外键关系
        :param sql: SQL语句
        :param tables: 表名集合
        :return: 关系列表
        """
        relationships = []

        # 匹配WHERE子句中的表关联
        # 格式: table1.field1 = table2.field2
        where_pattern = r"WHERE.*?([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*)"

        matches = re.findall(where_pattern, sql, re.IGNORECASE)

        for match in matches:
            left_field, right_field = match

            left_parts = left_field.split(".")
            right_parts = right_field.split(".")

            if len(left_parts) == 2 and len(right_parts) == 2:
                table1_ref = left_parts[0].lower()
                field1 = left_parts[1].lower()
                table2_ref = right_parts[0].lower()
                field2 = right_parts[1].lower()

                from_table = self._resolve_table_name(table1_ref, tables)
                to_table = self._resolve_table_name(table2_ref, tables)

                if from_table and to_table and from_table != to_table:
                    relationship = {
                        "from_table": from_table,
                        "to_table": to_table,
                        "description": f"{from_table} references {to_table}",
                        "field_relation": f"{field1} references {field2}",
                        "join_type": "WHERE",
                    }
                    relationships.append(relationship)
                    print(f"   🔗 发现WHERE关系: {from_table}.{field1} -> {to_table}.{field2}")

        return relationships

    def _resolve_table_name(self, table_ref: str, tables: Set[str]) -> str:
        """
        解析表名（处理别名）
        :param table_ref: 表引用（可能是别名）
        :param tables: 已知表名集合
        :return: 实际表名
        """
        table_ref_lower = table_ref.lower()

        # 如果直接匹配，返回
        if table_ref_lower in tables:
            return table_ref_lower

        # 尝试匹配表名的首字母缩写
        for table in tables:
            # 检查是否是表名的缩写（如 t_user -> tu 或 u）
            if table.startswith(table_ref_lower):
                return table

            # 检查首字母
            initials = "".join([word[0] for word in table.split("_") if word])
            if initials == table_ref_lower:
                return table

        return table_ref_lower

    def parse_all_mappers(self) -> List[Dict]:
        """
        解析所有mapper文件
        :return: 所有关系列表
        """
        all_relationships = []

        for mapper_file in self.mapper_files:
            rels = self.parse_mapper_file(mapper_file)
            all_relationships.extend(rels)

        # 去重
        self.relationships = self._deduplicate_relationships(all_relationships)
        print(f"\n📊 共提取 {len(self.relationships)} 个唯一的表关系")

        return self.relationships

    def _deduplicate_relationships(self, relationships: List[Dict]) -> List[Dict]:
        """
        去除重复的关系
        :param relationships: 关系列表
        :return: 去重后的关系列表
        """
        seen = set()
        unique_relationships = []

        for rel in relationships:
            # 创建关系的唯一标识
            key = (rel["from_table"], rel["to_table"], rel["field_relation"])

            if key not in seen:
                seen.add(key)
                unique_relationships.append(rel)

        return unique_relationships

    def export_to_json(self, output_file: str):
        """
        导出关系到JSON文件
        :param output_file: 输出文件路径
        """
        data = {
            "tables": list(self.tables),
            "relationships": self.relationships,
            "total_tables": len(self.tables),
            "total_relationships": len(self.relationships),
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"✅ 关系已导出到: {output_file}")

    def export_to_python_format(self, output_file: str):
        """
        导出为Python格式（用于initialize_neo4j.py）
        :param output_file: 输出文件路径
        """
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("# 自动生成的表关系配置\n")
            f.write("# 生成时间: " + __import__("datetime").datetime.now().isoformat() + "\n\n")
            f.write("RELATIONSHIPS = [\n")

            for rel in self.relationships:
                f.write("    {\n")
                f.write(f"        \"from_table\": \"{rel['from_table']}\",\n")
                f.write(f"        \"to_table\": \"{rel['to_table']}\",\n")
                f.write(f"        \"description\": \"{rel['description']}\",\n")
                f.write(f"        \"field_relation\": \"{rel['field_relation']}\",\n")
                if "source_file" in rel:
                    f.write(f"        # 来源: {rel['source_file']} - {rel['sql_id']}\n")
                f.write("    },\n")

            f.write("]\n")

        print(f"✅ Python格式关系已导出到: {output_file}")


def main():
    """主函数示例"""
    # 配置Java项目路径
    java_project_path = input("请输入Java Spring Boot项目路径: ").strip()

    if not os.path.exists(java_project_path):
        print(f"❌ 路径不存在: {java_project_path}")
        return

    # 创建解析器
    parser = MyBatisMapperParser(java_project_path)

    # 扫描mapper文件并保存结果
    mapper_files = parser.scan_mapper_files()
    parser.mapper_files = mapper_files  # 确保将结果保存到实例变量

    if not parser.mapper_files:
        print("❌ 未找到任何mapper.xml文件")
        return

    # 解析所有mapper
    parser.parse_all_mappers()

    # 导出结果
    output_dir = os.path.dirname(os.path.abspath(__file__))
    json_output = os.path.join(output_dir, "mapper_relationships.json")
    python_output = os.path.join(output_dir, "generated_relationships.py")

    parser.export_to_json(json_output)
    parser.export_to_python_format(python_output)

    print("\n🎉 解析完成！")
    print(f"📊 发现表: {len(parser.tables)} 个")
    print(f"🔗 发现关系: {len(parser.relationships)} 个")


if __name__ == "__main__":
    main()
