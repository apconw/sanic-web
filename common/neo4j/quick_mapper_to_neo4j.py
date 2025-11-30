#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
快速执行脚本 - MyBatis Mapper 到 Neo4j
使用预设配置快速执行完整流程
"""

import os
import sys

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mapper_to_neo4j import MapperToNeo4jPipeline

# ==================== 配置区域 ====================
# 在这里修改你的配置

# Java Spring Boot 项目路径（必填）
JAVA_PROJECT_PATH = "/Users/lihuan/java-projects/microbrain-passcloud"

# Neo4j 配置
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "neo4j123"

# 是否清空Neo4j现有数据
CLEAR_EXISTING_DATA = False


# ==================== 配置结束 ====================


def get_user_confirmation(prompt="确认执行?"):
    """
    获取用户确认输入

    Args:
        prompt (str): 提示信息

    Returns:
        bool: True表示确认，False表示取消
    """
    while True:
        user_input = input(f"{prompt} (Y/n): ").strip().lower()

        # 如果用户直接回车，默认为yes
        if user_input == "":
            return True

        # 处理各种可能的yes输入
        if user_input in ["y", "yes", "是", "确认", "ok"]:
            return True

        # 处理各种可能的no输入
        if user_input in ["n", "no", "否", "取消"]:
            return False

        print("请输入 y/yes/是 或 n/no/否")


def validate_configuration():
    """
    验证配置是否正确

    Returns:
        tuple: (is_valid, error_message)
    """
    # 检查是否已修改默认路径
    if JAVA_PROJECT_PATH == "/path/to/your/java/project" and not os.path.exists(JAVA_PROJECT_PATH):
        return (
            False,
            "请先在脚本中配置 JAVA_PROJECT_PATH\n   编辑文件: common/quick_mapper_to_neo4j.py\n   修改 JAVA_PROJECT_PATH 为你的Java项目路径",
        )

    # 检查项目路径是否存在
    if not os.path.exists(JAVA_PROJECT_PATH):
        return False, f"项目路径不存在: {JAVA_PROJECT_PATH}"

    return True, ""


def display_configuration():
    """显示当前配置信息"""
    print(f"\n📋 当前配置:")
    print(f"  Java项目: {JAVA_PROJECT_PATH}")
    print(f"  Neo4j URI: {NEO4J_URI}")
    print(f"  Neo4j用户: {NEO4J_USER}")
    print(f"  清空现有数据: {'是' if CLEAR_EXISTING_DATA else '否'}")
    print()


def quick_run():
    """快速运行"""
    print("=" * 60)
    print("🚀 快速执行模式")
    print("=" * 60)

    # 验证配置
    is_valid, error_msg = validate_configuration()
    if not is_valid:
        print(f"\n❌ 错误: {error_msg}")
        return False

    display_configuration()

    # 确认执行
    if not get_user_confirmation():
        print("❌ 已取消")
        return False

    try:
        # 创建并运行流程
        print("⏳ 正在执行转换流程...")
        pipeline = MapperToNeo4jPipeline(
            java_project_path=JAVA_PROJECT_PATH,
            neo4j_uri=NEO4J_URI,
            neo4j_user=NEO4J_USER,
            neo4j_password=NEO4J_PASSWORD,
        )

        success = pipeline.run(clear_existing=CLEAR_EXISTING_DATA)

        if success:
            print("\n" + "=" * 60)
            print("✅ 执行成功！")
            print("=" * 60)
            print("\n📁 生成的文件:")
            print("  - mapper_relationships.json (JSON格式关系)")
            print("  - generated_relationships.py (Python格式关系)")
            print("\n💡 下一步:")
            print("  1. 打开 Neo4j Browser: http://localhost:7474")
            print("  2. 执行查询查看图谱:")
            print("     MATCH (n:Table)-[r]->(m:Table) RETURN n,r,m LIMIT 50")
            print("=" * 60)
        else:
            print("\n❌ 执行失败")
            return False

    except Exception as e:
        print(f"\n❌ 执行过程中发生错误: {str(e)}")
        return False

    return True


if __name__ == "__main__":
    success = quick_run()
    sys.exit(0 if success else 1)
