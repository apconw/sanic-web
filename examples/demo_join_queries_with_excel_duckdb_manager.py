#!/usr/bin/env python3
"""
Excel多Sheet和CSV文件联合查询演示
展示如何读取Excel中的多个sheet和一个CSV文件，注册到DuckDB并执行复杂的联合查询
"""

import os
import sys
import logging
import tempfile
import pandas as pd
from typing import Dict, List, Tuple

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from agent.excel.excel_duckdb_manager import get_duckdb_manager
from agent.excel.excel_duckdb_manager import get_chat_duckdb_manager
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_demo_excel_file() -> str:
    """创建演示用的Excel文件，包含多个sheet"""
    print("📁 创建演示Excel文件...")

    # Sheet 1: 员工信息
    employees_df = pd.DataFrame({
        'employee_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'name': ['张三', '李四', '王五', '赵六', '钱七', '孙八', '周九', '吴十', '郑十一', '陈十二'],
        'department': ['技术部', '销售部', '技术部', '市场部', '销售部', '技术部', '人事部', '市场部', '销售部', '技术部'],
        'position': ['高级工程师', '销售经理', '工程师', '市场专员', '销售代表', '架构师', 'HR专员', '市场经理', '销售总监', '工程师'],
        'salary': [25000, 18000, 15000, 12000, 13000, 30000, 10000, 15000, 22000, 16000],
        'hire_date': ['2020-01-15', '2019-03-20', '2021-06-10', '2022-01-05', '2020-08-12', '2018-11-30', '2021-09-15', '2019-12-01', '2017-05-20', '2022-03-10'],
        'manager_id': [None, 10, 1, 4, 10, None, 7, 4, 10, 1]
    })

    # Sheet 2: 部门信息
    departments_df = pd.DataFrame({
        'department_id': [1, 2, 3, 4, 5],
        'department_name': ['技术部', '销售部', '市场部', '人事部', '财务部'],
        'budget': [500000, 300000, 200000, 150000, 180000],
        'location': ['北京', '上海', '深圳', '广州', '杭州'],
        'head_id': [1, 2, 3, 7, None]
    })

    # Sheet 3: 项目信息
    projects_df = pd.DataFrame({
        'project_id': [101, 102, 103, 104, 105, 106, 107, 108],
        'project_name': ['电商平台升级', '移动应用开发', '数据分析系统', '品牌推广活动', 'ERP系统实施', 'CRM系统优化', '员工培训计划', '财务系统迁移'],
        'department_id': [1, 1, 1, 3, 2, 2, 4, 5],
        'start_date': ['2023-01-01', '2023-02-15', '2023-03-01', '2023-04-01', '2023-05-01', '2023-06-01', '2023-07-01', '2023-08-01'],
        'end_date': ['2023-06-30', '2023-12-31', '2023-09-30', '2023-06-30', '2024-02-28', '2023-12-31', '2023-12-31', '2024-03-31'],
        'budget': [200000, 150000, 100000, 80000, 120000, 90000, 50000, 180000],
        'status': ['进行中', '进行中', '已完成', '已完成', '计划中', '计划中', '进行中', '计划中']
    })

    # 创建Excel文件
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
        excel_path = tmp_file.name

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        employees_df.to_excel(writer, sheet_name='员工信息', index=False)
        departments_df.to_excel(writer, sheet_name='部门信息', index=False)
        projects_df.to_excel(writer, sheet_name='项目信息', index=False)

    print(f"✅ Excel文件创建完成: {excel_path}")
    return excel_path

def create_demo_csv_file() -> str:
    """创建演示用的CSV文件"""
    print("📁 创建演示CSV文件...")

    # 销售数据
    sales_df = pd.DataFrame({
        'sale_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        'employee_id': [2, 5, 10, 3, 1, 8, 9, 4, 6, 7, 1, 5, 2, 9, 3],
        'customer_name': ['腾讯科技', '阿里巴巴', '京东集团', '字节跳动', '美团', '百度', '网易', '小米', '华为', 'OPPO', 'VIVO', '腾讯科技', '阿里巴巴', '京东集团', '字节跳动'],
        'product_name': ['企业版软件', '云服务套餐', '物流解决方案', '广告投放', '外卖配送服务', '搜索引擎优化', '游戏联运', '手机销售', '通信设备', '智能手机', '智能手机', '企业版软件', '云服务套餐', '物流解决方案', '广告投放'],
        'amount': [50000, 80000, 120000, 60000, 30000, 45000, 70000, 200000, 150000, 180000, 160000, 55000, 90000, 130000, 65000],
        'sale_date': ['2023-10-01', '2023-10-02', '2023-10-03', '2023-10-04', '2023-10-05', '2023-10-06', '2023-10-07', '2023-10-08', '2023-10-09', '2023-10-10', '2023-10-11', '2023-10-12', '2023-10-13', '2023-10-14', '2023-10-15'],
        'region': ['华南', '华东', '华北', '华北', '华南', '华东', '华东', '华西', '华西', '华西', '华西', '华南', '华东', '华北', '华北']
    })

    # 创建CSV文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as tmp_file:
        csv_path = tmp_file.name
        sales_df.to_csv(csv_path, index=False, encoding='utf-8')

    print(f"✅ CSV文件创建完成: {csv_path}")
    return csv_path

def demo_multi_sheet_csv_queries():
    """演示多Sheet和CSV文件的联合查询功能"""
    print("🚀 开始演示Excel多Sheet和CSV文件联合查询")
    print("=" * 80)

    try:
        # 创建演示文件
        excel_file = create_demo_excel_file()
        csv_file = create_demo_csv_file()

        # 使用chat_id级别的DuckDB管理器
        chat_id = "demo_session_001"
        manager = get_duckdb_manager(chat_id=chat_id)
        print(f"🔧 使用chat_id '{chat_id}' 的DuckDB管理器")

        # 注册Excel文件
        print("\n📊 注册Excel文件...")
        catalog_excel, tables_excel = manager.register_excel_file(excel_file, "公司数据.xlsx")
        print(f"✅ Excel文件注册完成:")
        print(f"   Catalog: {catalog_excel}")
        print(f"   表数量: {len(tables_excel)}")
        for table_name in tables_excel:
            print(f"   - {table_name}: {tables_excel[table_name].row_count} 行, {tables_excel[table_name].column_count} 列")

        # 注册CSV文件
        print("\n📊 注册CSV文件...")
        catalog_csv, tables_csv = manager.register_csv_file(csv_file, "销售数据.csv")
        print(f"✅ CSV文件注册完成:")
        print(f"   Catalog: {catalog_csv}")
        print(f"   表数量: {len(tables_csv)}")
        for table_name in tables_csv:
            print(f"   - {table_name}: {tables_csv[table_name].row_count} 行, {tables_csv[table_name].column_count} 列")

        # 构建表名引用
        employees_table = f'"{catalog_excel}"."员工信息"'
        departments_table = f'"{catalog_excel}"."部门信息"'
        projects_table = f'"{catalog_excel}"."项目信息"'
        sales_table = f'"{catalog_csv}"."销售数据"'

        print(f"\n🔍 开始执行联合查询...")
        print("-" * 60)

        # 查询1: 员工及其部门详细信息
        print("\n📋 查询1: 员工及其部门详细信息")
        print("-" * 40)
        sql1 = f"""
        SELECT
            e.name AS 员工姓名,
            e.position AS 职位,
            e.salary AS 薪资,
            d.department_name AS 部门,
            d.location AS 地点,
            d.budget AS 部门预算,
            CASE
                WHEN e.manager_id IS NULL THEN '是'
                ELSE '否'
            END AS 是否主管
        FROM {employees_table} e
        LEFT JOIN {departments_table} d ON SUBSTRING(e.department, 1, 2) = SUBSTRING(d.department_name, 1, 2)
        ORDER BY e.salary DESC
        """

        columns1, data1 = manager.execute_sql(sql1)
        print(f"结果: {len(data1)} 条记录")
        for record in data1[:10]:  # 显示前10条
            print(f"  {record['员工姓名']:8} | {record['职位']:10} | {record['薪资']:>8,} | "
                  f"{record['部门']:6} | {record['地点']:6} | "
                  f"{record['是否主管']:4} | {record['部门预算']:>8,.0f}")

        # 查询2: 部门项目统计
        print(f"\n📊 查询2: 部门项目统计")
        print("-" * 40)
        sql2 = f"""
        SELECT
            d.department_name AS 部门名称,
            d.location AS 地点,
            COUNT(p.project_id) AS 项目数量,
            SUM(p.budget) AS 总预算,
            ROUND(AVG(p.budget), 0) AS 平均预算,
            STRING_AGG(p.project_name, ', ' ORDER BY p.project_name) AS 项目列表
        FROM {departments_table} d
        LEFT JOIN {projects_table} p ON d.department_id = p.department_id
        GROUP BY d.department_id, d.department_name, d.location, d.budget
        ORDER BY 总预算 DESC
        """

        columns2, data2 = manager.execute_sql(sql2)
        print(f"结果: {len(data2)} 条记录")
        for record in data2:
            print(f"  {record['部门名称']:8} | {record['地点']:6} | "
                  f"{record['项目数量']:3}个 | {record['总预算']:>8,.0f} | "
                  f"{record['平均预算']:>8,.0f}")
            print(f"    项目: {record['项目列表'][:50]}...")

        # 查询3: 员工销售业绩分析
        print(f"\n💰 查询3: 员工销售业绩分析")
        print("-" * 40)
        sql3 = f"""
        SELECT
            e.name AS 员工姓名,
            e.department AS 部门,
            COUNT(s.sale_id) AS 销售单数,
            ROUND(SUM(s.amount), 2) AS 总销售额,
            ROUND(AVG(s.amount), 2) AS 平均单笔金额,
            MAX(s.amount) AS 最大单笔金额,
            COUNT(DISTINCT s.customer_name) AS 客户数
        FROM {sales_table} s
        JOIN {employees_table} e ON s.employee_id = e.employee_id
        GROUP BY e.employee_id, e.name, e.department
        ORDER BY 总销售额 DESC
        """

        columns3, data3 = manager.execute_sql(sql3)
        print(f"结果: {len(data3)} 条记录")
        for record in data3:
            print(f"  🏆 {record['员工姓名']:8} ({record['部门']:6})")
            print(f"     销售单数: {record['销售单数']:2} | "
                  f"总销售额: {record['总销售额']:>10,.2f} | "
                  f"平均单笔: {record['平均单笔金额']:>8,.2f} | "
                  f"最大单笔: {record['最大单笔金额']:>8,.2f} | "
                  f"客户数: {record['客户数']:2}")

        # 查询4: 项目参与详情
        print(f"\n🔗 查询4: 项目参与详情")
        print("-" * 40)
        sql4 = f"""
        SELECT
            p.project_name AS 项目名称,
            p.status AS 状态,
            e.name AS 负责人,
            e.position AS 职位,
            d.department_name AS 部门
        FROM {projects_table} p
        LEFT JOIN {employees_table} e ON p.department_id = (
            SELECT department_id FROM {departments_table}
            WHERE department_name = e.department
        )
        LEFT JOIN {departments_table} d ON p.department_id = d.department_id
        ORDER BY p.start_date, p.project_name
        """

        columns4, data4 = manager.execute_sql(sql4)
        print(f"结果: {len(data4)} 条记录")
        for record in data4:
            name = record['负责人'] or '未指定'
            position = record['职位'] or '未指定'
            department = record['部门'] or '未指定'
            print(f"  📋 {record['项目名称']:20} ({record['状态']:6})")
            print(f"     负责人: {name:8} ({position:10}) - {department}")
            print()

        # 查询5: 综合业绩报告（技术部门专项）
        print(f"\n📈 查询5: 综合业绩报告（技术部门专项）")
        print("-" * 40)
        sql5 = f"""
        WITH tech_employees AS (
            SELECT employee_id, name, department, salary
            FROM {employees_table}
            WHERE department = '技术部'
        ),
        tech_sales AS (
            SELECT
                e.employee_id,
                e.name,
                COUNT(s.sale_id) as sales_count,
                SUM(s.amount) as sales_amount
            FROM {sales_table} s
            JOIN tech_employees e ON s.employee_id = e.employee_id
            GROUP BY e.employee_id, e.name
        ),
        tech_projects AS (
            SELECT
                COUNT(DISTINCT p.project_id) as project_count,
                SUM(p.budget) as project_budget
            FROM {projects_table} p
            JOIN {departments_table} d ON p.department_id = d.department_id
            WHERE d.department_name = '技术部'
        )
        SELECT
            (SELECT COUNT(*) FROM tech_employees) as 技术部人数,
            (SELECT ROUND(AVG(salary), 0) FROM tech_employees) as 平均薪资,
            (SELECT ROUND(SUM(COALESCE(sales_amount, 0)), 2) FROM tech_sales) as 技术部总销售额,
            (SELECT project_count FROM tech_projects) as 参与项目数,
            (SELECT project_budget FROM tech_projects) as 项目总预算
        """

        columns5, data5 = manager.execute_sql(sql5)
        if data5:
            record = data5[0]
            print(f"  👥 技术部人数: {record['技术部人数']} 人")
            print(f"  💰 平均薪资: ¥{record['平均薪资']:,}")
            print(f"  💎 总销售额: ¥{record['技术部总销售额']:,}")
            print(f"  📋 参与项目数: {record['参与项目数']} 个")
            print(f"  💵 项目总预算: ¥{record['项目总预算']:,}")

        # 查询6: 月度销售趋势
        print(f"\n📅 查询6: 月度销售趋势分析")
        print("-" * 40)
        sql6 = f"""
        SELECT
            strftime(CAST(s.sale_date AS DATE), '%Y-%m') AS 月份,
            COUNT(DISTINCT s.employee_id) AS 销售人数,
            COUNT(s.sale_id) AS 销售单数,
            ROUND(SUM(s.amount), 2) AS 月度销售额,
            COUNT(DISTINCT s.customer_name) AS 客户数,
            STRING_AGG(DISTINCT s.region, ', ') AS 销售区域
        FROM {sales_table} s
        GROUP BY strftime(CAST(s.sale_date AS DATE), '%Y-%m')
        ORDER BY 月份
        """

        columns6, data6 = manager.execute_sql(sql6)
        print(f"结果: {len(data6)} 个月的数据")
        for record in data6:
            print(f"  📅 {record['月份']}: {record['销售单数']}单 | "
                  f"¥{record['月度销售额']:>10,.2f} | "
                  f"{record['客户数']} 客户 | 区域: {record['销售区域']}")

        # 显示会话统计信息
        print(f"\n📊 会话统计信息")
        print("-" * 40)

        chat_manager = get_chat_duckdb_manager()
        stats = {
            "active_chat_count": chat_manager.get_active_chat_count(),
            "chat_list": chat_manager.get_chat_list()
        }
        print(f"  活跃会话数: {stats['active_chat_count']}")
        print(f"  会话列表: {stats['chat_list']}")

        print(f"\n✅ 演示完成！成功处理了Excel多Sheet和CSV文件的复杂联合查询。")
        print(f"\n💡 演示总结:")
        print(f"  - Excel文件包含3个Sheet: 员工信息、部门信息、项目信息")
        print(f"  - CSV文件包含销售数据: 15条销售记录")
        print(f"  - 执行了6个不同类型的联合查询")
        print(f"  - 使用chat_id '{chat_id}' 确保数据隔离")
        print(f"  - 总计处理了: {len(tables_excel) + len(tables_csv)} 个数据表")

    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # 清理临时文件
        try:
            if 'excel_file' in locals() and os.path.exists(excel_file):
                os.unlink(excel_file)
            if 'csv_file' in locals() and os.path.exists(csv_file):
                os.unlink(csv_file)
        except:
            pass

        # 清理会话
        try:
            from agent.excel.excel_duckdb_manager import close_duckdb_manager
            close_duckdb_manager(chat_id=chat_id)
        except:
            pass

    return True

if __name__ == "__main__":
    print("🎯 Excel多Sheet和CSV文件联合查询演示")
    print("=" * 80)
    print("本演示展示如何:")
    print("1. 读取Excel文件中的多个Sheet")
    print("2. 读取CSV文件")
    print("3. 将数据注册到DuckDB")
    print("4. 执行复杂的联合查询")
    print("5. 使用chat_id级别的数据隔离")
    print("=" * 80)

    success = demo_multi_sheet_csv_queries()

    if success:
        print(f"\n🎉 演示成功完成！")
        print(f"\n📚 更多信息:")
        print(f"  - 查看文档: docs/chat_id_duckdb_manager.md")
        print(f"  - 查看代码: agent/excel/excel_duckdb_manager.py")
        print(f"  - 查看示例: examples/")
        sys.exit(0)
    else:
        print(f"\n❌ 演示失败")
        sys.exit(1)