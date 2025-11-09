import json
import traceback

from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
import logging

from common.llm_util import get_llm

logger = logging.getLogger(__name__)


def sql_generate(state):

    llm = get_llm()

    prompt = ChatPromptTemplate.from_template(
        """
        你是一位专业的数据库管理员（DBA），任务是根据提供的数据库结构、表关系以及用户需求，生成优化的MYSQL SQL查询语句，并推荐合适的可视化图表。
          
        ## 任务
          - 根据用户问题生成一条优化的SQL语句。
          - 根据查询生成逻辑从**图表定义**中选择最合适的图表类型。
          
        ## 约束条件
         1. 你必须仅生成一条合法、可执行的SQL查询语句 —— 不得包含解释、Markdown、注释或额外文本。
         2. **必须直接且完整地使用所提供的表结构和表关系来生成SQL语句**。
         3. 你必须严格遵守数据类型、外键关系及表结构中定义的约束。
         4. 使用适当的SQL子句（JOIN、WHERE、GROUP BY、HAVING、ORDER BY、LIMIT等）以确保准确性和性能。
         5. 若问题涉及时序，请合理使用提供的“当前时间”上下文（例如用于相对日期计算）。
         6. 不得假设表结构中未明确定义的列或表。
         7. 如果用户问题模糊或者缺乏足够的信息以生成正确的查询，请返回：`NULL`
         8. 当用户明确要求查看明细数据且未指定具体数量时，应适当限制返回结果数量（如LIMIT 50）以提高查询性能，但如果用户指定了具体数量则按照用户要求执行
         9. 对于聚合查询或统计类查询，不应随意添加LIMIT子句
        
       ## 提供的信息
        - 表结构：{db_schema}
        - 表关系：{table_relationship}
        - 用户提问：{user_query}
        - 当前时间：{current_time}
             
        ## 图表定义
        - generate_area_chart: used to display the trend of data under a continuous independent variable, allowing observation of overall data trends.
        - generate_bar_chart: used to compare values across different categories, suitable for horizontal comparisons.
        - generate_boxplot_chart: used to display the distribution of data, including the median, quartiles, and outliers.
        - generate_column_chart: used to compare values across different categories, suitable for vertical comparisons.
        - generate_district_map: Generate a district-map, used to show administrative divisions and data distribution.
        - generate_dual_axes_chart: Generate a dual-axes chart, used to display the relationship between two variables with different units or ranges.
        - generate_fishbone_diagram: Generate a fishbone diagram, also known as an Ishikawa diagram, used to identify and display the root causes of a problem.
        - generate_flow_diagram: Generate a flowchart, used to display the steps and sequence of a process.
        - generate_funnel_chart: Generate a funnel chart, used to display data loss at different stages.
        - generate_histogram_chart: Generate a histogram, used to display the distribution of data by dividing it into intervals and counting the number of data points in each interval.
        - generate_line_chart: Generate a line chart, used to display the trend of data over time or another continuous variable.
        - generate_liquid_chart: Generate a liquid chart, used to display the proportion of data, visually representing percentages in the form of water-filled spheres.
        - generate_mind_map: Generate a mind-map, used to display thought processes and hierarchical information.
        - generate_network_graph: Generate a network graph, used to display relationships and connections between nodes.
        - generate_organization_chart: Generate an organizational chart, used to display the structure of an organization and personnel relationships.
        - generate_path_map: Generate a path-map, used to display route planning results for POIs.
        - generate_pie_chart: Generate a pie chart, used to display the proportion of data, dividing it into parts represented by sectors showing the percentage of each part.
        - generate_pin_map: Generate a pin-map, used to show the distribution of POIs.
        - generate_radar_chart: Generate a radar chart, used to display multi-dimensional data comprehensively, showing multiple dimensions in a radar-like format.
        - generate_sankey_chart: Generate a sankey chart, used to display data flow and volume, representing the movement of data between different nodes in a Sankey-style format.
        - generate_scatter_chart: Generate a scatter plot, used to display the relationship between two variables, showing data points as scattered dots on a coordinate system.
        - generate_treemap_chart: Generate a treemap, used to display hierarchical data, showing data in rectangular forms where the size of rectangles represents the value of the data.
        - generate_venn_chart: Generate a venn diagram, used to display relationships between sets, including intersections, unions, and differences.
        - generate_violin_chart: Generate a violin plot, used to display the distribution of data, combining features of boxplots and density plots to provide a more detailed view of the data distribution.
        - generate_word_cloud_chart: Generate a word-cloud, used to display the frequency of words in textual data, with font sizes indicating the frequency of each word.
        - generate_table: Generate a structured table, used to organize and present data in rows and columns, facilitating clear and concise information display for easy reading and analysis.
        
        ## 输出格式
        - 你**必须且只能**输出一个符合以下结构的 **纯 JSON 对象**，不得包含任何额外文本、注释、换行或 Markdown 格式：
        ```json
        {{
            "sql_query": "生成的SQL语句字符串",
            "chart_type": "推荐的图表类型字符串，如 \"generate_area_chart\""
        }}
    """
    )

    chain = prompt | llm

    try:
        response = chain.invoke(
            {
                "db_schema": state["db_info"],
                "user_query": state["user_query"],
                "table_relationship": state.get("table_relationship", []),
                "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

        state["attempts"] += 1
        clean_json_str = response.content.strip().removeprefix("```json").strip().removesuffix("```").strip()
        state["generated_sql"] = json.loads(clean_json_str)["sql_query"]
        # mcp-hub 服务默认添加前缀防止重复问题
        state["chart_type"] = "mcp-server-chart-" + json.loads(clean_json_str)["chart_type"]

    except Exception as e:
        traceback.print_exception(e)
        logger.error(f"Error in generating: {e}")
        state["generated_sql"] = "No SQL query generated"

    return state
