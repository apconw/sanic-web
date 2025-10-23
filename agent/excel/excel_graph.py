import logging

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from agent.excel.excel_agent_state import ExcelAgentState
from agent.excel.excel_data_render_apache import excel_data_render_apache
from agent.excel.excel_excute_sql import exe_sql_excel_query
from agent.excel.excel_mapping_node import read_excel_columns
from agent.excel.excel_sql_node import sql_generate_excel
from agent.text2sql.analysis.data_render_antv import data_render_ant
from agent.text2sql.analysis.data_render_apache import data_render_apache
from agent.text2sql.analysis.llm_summarizer import summarize

logger = logging.getLogger(__name__)


def data_render_condition(state: ExcelAgentState) -> str:
    """
    根据 chart_type 判断如果是表格则使用eChart进行渲染
    否则使用antV进行渲染
    """
    chart_type = state.get("chart_type")
    logger.info(f"chart_type: {chart_type}")
    if not chart_type or chart_type.lower() in ["mcp-server-chart-generate_table"]:
        return "data_render_apache"  # 直接跳转到总结节点

    return "data_render"


def create_excel_graph():
    """
    :return:
    """
    graph = StateGraph(ExcelAgentState)

    graph.add_node("excel_parsing", read_excel_columns)
    graph.add_node("sql_generator", sql_generate_excel)
    graph.add_node("sql_executor", exe_sql_excel_query)
    graph.add_node("summarize", summarize)
    graph.add_node("data_render", data_render_ant)
    graph.add_node("data_render_apache", excel_data_render_apache)

    graph.set_entry_point("excel_parsing")
    graph.add_edge("excel_parsing", "sql_generator")
    graph.add_edge("sql_generator", "sql_executor")
    graph.add_edge("sql_executor", "summarize")

    graph.add_conditional_edges(
        "summarize", data_render_condition, {"data_render": "data_render", "data_render_apache": "data_render_apache"}
    )

    graph.add_edge("data_render", END)
    graph.add_edge("data_render_apache", END)
    graph_compiled: CompiledStateGraph = graph.compile()

    logger.info(f"excel_graph mermaid_code: {graph_compiled.get_graph().draw_mermaid()}")

    return graph_compiled
