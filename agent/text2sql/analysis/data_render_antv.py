import logging
import os

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver

from agent.middleware.customer_middleware import modify_args
from agent.text2sql.state.agent_state import AgentState
from common.llm_util import get_llm

"""
AntV mcp 数据渲染节点
"""

memory = InMemorySaver()


async def data_render_ant(state: AgentState):
    """
    蚂蚁antV数据图表渲染
    mcphub 按group指定查询AntV-chart工具 减少上下文token
    :return:
    """
    client = MultiServerMCPClient(
        {
            "mcphub-sse": {
                "url": os.getenv("MCP_HUB_DATABASE_QA_GROUP_URL"),
                "transport": "streamable_http",
            }
        }
    )

    # 获取上一个步骤目标工具 过滤工具集减少token和大模型幻觉问题
    chart_type = state["chart_type"]
    tools = await client.get_tools()
    tools = [tool for tool in tools if tool.name == chart_type]
    llm = get_llm()

    result_data = state["execution_result"]
    chart_agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=f"""
        你是一位经验丰富的BI专家，必须严格按照以下步骤操作，并且必须调用MCP图表工具：

        ### 重要说明
        - 你必须调用可用的MCP图表工具来生成图表，这是强制要求
        - 不允许返回默认示例链接或虚构链接
        - 如果工具调用失败，请明确说明失败原因

        ### 任务步骤
        1. **分析数据特征**：仔细理解输入数据结构
        2. **调用工具**：必须使用"{chart_type}"工具进行图表渲染
        3. **填充参数**：根据数据特征填充图表参数
        4. **生成图表**：调用MCP工具并等待真实响应
        5. **返回结果**：只返回真实的图表链接

        ### 严格要求
        - 必须实际调用MCP工具，不能模拟或假设
        - 必须返回真实的图表链接，不能返回示例链接
        - x轴和y轴标签使用中文
        - 如果无法生成图表，请说明具体原因
        - 工具调用成功后，返回真实的图表链接，格式如下： "![图表名称](真实的图表链接)"
        
        请注意，你必须严格遵守这些要求，否则你的回答将被视为无效。
        """,
        middleware=[modify_args],
    )

    result = await chart_agent.ainvoke(
        {"messages": [("user", f"输入数据如下:\n<data>{result_data}</data>")]},
        config={"configurable": {"thread_id": "chart-render"}},
    )

    logging.info(f"图表代理调用结果: {result}")

    state["chart_url"] = result["messages"][-1].content

    return state
