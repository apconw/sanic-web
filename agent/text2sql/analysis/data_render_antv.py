import os
import logging

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from agent.text2sql.state.agent_state import AgentState

"""
AntV mcp 数据渲染节点
"""


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

    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "qwen-plus"),
        temperature=float(os.getenv("MODEL_TEMPERATURE", 0.75)),
        base_url=os.getenv("MODEL_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        api_key=os.getenv("MODEL_API_KEY"),
        # max_tokens=int(os.getenv("MAX_TOKENS", 20000)),
        top_p=float(os.getenv("TOP_P", 0.8)),
        frequency_penalty=float(os.getenv("FREQUENCY_PENALTY", 0.0)),
        presence_penalty=float(os.getenv("PRESENCE_PENALTY", 0.0)),
        timeout=float(os.getenv("REQUEST_TIMEOUT", 30.0)),
        max_retries=int(os.getenv("MAX_RETRIES", 3)),
        streaming=os.getenv("STREAMING", "True").lower() == "true",
        # 将额外参数通过 extra_body 传递
        extra_body={},
    )

    result_data = state["execution_result"]
    chart_agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=f"""
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

        ### 输入数据
        {result_data}

        ### 严格要求
        - 必须实际调用MCP工具，不能模拟或假设
        - 必须返回真实的图表链接，不能返回示例链接
        - x轴和y轴标签使用中文
        - 如果无法生成图表，请说明具体原因

        ### 返回格式
        ![图表](真实的图表链接)
        """,
    )

    result = await chart_agent.ainvoke(
        {"messages": [("user", "根据输入数据选择合适的MCP图表工具进行渲染")]},
        config={"configurable": {"thread_id": "chart-render"}},
    )

    logging.info(f"图表代理调用结果: {result}")

    state["chart_url"] = result["messages"][-1].content

    return state
