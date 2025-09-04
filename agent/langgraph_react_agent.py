import asyncio
import json
import logging
import os
import traceback
from typing import Optional
from uuid import uuid4

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.messages.utils import trim_messages
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

from constants.code_enum import DataTypeEnum, DiFyAppEnum
from services.user_service import add_user_record, decode_jwt_token

logger = logging.getLogger(__name__)


class LangGraphReactAgent:
    """
    基于LangGraph的React智能体，支持多轮对话记忆
    """

    def __init__(self):
        # 校验并获取环境变量
        required_env_vars = [
            "MODEL_NAME",
            "MODEL_TEMPERATURE",
            "MODEL_BASE_URL",
            "MODEL_API_KEY",
            "MCP_HUB_COMMON_QA_GROUP_URL",
        ]
        for var in required_env_vars:
            if not os.getenv(var):
                raise ValueError(f"Missing required environment variable: {var}")

        self.llm = ChatOpenAI(
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

        # 使用 os.path 构建路径
        # current_dir = os.path.dirname(os.path.abspath(__file__))
        # mcp_tool_path = os.path.join(current_dir, "mcp", "query_db_tool.py")
        self.client = MultiServerMCPClient(
            {
                "mcp-hub": {
                    "url": os.getenv("MCP_HUB_COMMON_QA_GROUP_URL"),
                    "transport": "streamable_http",
                },
                # "query_qa_record": {
                #     "command": "python",
                #     "args": [mcp_tool_path],
                #     "transport": "stdio",
                # },
                # "undoom-douyin-data-analysis": {
                #     "command": "uvx",
                #     "transport": "stdio",
                #     "args": [
                #         "--index-url",
                #         "https://mirrors.aliyun.com/pypi/simple/",
                #         "--from",
                #         "undoom-douyin-data-analysis",
                #         "undoom-douyin-mcp",
                #     ],
                # },
            }
        )

        # 全局checkpointer用于持久化所有用户的对话状态
        self.checkpointer = InMemorySaver()

        # 存储运行中的任务
        self.running_tasks = {}

    @staticmethod
    def _create_response(
        content: str, message_type: str = "continue", data_type: str = DataTypeEnum.ANSWER.value[0]
    ) -> str:
        """封装响应结构"""
        res = {
            "data": {"messageType": message_type, "content": content},
            "dataType": data_type,
        }
        return "data:" + json.dumps(res, ensure_ascii=False) + "\n\n"

    @staticmethod
    def short_trim_messages(state):
        """
        模型调用前的消息清理的钩子函数
        短期记忆：限制模型调用前的消息数量，只保留最近的若干条消息
        :param state: 状态对象，包含对话消息
        :return: 修剪后的消息列表
        """
        trimmed_messages = trim_messages(
            messages=state["messages"],
            max_tokens=20000,  # 设置更合理的token限制（根据模型上下文窗口调整）
            token_counter=lambda messages: sum(len(msg.content or "") for msg in messages),  # 更准确的token计算方式
            strategy="last",  # 保留最新的消息
            allow_partial=False,
            start_on="human",  # 确保从人类消息开始
            include_system=True,  # 包含系统消息
            text_splitter=None,  # 不使用文本分割器
        )
        return {"llm_input_messages": trimmed_messages}

    async def run_agent(
        self, query: str, response, session_id: Optional[str] = None, uuid_str: str = None, user_token=None
    ):
        """
        运行智能体，支持多轮对话记忆
        :param query: 用户输入
        :param response: 响应对象
        :param session_id: 会话ID，用于区分同一轮对话
        :param uuid_str: 自定义ID，用于唯一标识一次问答
        :param user_token:
        :return:
        """

        # 获取用户信息 标识对话状态
        user_dict = await decode_jwt_token(user_token)
        task_id = user_dict["id"]
        task_context = {"cancelled": False}
        self.running_tasks[task_id] = task_context

        try:
            t02_answer_data = []

            tools = await self.client.get_tools()

            # 使用用户会话ID作为thread_id，如果未提供则使用默认值
            thread_id = session_id if session_id else "default_thread"
            config = {"configurable": {"thread_id": thread_id}}

            system_message = SystemMessage(content="""You are an advanced AI assistant""")

            agent = create_react_agent(
                model=self.llm,
                tools=tools,
                prompt=system_message,
                checkpointer=self.checkpointer,  # 使用全局checkpointer
                pre_model_hook=self.short_trim_messages,
            )

            async for message_chunk, metadata in agent.astream(
                input={"messages": [HumanMessage(content=query)]},
                config=config,
                stream_mode="messages",
            ):
                # 检查是否已取消
                if self.running_tasks[task_id]["cancelled"]:
                    await response.write(
                        self._create_response("\n> 这条消息已停止", "info", DataTypeEnum.ANSWER.value[0])
                    )
                    # 发送最终停止确认消息
                    await response.write(self._create_response("", "end", DataTypeEnum.STREAM_END.value[0]))
                    break

                # print(message_chunk)
                # 工具输出
                if metadata["langgraph_node"] == "tools":
                    tool_name = message_chunk.name or "未知工具"
                    # logger.info(f"工具调用结果:{message_chunk.content}")
                    tool_use = "> 调用工具:" + tool_name + "\n\n"
                    await response.write(self._create_response(tool_use))
                    t02_answer_data.append(tool_use)
                    continue

                # await response.write(self._create_response(agent.get_graph().draw_mermaid_png()))
                # 输出最终结果
                # print(message_chunk)
                if message_chunk.content:
                    content = message_chunk.content
                    t02_answer_data.append(content)
                    await response.write(self._create_response(content))
                    # 确保实时输出
                    if hasattr(response, "flush"):
                        await response.flush()
                    await asyncio.sleep(0)

            # 只有在未取消的情况下才保存记录
            if not self.running_tasks[task_id]["cancelled"]:
                await add_user_record(
                    uuid_str, session_id, query, t02_answer_data, {}, DiFyAppEnum.COMMON_QA.value[0], user_token
                )

        except asyncio.CancelledError:
            await response.write(self._create_response("\n> 这条消息已停止", "info", DataTypeEnum.ANSWER.value[0]))
            await response.write(self._create_response("", "end", DataTypeEnum.STREAM_END.value[0]))
        except Exception as e:
            print(f"[ERROR] Agent运行异常: {e}")
            traceback.print_exception(e)
            await response.write(
                self._create_response("[ERROR] 智能体运行异常:", "error", DataTypeEnum.ANSWER.value[0])
            )
        finally:
            # 清理任务记录
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]

    async def cancel_task(self, task_id: str) -> bool:
        """
        取消指定的任务
        :param task_id: 任务ID
        :return: 是否成功取消
        """
        if task_id in self.running_tasks:
            self.running_tasks[task_id]["cancelled"] = True
            return True
        return False

    def get_running_tasks(self):
        """
        获取当前运行中的任务列表
        :return: 运行中的任务列表
        """
        return list(self.running_tasks.keys())
