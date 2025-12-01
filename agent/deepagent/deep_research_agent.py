import asyncio
import json
import logging
import os
import re
import traceback
import ast
from typing import Optional

from deepagents import create_deep_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent.deepagent.tools import search_web
from common.llm_util import get_llm
from common.minio_util import MinioUtils
from constants.code_enum import DataTypeEnum, DiFyAppEnum
from services.user_service import add_user_record, decode_jwt_token
from langfuse import get_client
from langfuse.langchain import CallbackHandler

logger = logging.getLogger(__name__)

minio_utils = MinioUtils()

current_dir = os.path.dirname(os.path.abspath(__file__))


class DeepAgent:
    """
    基于DeepAgents的智能体，支持多轮对话记忆
    """

    def __init__(self):

        # 初始化LLM
        self.llm = get_llm()

        # 全局checkpointer用于持久化所有用户的对话状态
        self.checkpointer = InMemorySaver()

        # 是否启用链路追踪
        self.ENABLE_TRACING = os.getenv("LANGFUSE_TRACING_ENABLED", "true").lower() == "true"

        # 存储运行中的任务
        self.running_tasks = {}

        # === 配置参数 ===
        self.RECURSION_LIMIT = int(os.getenv("RECURSION_LIMIT", 25))

        # === 加载核心指令 ===
        # 从 instructions.md 文件读取系统提示词
        with open(os.path.join(current_dir, "instructions.md"), "r", encoding="utf-8") as f:
            self.CORE_INSTRUCTIONS = f.read()

        # === 加载子智能体配置 ===
        # 从 subagents.json 文件读取各个子智能体的角色定义
        with open(os.path.join(current_dir, "subagents.json"), "r", encoding="utf-8") as f:
            self.subagents_config = json.load(f)

        # 提取三个子智能体的配置
        self.planner = self.subagents_config["planner"]  # 规划师 使用默认的
        self.researcher = self.subagents_config["researcher"]  # 研究员
        self.analyst = self.subagents_config["analyst"]  # 分析师

        # 定义智能体可以使用的工具
        self.tools = [search_web]

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

    async def run_agent(
        self,
        query: str,
        response,
        session_id: Optional[str] = None,
        uuid_str: str = None,
        user_token=None,
        file_list: dict = None,
    ):
        """
        运行智能体，支持多轮对话记忆和实时思考过程输出
        :param query: 用户输入
        :param response: 响应对象
        :param session_id: 会话ID，用于区分同一轮对话
        :param uuid_str: 自定义ID，用于唯一标识一次问答
        :param file_list: 附件
        :param user_token: 用户令牌
        :return:
        """
        # 获取用户信息 标识对话状态
        user_dict = await decode_jwt_token(user_token)
        task_id = user_dict["id"]
        task_context = {"cancelled": False}
        self.running_tasks[task_id] = task_context

        try:
            t02_answer_data = []

            # 使用用户会话ID作为thread_id，如果未提供则使用默认值
            thread_id = session_id if session_id else "default_thread"
            config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}

            # 准备 tracing 配置
            if self.ENABLE_TRACING:
                langfuse_handler = CallbackHandler()
                callbacks = [langfuse_handler]
                config["callbacks"] = callbacks
                config["metadata"] = {"langfuse_session_id": session_id}

            # 发送开始消息
            start_msg = "🔍 **开始分析问题...**\n\n"
            await response.write(self._create_response(start_msg, "info"))
            t02_answer_data.append(start_msg)

            agent = create_deep_agent(
                tools=self.tools,  # 可用工具列表
                system_prompt=self.CORE_INSTRUCTIONS,  # 系统提示词
                subagents=[self.planner, self.researcher, self.analyst],
                model=self.llm,
                backend=self.checkpointer,
            ).with_config({"recursion_limit": self.RECURSION_LIMIT})

            # 如果有文件内容，则将其添加到查询中
            formatted_query = query

            # 准备流式处理参数
            stream_args = {
                "input": {"messages": [HumanMessage(content=formatted_query)]},
                "config": config,
                "stream_mode": "messages",
            }

            # 如果启用 tracing，包裹在 trace 上下文中
            if self.ENABLE_TRACING:
                langfuse = get_client()
                with langfuse.start_as_current_observation(
                    input=query,
                    as_type="agent",
                    name="深度搜索",
                ) as rootspan:
                    user_info = await decode_jwt_token(user_token)
                    user_id = user_info.get("id")
                    rootspan.update_trace(session_id=session_id, user_id=user_id)
                    await self._stream_agent_response(
                        agent,
                        stream_args,
                        response,
                        task_id,
                        t02_answer_data,
                        uuid_str,
                        session_id,
                        query,
                        file_list,
                        user_token,
                    )
            else:
                await self._stream_agent_response(
                    agent,
                    stream_args,
                    response,
                    task_id,
                    t02_answer_data,
                    uuid_str,
                    session_id,
                    query,
                    file_list,
                    user_token,
                )

        except asyncio.CancelledError:
            await response.write(self._create_response("\n> ⚠️ 任务已被取消", "info", DataTypeEnum.ANSWER.value[0]))
            await response.write(self._create_response("", "end", DataTypeEnum.STREAM_END.value[0]))
        except Exception as e:
            logger.error(f"Agent运行异常: {e}")
            traceback.print_exception(e)
            error_msg = f"❌ **错误**: 智能体运行异常\n\n```\n{str(e)}\n```\n"
            await response.write(self._create_response(error_msg, "error", DataTypeEnum.ANSWER.value[0]))
        finally:
            # 清理任务记录
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]

    async def _stream_agent_response(
        self, agent, stream_args, response, task_id, t02_answer_data, uuid_str, session_id, query, file_list, user_token
    ):
        """处理agent流式响应的核心逻辑"""
        plan_task_mesage = False
        async for message_chunk, metadata in agent.astream(**stream_args):
            # print(metadata)
            # print(message_chunk)
            # 检查是否已取消
            if self.running_tasks[task_id]["cancelled"]:
                await response.write(
                    self._create_response("\n> ⚠️ 任务已被用户取消", "info", DataTypeEnum.ANSWER.value[0])
                )
                await response.write(self._create_response("", "end", DataTypeEnum.STREAM_END.value[0]))
                break

            # 获取当前节点信息
            node_name = metadata.get("langgraph_node", "unknown")

            # 工具调用输出
            if node_name == "tools":
                tool_name = message_chunk.name or "未知工具"

                if tool_name == "write_todos":
                    if not plan_task_mesage:
                        plan_markdown_str, plan_markdown_list = self.extract_content_as_markdown_list(
                            message_chunk.content
                        )
                        think_html = f"""<details style="color:gray;background-color: #f8f8f8;padding: 2px;border-radius: 
                                          6px;margin-top:5px;">
                                                <summary>{query}-任务规划如下:\n</summary>"""
                        think_html += f"""{plan_markdown_str}"""
                        think_html += """</details>\n\n"""
                        await response.write(self._create_response(think_html, "info"))
                        t02_answer_data.append(think_html)
                        plan_task_mesage = True

                if tool_name == "search_web":
                    search_content = message_chunk.content
                    content_json = json.loads(search_content)
                    think_html = f"""\n > ✅ 搜索{content_json["query"]}\n\n"""
                    await response.write(self._create_response(think_html, "info"))
                    t02_answer_data.append(think_html)

                continue

            # 输出智能体的思考和回答内容
            if message_chunk.content:
                content = message_chunk.content
                t02_answer_data.append(content)
                await response.write(self._create_response(content))

                # 确保实时输出
                if hasattr(response, "flush"):
                    await response.flush()
                await asyncio.sleep(0)

        # 发送完成消息
        if not self.running_tasks[task_id]["cancelled"]:
            completion_msg = "\n\n---\n\n✨ **报告生成完成！**\n"
            await response.write(self._create_response(completion_msg, "info"))
            t02_answer_data.append(completion_msg)

            # 保存记录
            await add_user_record(
                uuid_str,
                session_id,
                query,
                t02_answer_data,
                {},
                DiFyAppEnum.REPORT_QA.value[0],
                user_token,
                file_list,
            )

    @staticmethod
    def extract_content_as_markdown_list(text: str) -> tuple[Optional[str], list]:
        """
        安全地从类 JSON 字符串中提取 content 字段，生成纯 Markdown 列表。
        使用 JSON 解析（而非 ast.literal_eval），更符合安全规范。
        """
        # 1. 提取 [...] 区域
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return None
        raw_list_str = match.group(0).strip()
        try:
            json_str = raw_list_str.replace("'", '"').replace('""', '"')

            todo_list = json.loads(json_str)

        except (json.JSONDecodeError, ValueError):
            return None

        if not isinstance(todo_list, list):
            return None

        lines = []
        for index, item in enumerate(todo_list, 1):  # 从1开始编号
            if isinstance(item, dict) and isinstance(item.get("content"), str):
                if index == 1:
                    lines.append(f"  {index}. {item['content']}")  # 在编号前添加空格
                else:
                    lines.append(f" {index}. {item['content']}")  # 在编号前添加空格

        return "\n\n".join(lines), lines

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
