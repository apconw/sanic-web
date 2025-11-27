import asyncio
import json
import logging
import os
import re
import traceback
from typing import Optional

from deepagents import create_deep_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent.deepagent.tools import search_web
from common.llm_util import get_llm
from common.minio_util import MinioUtils
from constants.code_enum import DataTypeEnum, DiFyAppEnum
from services.user_service import add_user_record, decode_jwt_token

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

        # 存储运行中的任务
        self.running_tasks = {}

        # === 配置参数 ===
        self.RECURSION_LIMIT = int(os.getenv("RECURSION_LIMIT", 25))

        # === 加载核心指令 ===
        with open(os.path.join(current_dir, "instructions.md"), "r", encoding="utf-8") as f:
            self.CORE_INSTRUCTIONS = f.read()

        # === 加载子智能体配置 ===
        with open(os.path.join(current_dir, "subagents.json"), "r", encoding="utf-8") as f:
            self.subagents_config = json.load(f)

        self.planner = self.subagents_config["planner"]  # 规划师
        self.researcher = self.subagents_config["researcher"]  # 研究员
        self.analyst = self.subagents_config["analyst"]  # 分析师

        # 定义智能体可以使用的工具
        self.tools = [search_web]

    @staticmethod
    def _create_response(
        content: str,
        message_type: str = "continue",
        data_type: str = DataTypeEnum.ANSWER.value[0],
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

            thread_id = session_id if session_id else "default_thread"
            config = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": 50,
            }

            # 发送开始消息
            start_msg = "🔍 **开始分析问题...**\n\n"
            await response.write(self._create_response(start_msg, "info"))
            t02_answer_data.append(start_msg)

            agent = create_deep_agent(
                tools=self.tools,
                system_prompt=self.CORE_INSTRUCTIONS,
                subagents=[self.researcher, self.analyst],
                model=self.llm,
                backend=self.checkpointer,
            ).with_config({"recursion_limit": self.RECURSION_LIMIT})

            formatted_query = query
            current_node = None
            step_count = 0

            async for message_chunk, metadata in agent.astream(
                input={"messages": [HumanMessage(content=formatted_query)]},
                config=config,
                stream_mode="messages",
            ):
                # 检查是否已取消
                if self.running_tasks[task_id]["cancelled"]:
                    await response.write(
                        self._create_response(
                            "\n> ⚠️ 任务已被用户取消",
                            "info",
                            DataTypeEnum.ANSWER.value[0],
                        )
                    )
                    await response.write(self._create_response("", "end", DataTypeEnum.STREAM_END.value[0]))
                    break

                node_name = metadata.get("langgraph_node", "unknown")

                # 节点切换时输出提示
                if node_name != current_node and node_name != "unknown":
                    current_node = node_name
                    step_count += 1

                    thinking_msg = ""
                    if node_name == "planner":
                        thinking_msg = f"<details>\n<summary>📋 步骤 {step_count}: 规划阶段</summary>\n\n"
                    elif node_name == "researcher":
                        thinking_msg = f"<details>\n<summary>🔎 步骤 {step_count}: 研究阶段</summary>\n\n"
                    elif node_name == "analyst":
                        thinking_msg = f"<details>\n<summary>📊 步骤 {step_count}: 分析阶段</summary>\n\n"
                    elif node_name == "tools":
                        thinking_msg = f"<details>\n<summary>🛠️ 步骤 {step_count}: 工具调用</summary>\n\n"

                    if thinking_msg:
                        await response.write(self._create_response(thinking_msg, "info"))
                        t02_answer_data.append(thinking_msg)

                # 工具调用输出
                if node_name == "tools":
                    tool_name = message_chunk.name or "未知工具"
                    if hasattr(message_chunk, "content") and message_chunk.content:
                        tool_result = f"<details>\n<summary>✅ 工具 `{tool_name}` 执行完成</summary>\n\n"
                        await response.write(self._create_response(tool_result, "info"))
                        t02_answer_data.append(tool_result)

                        try:
                            content_str = str(message_chunk.content)
                            img_urls = re.findall(
                                r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+\.(?:jpg|png|jpeg)",
                                content_str,
                            )
                            for url in img_urls[:3]:
                                image_markdown = f"[数据来源]({url})\n\n"
                                await response.write(self._create_response(image_markdown, "info"))
                                t02_answer_data.append(image_markdown)

                            result_preview = content_str[:500]
                            if len(content_str) > 500:
                                result_preview += "..."

                            preview_msg = f"\n{result_preview}\n\n</details>\n\n"
                            await response.write(self._create_response(preview_msg, "info"))
                            t02_answer_data.append(preview_msg)

                        except Exception as e:
                            preview_msg = "</details>\n\n"
                            await response.write(self._create_response(preview_msg, "info"))
                            t02_answer_data.append(preview_msg)
                    else:
                        tool_call = f"<details>\n<summary>🔧 正在调用工具: `{tool_name}`</summary>\n\n"
                        await response.write(self._create_response(tool_call, "info"))
                        t02_answer_data.append(tool_call)

                    continue

                # 输出智能体的思考和回答内容
                if message_chunk.content:
                    content = message_chunk.content
                    img_urls = re.findall(
                        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+\.(?:jpg|png|jpeg)",
                        content,
                    )
                    for url in img_urls[:3]:
                        image_markdown = f"[数据来源]({url})\n\n"
                        content += "\n\n" + image_markdown

                    t02_answer_data.append(content)
                    await response.write(self._create_response(content))

                    if hasattr(response, "flush"):
                        await response.flush()
                    await asyncio.sleep(0)

            # 发送完成消息
            if not self.running_tasks[task_id]["cancelled"]:
                completion_msg = "\n\n---\n\n✨ **报告生成完成！**\n"
                await response.write(self._create_response(completion_msg, "info"))
                t02_answer_data.append(completion_msg)

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

        except asyncio.CancelledError:
            await response.write(self._create_response("\n> ⚠️ 任务已被取消", "info", DataTypeEnum.ANSWER.value[0]))
            await response.write(self._create_response("", "end", DataTypeEnum.STREAM_END.value[0]))
        except Exception as e:
            logger.error(f"Agent运行异常: {e}")
            traceback.print_exception(e)
            error_msg = f"❌ **错误**: 智能体运行异常\n\n\n{str(e)}\n\n"
            await response.write(self._create_response(error_msg, "error", DataTypeEnum.ANSWER.value[0]))
        finally:
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
