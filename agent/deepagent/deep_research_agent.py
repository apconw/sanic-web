import asyncio
import json
import logging
import os
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
        # 从 instructions.md 文件读取系统提示词
        with open(os.path.join(current_dir, "instructions.md"), "r", encoding="utf-8") as f:
            self.CORE_INSTRUCTIONS = f.read()

        # === 加载子智能体配置 ===
        # 从 subagents.json 文件读取各个子智能体的角色定义
        with open(os.path.join(current_dir, "subagents.json"), "r", encoding="utf-8") as f:
            self.subagents_config = json.load(f)

        # 提取三个子智能体的配置
        self.planner = self.subagents_config["planner"]  # 规划师
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
        运行智能体，支持多轮对话记忆
        :param query: 用户输入
        :param response: 响应对象
        :param session_id: 会话ID，用于区分同一轮对话
        :param uuid_str: 自定义ID，用于唯一标识一次问答
        :param file_list: 附件
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

            # 使用用户会话ID作为thread_id，如果未提供则使用默认值
            thread_id = session_id if session_id else "default_thread"
            config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}

            agent = create_deep_agent(
                tools=self.tools,  # 可用工具列表
                system_prompt=self.CORE_INSTRUCTIONS,  # 系统提示词
                subagents=[self.researcher, self.analyst],
                model=self.llm,
                backend=self.checkpointer,
            ).with_config({"recursion_limit": self.RECURSION_LIMIT})

            # 如果有文件内容，则将其添加到查询中
            formatted_query = query
            async for message_chunk, metadata in agent.astream(
                input={"messages": [HumanMessage(content=formatted_query)]},
                config=config,
                stream_mode="messages",
            ):
                print(message_chunk)
                # 检查是否已取消
                if self.running_tasks[task_id]["cancelled"]:
                    await response.write(
                        self._create_response("\n> 这条消息已停止", "info", DataTypeEnum.ANSWER.value[0])
                    )
                    # 发送最终停止确认消息
                    await response.write(self._create_response("", "end", DataTypeEnum.STREAM_END.value[0]))
                    break

                # 工具输出
                if metadata["langgraph_node"] == "tools":
                    tool_name = message_chunk.name or "未知工具"
                    # logger.info(f"工具调用结果:{message_chunk.content}")
                    tool_use = "> 调用工具:" + tool_name + "\n\n"
                    await response.write(self._create_response(tool_use))
                    t02_answer_data.append(tool_use)
                    continue

                # 输出最终结果
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
