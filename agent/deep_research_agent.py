import asyncio
import json
import logging
import os
import traceback
from typing import Optional, Literal

from deepagents import create_deep_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from tavily import TavilyClient

from common.llm_util import get_llm
from common.minio_util import MinioUtils
from constants.code_enum import DataTypeEnum, DiFyAppEnum
from services.user_service import add_user_record, decode_jwt_token

logger = logging.getLogger(__name__)

minio_utils = MinioUtils()

# 初始化Tavily客户端
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# Web search tool
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    return tavily_client.search(
        query,
        include_images=True,
        include_favicon=True,
        search_depth="advanced",
        include_image_descriptions=True,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


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

            # System prompt to steer the agent to be an expert researcher
            research_instructions = """
            
            你是一位专业的信息整合与内容撰写专家。请基于提供的深度搜索结果，撰写一篇结构清晰、内容权威、图文并茂的介绍性文章。要求如下：
            
            ## 大纲结构清晰
            
            - 使用层级标题（如 `## 一、公司概况`、`### 1.1 成立背景`）组织内容。
            - 内容应涵盖：公司基本信息、发展历程、核心技术/产品、业务布局、所获荣誉、社会影响等关键维度。
            - 若信息不足某部分，可略过，但不得虚构。
            
            ## 保留合适的图片与图标
            
            - 从搜索结果的 `images` 字段中精选 **1–3 张最具代表性** 的图片（如公司 Logo、重要合作揭牌、产品界面等）。
            - 每张图片需以标准 Markdown 格式插入：`![描述](URL)`，并附简要说明（如“中关村科金 Logo”）。
            - 在章节标题或关键要点前，适当使用 Unicode 图标增强可读性（例如：🏢 公司概况、🧠 核心技术、🌍 全球布局、🏆 所获荣誉、📈 业务影响 等）。
            
            ## 保留引用链接
            
            - 所有事实性陈述（如成立时间、融资金额、专利数量、榜单入选等）必须关联到原始来源。
            - 引用格式为 Markdown 超链接：`[来源名称](URL)`，例如：[亿欧网](https://www.iyiou.com/company/zhongguancunkejin)。
            - 避免直接复制原文长段落，应进行归纳与转述，并标注出处。
            
            ## 语言风格
            
            - 采用客观、简洁、专业的中文书面语。
            - 面向企业决策者、投资人或行业研究者，避免过度营销化表述。
            
            ## 通用性要求
            
            - 本提示词适用于任何实体（公司、人物、技术、事件等）的深度搜索结果。
            - 不依赖特定领域知识，仅基于工具返回的 `results` 和 `images` 数据生成内容。
            
            请输出纯 Markdown 格式文本，无需额外解释或包装。
            """

            agent = create_deep_agent(
                model=self.llm,
                tools=[internet_search],
                system_prompt=research_instructions,
                checkpointer=self.checkpointer,
            )

            # 如果有文件内容，则将其添加到查询中
            formatted_query = query
            async for message_chunk, metadata in agent.astream(
                input={"messages": [HumanMessage(content=formatted_query)]},
                config=config,
                stream_mode="messages",
            ):
                # print(message_chunk)

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
