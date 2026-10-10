"""
Deep Research Agent - 基于 DeepAgents 的多阶段 Text-to-SQL 智能体

架构特性：
1. 保留 create_deep_agent 核心架构
2. 多阶段执行：思考规划 → 执行 → 回答/报告
3. 实时 SSE 流推送，思考过程用 <details> 包裹，内容直接输出
4. 多智能体协作可见性（子代理活动带标签展示）
5. 不保存对话历史记录
"""

import asyncio
import json
import logging
import os
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.deepagent.tools.native_sql_tools import (
    set_native_datasource_info,
    sql_db_query,
    sql_db_query_checker,
    sql_db_schema,
    sql_db_smart_search,
    sql_db_table_relationship,
)
from agent.deepagent.tools.youcom_search_tool import youcom_search
from agent.deepagent.tools.tool_call_manager import get_tool_call_manager
from services.skill_service import SkillService
from common.datasource_util import (
    DB,
    ConnectType,
    DatasourceConfigUtil,
    DatasourceConnectionUtil,
)
from common.llm_util import get_llm
from constants.code_enum import DataTypeEnum, IntentEnum
from model.db_connection_pool import get_db_pool
from services.datasource_service import DatasourceService
from services.user_service import add_user_record, decode_jwt_token

logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))


# ==================== 阶段枚举与追踪 ====================


class Phase(Enum):
    """Agent 执行阶段"""

    PLANNING = "planning"  # 思考规划（首次工具调用前的输出）
    EXECUTION = "execution"  # 执行回答（默认阶段）
    SUB_AGENT = "sub_agent"  # 子代理运行中
    REPORTING = "reporting"  # 报告生成（HTML 标记透传）


@dataclass
class PhaseTracker:
    """
    阶段追踪器：管理 <details> 标签的开关状态

    核心职责：
    - 追踪当前执行阶段
    - 管理 <details> 区域的打开/关闭
    - 判断是否已进入正式内容阶段
    """

    current_phase: Phase = Phase.PLANNING  # 初始为规划阶段
    planning_opened: bool = False  # 思考区 <details> 是否已打开
    subagent_opened: bool = False  # 子代理区 <details> 是否已打开
    current_node: str = ""  # 当前 langgraph 节点名
    has_tool_called: bool = False  # 是否已发生过工具调用
    has_sent_content: bool = False  # 是否已输出过正式内容


# ==================== 子代理标签映射 ====================

SUB_AGENT_LABELS = {
    "task": "子代理任务",
    "report-generation": "报告生成",
    "query-writing": "SQL 查询",
    "schema-exploration": "架构探索",
}

# ==================== <details> 标签模板 ====================

THINKING_SECTION_OPEN = (
    '<details style="margin:8px 0;padding:8px 12px;background:#f8f9fa;'
    "border-left:3px solid #4a90d9;border-radius:4px;font-size:14px;color:#555"
    '">\n'
    '<summary style="cursor:pointer;font-weight:600;color:#333">'
    "🧠 思考与规划</summary>\n\n"
)

SUBAGENT_SECTION_OPEN_TPL = (
    '<details style="margin:8px 0;padding:8px 12px;background:#fff8e6;'
    "border-left:3px solid #f0a020;border-radius:4px;font-size:14px;color:#555"
    '">\n'
    '<summary style="cursor:pointer;font-weight:600;color:#333">'
    "🤖 {label}</summary>\n\n"
)

SECTION_CLOSE = "\n</details>\n\n"


# ==================== DeepAgent 主类 ====================


class DeepAgent:
    """基于 DeepAgents 的多阶段 Text-to-SQL 智能体"""

    # 递归限制：子代理也消耗递归次数，150 足够完成复杂任务同时防止无限循环
    DEFAULT_RECURSION_LIMIT = 150

    # LLM 单次请求超时（秒）- 公网大模型高峰期或生成长报告时需要较长时间
    DEFAULT_LLM_TIMEOUT = 15 * 60

    # 单次 LLM 输出 token 上限 - 报告生成需要大量 token
    # 复杂 HTML 报告（含 ECharts、CSS、数据表格）可能需要数万 token
    # DeepSeek 等模型默认上限较低(4096-8192)，需显式设置更高值
    DEFAULT_LLM_MAX_TOKENS = 65536

    # SSE 保活间隔（秒）：防止代理/浏览器约 2 分钟无数据断开
    STREAM_KEEPALIVE_INTERVAL = 25

    # 总任务超时（秒）- 复杂报告生成可能需要较长时间
    # 需与前端 fetch timeout 和 Nginx proxy_read_timeout 对齐
    TASK_TIMEOUT = 30 * 60

    def __init__(self):
        self.tool_manager = get_tool_call_manager()
        self.available_skills = self._load_available_skills()
        # 存储运行中的任务：task_id -> {"cancelled": bool, "session_id": str}
        self.running_tasks = {}

        # 从环境变量读取配置
        self.RECURSION_LIMIT = int(os.getenv("RECURSION_LIMIT", self.DEFAULT_RECURSION_LIMIT))
        self.LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", self.DEFAULT_LLM_TIMEOUT))
        self.LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", self.DEFAULT_LLM_MAX_TOKENS))

        # 链路追踪配置
        self.ENABLE_TRACING = os.getenv("LANGFUSE_TRACING_ENABLED", "false").lower() == "true"

    # ==================== 技能加载 ====================

    def _load_available_skills(self):
        """加载所有可用的技能"""
        from services.skill_service import SkillService

        return SkillService.list_skills(scope="deep")

    def get_available_skills(self):
        """获取所有可用的技能列表"""
        return self.available_skills

    # ==================== SSE 响应工具方法 ====================

    @staticmethod
    def _create_response(
        content: str,
        message_type: str = "continue",
        data_type: str = DataTypeEnum.ANSWER.value[0],
    ) -> str:
        """封装 SSE 响应结构"""
        res = {
            "data": {"messageType": message_type, "content": content},
            "dataType": data_type,
        }
        return "data:" + json.dumps(res, ensure_ascii=False) + "\n\n"

    async def _safe_write(
        self,
        response,
        content: str,
        message_type: str = "continue",
        data_type: str = None,
    ) -> bool:
        """安全地写入 SSE 响应，连接断开时返回 False"""
        try:
            if data_type is None:
                data_type = DataTypeEnum.ANSWER.value[0]
            await response.write(self._create_response(content, message_type, data_type))
            if hasattr(response, "flush"):
                await response.flush()
            return True
        except Exception as e:
            if self._is_connection_error(e):
                logger.info(f"客户端连接已断开: {type(e).__name__}")
                return False
            raise

    @staticmethod
    def _is_connection_error(exception: Exception) -> bool:
        """判断是否是连接断开相关的异常"""
        error_type = type(exception).__name__
        error_msg = str(exception).lower()

        connection_error_types = [
            "ConnectionClosed",
            "ConnectionResetError",
            "BrokenPipeError",
            "ConnectionError",
            "OSError",
        ]
        connection_error_keywords = [
            "connection closed",
            "connection reset",
            "broken pipe",
            "client disconnected",
            "connection aborted",
            "transport closed",
        ]

        if error_type in connection_error_types:
            return True
        for keyword in connection_error_keywords:
            if keyword in error_msg:
                return True
        return False

    # ==================== 格式化方法 ====================

    @staticmethod
    def _format_tool_call(name: str, args: dict) -> Optional[str]:
        """格式化工具调用信息（紧凑格式）"""
        if name == "sql_db_query":
            query = args.get("query", "")
            return f"\n```sql\n{query.strip()}\n```\n"
        elif name == "sql_db_schema":
            table_names = args.get("table_names", "")
            if isinstance(table_names, list):
                table_names = ", ".join(table_names)
            return f"- 查看表结构: `{table_names}`\n" if table_names else "- 查看表结构\n"
        elif name == "sql_db_smart_search":
            query_preview = args.get("user_query", "")[:50]
            return f"- 🔍 智能表检索: `{query_preview}`\n"
        elif name == "sql_db_list_tables":
            return "- 获取表列表\n"
        elif name == "sql_db_query_checker":
            return "- 校验 SQL\n"
        elif name == "sql_db_table_relationship":
            table_names = args.get("table_names", "")
            return f"- 查看表关系: `{table_names}`\n"
        return None

    @staticmethod
    def _format_tool_result(name: str, content: str) -> Optional[str]:
        """格式化工具执行结果（紧凑格式）"""
        if "sql" in name.lower():
            if "error" not in content.lower():
                return "  ✓ 成功\n"
            else:
                return f"  ✗ 失败: {content[:200].strip()}\n"
        return None

    @staticmethod
    def _extract_text(content) -> str:
        """从消息内容中提取文本（兼容字符串和列表格式）"""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    parts.append(part)
            return "".join(parts)
        return str(content) if content else ""

    # ==================== 阶段检测与 <details> 管理 ====================

    @staticmethod
    def _detect_phase(node_name: str, content: str, tracker: PhaseTracker) -> Phase:
        """
        基于 langgraph_node 元数据和内容检测当前阶段

        规则（优先级从高到低）：
        1. node_name 包含 "task" → SUB_AGENT
        2. content 包含 REPORT_HTML 标记 → REPORTING
        3. 尚未发生工具调用 → PLANNING（首次思考阶段）
        4. 默认 → EXECUTION
        """
        # 子代理检测
        if "task" in node_name.lower():
            return Phase.SUB_AGENT

        # 报告标记检测
        if "REPORT_HTML_START" in content or "REPORT_HTML_END" in content:
            return Phase.REPORTING

        # 首次工具调用前的输出视为思考规划
        if not tracker.has_tool_called:
            return Phase.PLANNING

        # 默认执行阶段
        return Phase.EXECUTION

    async def _open_thinking_section(self, response) -> bool:
        """打开思考规划 <details> 区域"""
        return await self._safe_write(response, THINKING_SECTION_OPEN)

    async def _open_subagent_section(self, response, node_name: str) -> bool:
        """打开子代理 <details> 区域"""
        label = SUB_AGENT_LABELS.get(node_name, f"子代理: {node_name}")
        html = SUBAGENT_SECTION_OPEN_TPL.format(label=label)
        return await self._safe_write(response, html)

    async def _close_sections(self, response, tracker: PhaseTracker) -> bool:
        """关闭所有已打开的 <details> 区域"""
        if tracker.planning_opened:
            if not await self._safe_write(response, SECTION_CLOSE):
                return False
            tracker.planning_opened = False
        if tracker.subagent_opened:
            if not await self._safe_write(response, SECTION_CLOSE):
                return False
            tracker.subagent_opened = False
        return True

    # ==================== Agent 创建 ====================

    def _create_sql_deep_agent(self, datasource_id: int, session_id: str):
        """
        创建 text-to-SQL Deep Agent，支持所有数据源类型

        Args:
            datasource_id: 数据源 ID
            session_id: 会话 ID，用于工具调用管理
        """
        logger.info(f"创建 Deep Agent - 数据源: {datasource_id}, 会话: {session_id}")

        db_pool = get_db_pool()
        with db_pool.get_session() as session:
            datasource = DatasourceService.get_datasource_by_id(session, datasource_id)
            if not datasource:
                raise ValueError(f"数据源 {datasource_id} 不存在")

            db_enum = DB.get_db(datasource.type, default_if_none=True)

            model = get_llm(timeout=self.LLM_TIMEOUT, max_tokens=self.LLM_MAX_TOKENS)
            logger.info(f"LLM 模型已创建，超时: {self.LLM_TIMEOUT}秒，" f"递归限制: {self.RECURSION_LIMIT}")

            if db_enum.connect_type == ConnectType.sqlalchemy:
                logger.info(f"数据源 {datasource_id} ({datasource.type}) 使用 SQLAlchemy 连接")
                config = DatasourceConfigUtil.decrypt_config(datasource.configuration)
                uri = DatasourceConnectionUtil.build_connection_uri(datasource.type, config)
                db = SQLDatabase.from_uri(uri, sample_rows_in_table_info=3)
                toolkit = SQLDatabaseToolkit(db=db, llm=model)
                # 设置 datasource_id，使 sql_db_smart_search 能读取元数据
                set_native_datasource_info(
                    datasource_id,
                    datasource.type,
                    datasource.configuration,
                    session_id,
                )
                # 过滤掉 sql_db_list_tables，由 sql_db_smart_search 替代
                toolkit_tools = [t for t in toolkit.get_tools() if t.name != "sql_db_list_tables"]
                sql_tools = [sql_db_smart_search, sql_db_table_relationship, youcom_search] + toolkit_tools
                logger.info(f"SQLAlchemy 工具列表: {[t.name for t in sql_tools]}")
            else:
                logger.info(f"数据源 {datasource_id} ({datasource.type}) 使用原生驱动连接")
                set_native_datasource_info(
                    datasource_id,
                    datasource.type,
                    datasource.configuration,
                    session_id,
                )
                sql_tools = [
                    sql_db_smart_search,
                    sql_db_schema,
                    sql_db_query,
                    sql_db_query_checker,
                    sql_db_table_relationship,
                    youcom_search,
                ]

        # 获取启用的 deep skill 路径
        skill_paths = [os.path.join(current_dir, "skills")]  # SkillService.get_enabled_skill_paths(scope="deep")

        # 注入当前日期，让 LLM 知道当前时间
        current_date = datetime.now().strftime("%Y-%m-%d")
        memory = [os.path.join(current_dir, "AGENTS.md"), f"当前日期: {current_date}"]

        agent = create_deep_agent(
            model=model,
            system_prompt=(
                "CRITICAL RULE: When working with a database, you MUST call "
                "`sql_db_smart_search` as your very first tool call — before "
                "sql_db_list_tables, sql_db_schema, or any other SQL tool. "
                "Pass the user's original question as `user_query`. "
                "Never skip this step."
            ),
            memory=memory,
            skills=skill_paths if skill_paths else None,
            tools=sql_tools,
            backend=FilesystemBackend(root_dir=current_dir),
        )
        return agent

    # ==================== 核心执行 ====================

    async def run_agent(
        self,
        query: str,
        response,
        session_id: Optional[str] = None,
        uuid_str: str = None,
        user_token=None,
        file_list: dict = None,
        datasource_id: int = None,
    ):
        """
        运行智能体，多阶段实时流推送

        Args:
            query: 用户输入
            response: SSE 响应对象
            session_id: 会话ID
            uuid_str: 唯一标识（兼容参数）
            user_token: 用户令牌
            file_list: 附件（兼容参数）
            datasource_id: 数据源ID
        """
        if not datasource_id:
            await self._safe_write(
                response,
                "❌ **错误**: 必须提供数据源ID (datasource_id)",
                "error",
                DataTypeEnum.ANSWER.value[0],
            )
            return

        # 获取用户信息，生成会话标识
        user_dict = await decode_jwt_token(user_token)
        task_id = user_dict["id"]
        effective_session_id = session_id or f"sql-agent-{datasource_id}-{task_id}"

        # 重置工具调用状态
        self.tool_manager.reset_session(effective_session_id)

        # 注册任务用于取消跟踪
        self.running_tasks[task_id] = {"cancelled": False, "session_id": effective_session_id}

        start_time = time.time()
        connection_closed = False
        # 收集所有输出内容，流结束后写入 t_user_qa_record
        answer_collector: list[str] = []

        try:
            agent = self._create_sql_deep_agent(datasource_id, effective_session_id)

            config = {
                "configurable": {"thread_id": effective_session_id},
                "recursion_limit": self.RECURSION_LIMIT,
            }

            # 如果启用追踪，添加 Langfuse Callback
            if self.ENABLE_TRACING:
                from langfuse.langchain import CallbackHandler

                config["callbacks"] = [CallbackHandler()]
                config["metadata"] = {"langfuse_session_id": session_id}

            try:
                # 根据是否启用追踪，选择执行方式
                if self.ENABLE_TRACING:
                    from langfuse import get_client

                    langfuse = get_client()
                    with langfuse.start_as_current_observation(
                        input=query,
                        as_type="agent",
                        name="Deep Research Agent (SQL)",
                    ) as rootspan:
                        rootspan.update_trace(session_id=session_id, user_id=str(task_id))
                        connection_closed = await asyncio.wait_for(
                            self._stream_response(
                                agent,
                                config,
                                query,
                                response,
                                effective_session_id,
                                answer_collector,
                                task_id,
                            ),
                            timeout=self.TASK_TIMEOUT,
                        )
                else:
                    connection_closed = await asyncio.wait_for(
                        self._stream_response(
                            agent,
                            config,
                            query,
                            response,
                            effective_session_id,
                            answer_collector,
                            task_id,
                        ),
                        timeout=self.TASK_TIMEOUT,
                    )
            except asyncio.TimeoutError:
                elapsed = time.time() - start_time
                logger.error(f"任务总超时 ({self.TASK_TIMEOUT}秒) - 实际耗时: {elapsed:.0f}秒")
                await self._safe_write(
                    response,
                    f"\n> ⚠️ **执行超时**: 任务执行时间超过上限"
                    f"（{self.TASK_TIMEOUT // 60} 分钟），请简化查询后重试。",
                    "error",
                    DataTypeEnum.ANSWER.value[0],
                )

        except asyncio.CancelledError:
            logger.info(f"任务被取消 - 会话: {effective_session_id}")
            connection_closed = True
            raise
        except Exception as e:
            if self._is_connection_error(e):
                logger.info(f"客户端连接已断开: {type(e).__name__}")
                connection_closed = True
            else:
                logger.error(f"Agent运行异常: {e}")
                traceback.print_exception(e)
                try:
                    await self._safe_write(
                        response,
                        f"❌ **错误**: 智能体运行异常\n\n```\n{str(e)[:200]}\n```\n",
                        "error",
                        DataTypeEnum.ANSWER.value[0],
                    )
                except Exception:
                    pass
        finally:
            # 写入对话记录到 t_user_qa_record
            try:
                if answer_collector:
                    record_id = await add_user_record(
                        uuid_str=uuid_str or "",
                        chat_id=session_id,
                        question=query,
                        to2_answer=answer_collector,
                        to4_answer={},
                        qa_type=IntentEnum.REPORT_QA.value[0],
                        user_token=user_token,
                        file_list=file_list,
                        datasource_id=datasource_id,
                    )
                    logger.info(
                        f"对话记录已保存 - record_id: {record_id}, "
                        f"会话: {effective_session_id}, 内容长度: {sum(len(s) for s in answer_collector)}"
                    )
                    # # 发送 record_id 到前端
                    # if record_id and not connection_closed:
                    #     await self._safe_write(
                    #         response,
                    #         json.dumps({"record_id": record_id}),
                    #         "continue",
                    #         DataTypeEnum.RECORD_ID.value[0],
                    #     )
            except Exception as e:
                logger.error(f"保存对话记录失败: {e}", exc_info=True)

            # 发送流结束标记
            if not connection_closed:
                try:
                    await self._safe_write(response, "", "end", DataTypeEnum.STREAM_END.value[0])
                except Exception as e:
                    logger.warning(f"发送 STREAM_END 失败: {e}")

            elapsed = time.time() - start_time
            stats = self.tool_manager.get_stats(effective_session_id)
            logger.info(f"任务结束 - 会话: {effective_session_id}, " f"耗时: {elapsed:.2f}秒, 工具调用统计: {stats}")

            # 清理任务记录
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]

    # ==================== 核心流处理 ====================

    async def _stream_response(
        self,
        agent,
        config: dict,
        query: str,
        response,
        session_id: str,
        answer_collector: list,
        task_id: str = None,
    ) -> bool:
        """
        处理 agent 流式响应，多阶段实时推送到前端

        执行阶段流转：
        PLANNING（思考规划，<details> 包裹）
            ↓ 首次工具调用
        EXECUTION（执行回答，直接输出）
            ↕ 子代理触发
        SUB_AGENT（子代理，带标签 <details>）
            ↓ 完成
        EXECUTION / REPORTING（回答或报告输出）

        Args:
            answer_collector: 收集所有输出内容的列表，流结束后用于写入数据库

        Returns:
            bool: 连接是否已断开（True=断开）
        """
        tracker = PhaseTracker()
        token_count = 0
        connection_closed = False

        logger.info(f"开始流式响应 - 会话: {session_id}, 查询: {query[:100]}")

        stream_iter = agent.astream(
            input={"messages": [HumanMessage(content=query)]},
            config=config,
            stream_mode=["messages", "updates"],
        )
        stream_anext = stream_iter.__anext__

        try:
            while True:
                # ---- 1. 等待下一 chunk（带 keepalive 超时）----
                try:
                    mode, chunk = await asyncio.wait_for(stream_anext(), timeout=self.STREAM_KEEPALIVE_INTERVAL)
                except asyncio.TimeoutError:
                    try:
                        await response.write(
                            'data: {"data":{"messageType": "info", "content": ""}, ' '"dataType": "keepalive"}\n\n'
                        )
                        if hasattr(response, "flush"):
                            await response.flush()
                    except Exception as e:
                        if self._is_connection_error(e):
                            connection_closed = True
                            break
                        raise
                    continue
                except StopAsyncIteration:
                    break

                # ---- 2. 检查工具调用管理器终止 ----
                ctx = self.tool_manager.get_session(session_id)
                if ctx.should_terminate:
                    logger.warning(f"工具调用管理器触发终止: {ctx.termination_reason}")
                    # 先关闭所有 <details> 区域
                    await self._close_sections(response, tracker)
                    await self._safe_write(
                        response,
                        f"\n> ⚠️ **执行中止**\n\n{ctx.termination_reason}",
                        "warning",
                        DataTypeEnum.ANSWER.value[0],
                    )
                    break

                # ---- 2.1 检查任务取消标记 ----
                if task_id and task_id in self.running_tasks and self.running_tasks[task_id].get("cancelled"):
                    logger.info(f"任务被取消 - task_id: {task_id}")
                    await self._close_sections(response, tracker)
                    await self._safe_write(
                        response,
                        "\n> 这条消息已停止",
                        "info",
                        DataTypeEnum.ANSWER.value[0],
                    )
                    await self._safe_write(response, "", "end", DataTypeEnum.STREAM_END.value[0])
                    break

                # ---- 3. messages 模式：token 级实时流式输出 ----
                if mode == "messages":
                    message_chunk, metadata = chunk
                    node_name = metadata.get("langgraph_node", "")

                    # print(f"node_name: {message_chunk}, metadata: {metadata}")

                    # 跳过工具节点（工具结果通过 updates 模式处理）
                    if node_name == "tools":
                        continue

                    if not (hasattr(message_chunk, "content") and message_chunk.content):
                        continue

                    token_text = self._extract_text(message_chunk.content)
                    if not token_text:
                        continue

                    # 阶段检测
                    new_phase = self._detect_phase(node_name, token_text, tracker)
                    tracker.current_node = node_name

                    # 阶段切换处理
                    if new_phase != tracker.current_phase:
                        closed = await self._handle_phase_transition(response, tracker, new_phase, node_name)
                        if not closed:
                            connection_closed = True
                            break

                    # 输出 token 并收集到 answer_collector
                    if not await self._safe_write(response, token_text):
                        connection_closed = True
                        break

                    answer_collector.append(token_text)
                    token_count += 1

                    # 刷新策略：每 10 token 或遇到 HTML 报告标记时刷新
                    if token_count % 10 == 0 or "REPORT_HTML_" in token_text:
                        if hasattr(response, "flush"):
                            try:
                                await response.flush()
                            except Exception as e:
                                if self._is_connection_error(e):
                                    connection_closed = True
                                    break
                                raise

                    await asyncio.sleep(0)

                # ---- 4. updates 模式：工具调用与结果 ----
                elif mode == "updates":
                    # 标记已发生工具调用（不提前关闭 <details>，让工具调用保持在当前区域内）
                    # 阶段切换交由 messages 模式中的 _detect_phase 自然触发
                    if not tracker.has_tool_called:
                        tracker.has_tool_called = True

                    for node_name, node_output in chunk.items():
                        if connection_closed:
                            break
                        if not isinstance(node_output, dict) or "messages" not in node_output:
                            continue

                        messages = node_output["messages"]
                        if not isinstance(messages, list):
                            messages = [messages]

                        for msg in messages:
                            if not await self._process_update_message(msg, response, answer_collector):
                                connection_closed = True
                                break

                    if connection_closed:
                        break

                    if hasattr(response, "flush"):
                        try:
                            await response.flush()
                        except Exception as e:
                            if self._is_connection_error(e):
                                connection_closed = True
                                break
                            raise
                    await asyncio.sleep(0)

        except asyncio.CancelledError:
            logger.info(f"流被取消 - 会话: {session_id}")
            connection_closed = True
            raise
        except Exception as e:
            if self._is_connection_error(e):
                logger.info(f"客户端连接已断开: {type(e).__name__}")
                connection_closed = True
            else:
                logger.error(f"流式响应异常: {type(e).__name__}: {e}", exc_info=True)
                try:
                    # 先关闭打开的 <details>
                    await self._close_sections(response, tracker)
                    await self._safe_write(
                        response,
                        f"\n> ❌ **处理异常**: {str(e)[:200]}\n\n请稍后重试。",
                        "error",
                        DataTypeEnum.ANSWER.value[0],
                    )
                except Exception:
                    pass
        finally:
            # 确保关闭所有打开的 <details> 区域
            if not connection_closed:
                try:
                    await self._close_sections(response, tracker)
                except Exception:
                    pass

            # 检测 HTML 报告是否被截断
            if not connection_closed and answer_collector:
                try:
                    full_output = "".join(answer_collector)
                    if "REPORT_HTML_START" in full_output and "REPORT_HTML_END" not in full_output:
                        logger.warning(f"HTML 报告被截断 - 会话: {session_id}, " f"输出长度: {len(full_output)}")
                        truncation_msg = (
                            "\n\n> ⚠️ **报告生成不完整**: HTML 报告在生成过程中被截断。"
                            "可能原因：模型输出 token 达到上限。请尝试简化报告需求后重试。\n"
                            "<!-- REPORT_HTML_END -->\n"
                        )
                        await self._safe_write(
                            response,
                            truncation_msg,
                            "warning",
                            DataTypeEnum.ANSWER.value[0],
                        )
                        answer_collector.append(truncation_msg)
                except Exception:
                    pass

        logger.info(
            f"流式响应结束 - 会话: {session_id}, " f"token数: {token_count}, 阶段: {tracker.current_phase.value}"
        )
        return connection_closed

    async def _handle_phase_transition(
        self,
        response,
        tracker: PhaseTracker,
        new_phase: Phase,
        node_name: str,
    ) -> bool:
        """
        处理阶段切换，管理 <details> 标签的打开/关闭

        Returns:
            bool: True=成功, False=连接断开
        """
        old_phase = tracker.current_phase

        if new_phase == Phase.PLANNING:
            if not tracker.planning_opened:
                if not await self._open_thinking_section(response):
                    return False
                tracker.planning_opened = True

        elif new_phase == Phase.SUB_AGENT:
            # 先关闭之前的区域
            if not await self._close_sections(response, tracker):
                return False
            if not await self._open_subagent_section(response, node_name):
                return False
            tracker.subagent_opened = True

        elif new_phase == Phase.EXECUTION:
            # 关闭思考区/子代理区，进入正式内容
            if not await self._close_sections(response, tracker):
                return False
            tracker.has_sent_content = True

        elif new_phase == Phase.REPORTING:
            # 报告阶段：先关闭所有区域，HTML 标记直接透传
            if not await self._close_sections(response, tracker):
                return False

        tracker.current_phase = new_phase
        logger.debug(f"阶段切换: {old_phase.value} → {new_phase.value}")
        return True

    async def _process_update_message(self, msg, response, answer_collector: list) -> bool:
        """
        处理 updates 模式下的单条消息（工具调用/结果）

        Args:
            answer_collector: 收集所有输出内容的列表

        Returns:
            bool: True=成功, False=连接断开
        """
        try:
            if isinstance(msg, AIMessage):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        name = tc.get("name", "unknown")
                        args = tc.get("args", {})
                        tool_msg = self._format_tool_call(name, args)
                        if tool_msg:
                            if not await self._safe_write(response, tool_msg, "info"):
                                return False
                            answer_collector.append(tool_msg)

            elif isinstance(msg, ToolMessage):
                name = getattr(msg, "name", "")
                content_str = str(msg.content) if msg.content else ""
                tool_result_msg = self._format_tool_result(name, content_str)
                if tool_result_msg:
                    msg_type = "error" if "error" in content_str.lower() else "info"
                    if not await self._safe_write(response, tool_result_msg, msg_type):
                        return False
                    answer_collector.append(tool_result_msg)

            return True
        except Exception as e:
            if self._is_connection_error(e):
                logger.info(f"处理消息时连接断开: {type(e).__name__}")
                return False
            raise

    # ==================== 兼容接口 ====================

    async def cancel_task(self, task_id: str) -> bool:
        """取消任务（兼容接口，供 llm_service 调用）"""
        logger.info(f"收到取消请求: {task_id}")
        if task_id in self.running_tasks:
            self.running_tasks[task_id]["cancelled"] = True
            logger.info(f"任务已标记取消: {task_id}")
            return True
        return False
