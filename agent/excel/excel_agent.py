import asyncio
import json
import logging
import os
import traceback
from typing import Optional, Dict, Any

from langgraph.graph.state import CompiledStateGraph

from agent.excel.excel_agent_state import ExcelAgentState
from agent.excel.excel_graph import create_excel_graph
from constants.code_enum import DataTypeEnum
from services.user_service import decode_jwt_token, add_user_record, query_user_qa_record
from agent.excel.excel_duckdb_manager import close_duckdb_manager, get_chat_duckdb_manager

logger = logging.getLogger(__name__)


class ExcelAgent:
    """
    表格问答智能体
    """

    def __init__(self):
        # 存储运行中的任务
        self.running_tasks = {}
        # 获取环境变量控制是否显示思考过程，默认为开启
        self.show_thinking_process = os.getenv("SHOW_THINKING_PROCESS", "true").lower() == "true"
        self.excel_graph = create_excel_graph()

    async def run_excel_agent(
        self,
        query: str,
        response=None,
        chat_id: str = None,
        uuid_str: str = None,
        user_token=None,
        file_list: list = None,
    ) -> None:
        """
        运行表格智能体
        :param query:
        :param response:
        :param chat_id:
        :param uuid_str:
        :param user_token:
        :param file_list
        :return:
        """
        t02_answer_data = []
        t04_answer_data = {}
        current_step = None

        # 实现上传一次多次对话的效果 默认单轮对话取最新上传的文件
        if file_list is None:
            # todo 使用graph的 state进行管理。
            user_qa_record = query_user_qa_record(chat_id)[0]
            if user_qa_record:
                file_list = json.loads(user_qa_record["file_key"])
        try:
            initial_state = ExcelAgentState(
                user_query=query,
                file_list=file_list,
                chat_id=chat_id,  # chat_id
                file_metadata={},  # 文件元数据
                sheet_metadata={},  # Sheet元数据
                db_info=[],  # 支持多个表结构
                catalog_info={},  # Catalog信息
                generated_sql="",
                chart_url="",
                chart_type="",
                apache_chart_data={},
                execution_result=None,  # 修改：使用ExecutionResult对象
                report_summary="",
            )
            graph: CompiledStateGraph = self.excel_graph

            # 获取用户信息 标识对话状态
            user_dict = await decode_jwt_token(user_token)
            task_id = user_dict["id"]
            task_context = {"cancelled": False}
            self.running_tasks[task_id] = task_context

            async for chunk_dict in graph.astream(initial_state, stream_mode="updates"):
                # 检查是否已取消
                if self.running_tasks[task_id]["cancelled"]:
                    if self.show_thinking_process:
                        await self._send_response(response, "</details>\n\n", "continue", DataTypeEnum.ANSWER.value[0])
                    await response.write(
                        self._create_response("\n> 这条消息已停止", "info", DataTypeEnum.ANSWER.value[0])
                    )
                    # 发送最终停止确认消息
                    await response.write(self._create_response("", "end", DataTypeEnum.STREAM_END.value[0]))
                    break

                logger.info(f"Processing chunk: {chunk_dict}")

                langgraph_step, step_value = next(iter(chunk_dict.items()))

                # 处理步骤变更
                current_step, t02_answer_data = await self._handle_step_change(
                    response, current_step, langgraph_step, t02_answer_data
                )

                # 处理具体步骤内容
                if step_value:
                    await self._process_step_content(
                        response, langgraph_step, step_value, t02_answer_data, t04_answer_data
                    )

            # 流结束时关闭最后的details标签
            if self.show_thinking_process:
                if current_step is not None and current_step not in ["summarize", "data_render", "data_render_apache"]:
                    await self._close_current_step(response, t02_answer_data)

            # 只有在未取消的情况下才保存记录
            if not self.running_tasks[task_id]["cancelled"]:
                await add_user_record(
                    uuid_str,
                    chat_id,
                    query,
                    t02_answer_data,
                    t04_answer_data,
                    "FILEDATA_QA",
                    user_token,
                    file_list,
                )

        except asyncio.CancelledError:
            await response.write(self._create_response("\n> 这条消息已停止", "info", DataTypeEnum.ANSWER.value[0]))
            await response.write(self._create_response("", "end", DataTypeEnum.STREAM_END.value[0]))
        except Exception as e:
            traceback.print_exception(e)
            logger.error(f"表格问答智能体运行异常: {e}")
            error_msg = f"处理过程中发生错误: {str(e)}"
            await self._send_response(response, error_msg, "error")

    async def _handle_step_change(
        self,
        response,
        current_step: Optional[str],
        new_step: str,
        t02_answer_data: list,
    ) -> tuple:
        """
        处理步骤变更
        """
        if self.show_thinking_process:
            if new_step != current_step:
                # 如果之前有打开的步骤，先关闭它
                if current_step is not None and current_step not in ["summarize", "data_render", "data_render_apache"]:
                    await self._close_current_step(response, t02_answer_data)

                # 打开新的步骤 (除了 summarize 和 data_render) think_html 标签里面添加open属性控制思考过程是否默认展开显示
                if new_step not in ["summarize", "data_render", "data_render_apache"]:
                    think_html = f"""<details style="color:gray;background-color: #f8f8f8;padding: 2px;border-radius: 
                    6px;margin-top:5px;">
                                 <summary>{new_step}...</summary>"""
                    await self._send_response(response, think_html, "continue", "t02")
                    t02_answer_data.append(think_html)
        else:
            # 如果不显示思考过程，则只处理特定的步骤
            if new_step in ["summarize", "data_render", "data_render_apache"]:
                # 对于需要显示的步骤，确保之前的步骤已关闭
                if current_step is not None and current_step not in ["summarize", "data_render", "data_render_apache"]:
                    pass  # 不需要关闭details标签，因为我们根本没有打开它

        return new_step, t02_answer_data

    async def _close_current_step(self, response, t02_answer_data: list) -> None:
        """
        关闭当前步骤的details标签
        """
        if self.show_thinking_process:
            close_tag = "</details>\n\n"
            await self._send_response(response, close_tag, "continue", "t02")
            t02_answer_data.append(close_tag)

    async def _process_step_content(
        self,
        response,
        step_name: str,
        step_value: Dict[str, Any],
        t02_answer_data: list,
        t04_answer_data: Dict[str, Any],
    ) -> None:
        """
        处理各个步骤的内容
        """
        content_map = {
            "excel_parsing": lambda: self._format_multi_file_table_info(step_value),
            "sql_generator": lambda: step_value.get("generated_sql", ""),
            "sql_executor": lambda: self._format_execution_result(step_value.get("execution_result")),
            "summarize": lambda: step_value.get("report_summary", ""),
            "data_render": lambda: step_value.get("chart_url", ""),
            "data_render_apache": lambda: step_value.get("apache_chart_data", {}),
        }

        if step_name in content_map:
            content = content_map[step_name]()
            if step_name == "data_render":
                content = "\n---\n" + content

            # 适配EChart表格
            data_type = (
                DataTypeEnum.ANSWER.value[0] if step_name != "data_render_apache" else DataTypeEnum.BUS_DATA.value[0]
            )

            # 根据环境变量决定是否发送非关键步骤的内容
            should_send = self.show_thinking_process or step_name in ["summarize", "data_render", "data_render_apache"]

            if should_send:
                await self._send_response(response=response, content=content, data_type=data_type)

                if data_type == DataTypeEnum.ANSWER.value[0]:
                    t02_answer_data.append(content)

            # 这里设置 Apache 表格数据
            if step_name == "data_render_apache" and data_type == DataTypeEnum.BUS_DATA.value[0]:
                t04_answer_data.clear()
                t04_answer_data.update({"data": step_value.get("apache_chart_data", {}), "dataType": data_type})

            # 对于非渲染步骤，刷新响应
            if step_name not in ["data_render", "data_render_apache"]:
                if hasattr(response, "flush"):
                    await response.flush()
                await asyncio.sleep(0)

    @staticmethod
    async def _send_response(
        response, content: str, message_type: str = "continue", data_type: str = DataTypeEnum.ANSWER.value[0]
    ) -> None:
        """
        发送响应数据
        """
        if response:
            if data_type == DataTypeEnum.ANSWER.value[0]:
                formatted_message = {
                    "data": {
                        "messageType": message_type,
                        "content": content,
                    },
                    "dataType": data_type,
                }
            else:
                # 适配EChart表格
                formatted_message = {"data": content, "dataType": data_type}

            await response.write("data:" + json.dumps(formatted_message, ensure_ascii=False) + "\n\n")

    @staticmethod
    def _create_response(
        content: str, message_type: str = "continue", data_type: str = DataTypeEnum.ANSWER.value[0]
    ) -> str:
        """
        封装响应结构（保持向后兼容）
        """
        res = {
            "data": {"messageType": message_type, "content": content},
            "dataType": data_type,
        }
        return "data:" + json.dumps(res, ensure_ascii=False) + "\n\n"

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

    def cleanup_chat_session(self, chat_id: str) -> bool:
        """
        清理指定chat_id的DuckDB会话
        :param chat_id: 聊天ID
        :return: 是否成功清理
        """
        try:
            return close_duckdb_manager(chat_id=chat_id)
        except Exception as e:
            logger.error(f"清理chat_id '{chat_id}' 的DuckDB会话失败: {str(e)}")
            return False

    def get_chat_session_stats(self) -> dict:
        """
        获取聊天会话统计信息
        :return: 会话统计信息
        """
        try:
            chat_manager = get_chat_duckdb_manager()
            return {
                "active_chat_count": chat_manager.get_active_chat_count(),
                "chat_list": chat_manager.get_chat_list(),
            }
        except Exception as e:
            logger.error(f"获取聊天会话统计失败: {str(e)}")
            return {"active_chat_count": 0, "chat_list": []}

    @staticmethod
    def _format_multi_file_table_info(state: Dict[str, Any]) -> str:
        """
        格式化多文件多Sheet信息为HTML格式
        :param state: 状态字典
        :return: 格式化后的HTML字符串
        """
        file_metadata = state.get("file_metadata", {})
        sheet_metadata = state.get("sheet_metadata", {})
        db_info = state.get("db_info", [])

        if not file_metadata and not db_info:
            return "未找到文件信息"

        html_content = """
        文件解析结果<br>
        """

        # 显示文件信息
        if file_metadata:
            html_content += "文件列表：<br><ol>"
            for file_key, file_info in file_metadata.items():
                html_content += f"<li>file_name: {file_info.file_name}  |"
                f"Catalog: {file_info.catalog_name} |"
                f"Sheets: {file_info.sheet_count} |"
                f"上传时间: {file_info.upload_time} </li>"
            html_content += "</ol><br>"

        # 显示表信息
        if db_info:
            html_content += "数据表：<br><ol>"
            for table in db_info:
                table_name = table.get("table_name", "未知表")
                table_comment = table.get("table_comment", "")
                columns = table.get("columns", {})
                html_content += (
                    f"<li>table_name:{table_name} | table_comment:{table_comment} | 列数: {len(columns)} </li>"
                )

            html_content += "</ol><br>"

        html_content += "<br>"
        return html_content

    @staticmethod
    def _format_execution_result(execution_result) -> str:
        """
        格式化SQL执行结果
        :param execution_result: ExecutionResult对象
        :return: 格式化后的字符串
        """
        if not execution_result:
            return "未找到执行结果"

        if execution_result.success:
            row_count = len(execution_result.data) if execution_result.data else 0
            column_count = len(execution_result.columns) if execution_result.columns else 0
            return f"✅ 查询执行成功！返回 {row_count} 行数据，{column_count} 列"
        else:
            return f"❌ 查询执行失败：{execution_result.error or '未知错误'}"

    @staticmethod
    def _format_table_columns_info(db_info: Dict[str, Any]) -> str:
        """
        格式化表格列信息为HTML details标签格式
        :param db_info: 数据库信息字典
        :return: 格式化后的HTML字符串
        """
        db_info = db_info["db_info"]
        if not db_info or "columns" not in db_info:
            return ""

        columns_info = db_info["columns"]

        html_content = """
        <ul>
        """
        for column_name, column_details in columns_info.items():
            comment = column_details.get("comment", column_name)
            type_ = column_details.get("type", "未知")
            html_content += f"<li><strong>{column_name}</strong>: {comment} (类型: {type_})</li>\n"

        html_content += """</ul>"""

        return html_content
