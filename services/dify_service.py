import json
import logging
import os
import re
import traceback

import aiohttp
import requests

from agent.common_react_agent import CommonReactAgent
from agent.deep_research_agent import DeepAgent
from agent.excel.excel_agent import ExcelAgent
from agent.text2sql.text2_sql_agent import Text2SqlAgent
from common.exception import MyException
from constants.code_enum import (
    DiFyAppEnum,
    DataTypeEnum,
    DiFyCodeEnum,
    SysCodeEnum,
)
from constants.dify_rest_api import DiFyRestApi
from services.db_qadata_process import process
from services.user_service import add_question_record, query_user_qa_record, decode_jwt_token

logger = logging.getLogger(__name__)


class QaContext:
    """问答上下文信息"""

    def __init__(self, token, question, chat_id):
        self.token = token
        self.question = question
        self.chat_id = chat_id


common_agent = CommonReactAgent()
sql_agent = Text2SqlAgent()
excel_agent = ExcelAgent()
deep_agent = DeepAgent()


class DiFyRequest:
    """
    DiFy操作服务类
    """

    def __init__(self):
        pass

    async def exec_query(self, res):
        """

        :return:
        """
        try:
            # 获取请求体内容 从res流对象获取request-body
            req_body_content = res.request.body
            # 将字节流解码为字符串
            body_str = req_body_content.decode("utf-8")

            req_obj = json.loads(body_str)
            logging.info(f"query param: {body_str}")

            # str(uuid.uuid4())
            chat_id = req_obj.get("chat_id")
            qa_type = req_obj.get("qa_type")
            # 自定义id
            uuid_str = req_obj.get("uuid")

            # 获取文件列表
            file_list = req_obj.get("file_list")

            #  使用正则表达式移除所有空白字符（包括空格、制表符、换行符等）
            query = req_obj.get("query")
            cleaned_query = re.sub(r"\s+", "", query)

            # 获取登录用户信息
            token = res.request.headers.get("Authorization")
            if not token:
                raise MyException(SysCodeEnum.c_401)
            if token.startswith("Bearer "):
                token = token.split(" ")[1]

            # 调用智能体
            if qa_type == DiFyAppEnum.COMMON_QA.value[0]:
                await common_agent.run_agent(query, res, chat_id, uuid_str, token, file_list)
                return None
            elif qa_type == DiFyAppEnum.DATABASE_QA.value[0]:
                await sql_agent.run_agent(query, res, chat_id, uuid_str, token)
                return None
            elif qa_type == DiFyAppEnum.FILEDATA_QA.value[0]:
                # cleaned_query = file_list[0]["source_file_key"] + "|" + query
                await excel_agent.run_excel_agent(cleaned_query, res, chat_id, uuid_str, token, file_list)
                return None
            elif qa_type == DiFyAppEnum.REPORT_QA.value[0]:
                # cleaned_query = file_list[0]["source_file_key"] + "|" + query
                await deep_agent.run_agent(cleaned_query, res, chat_id, uuid_str, token, file_list)
                return None

            # 封装问答上下文信息
            qa_context = QaContext(token, cleaned_query, chat_id)

            # 判断请求类别
            app_key = self._get_authorization_token(qa_type)

            # 构建请求参数
            dify_service_url, body_params, headers = self._build_request(chat_id, cleaned_query, app_key, qa_type)

            # 收集流式输出结果
            t02_answer_data = []
            # 收集业务数据流式输出结果
            t04_answer_data = {}

            async with aiohttp.ClientSession(read_bufsize=1024 * 16) as session:
                async with session.post(
                    dify_service_url,
                    headers=headers,
                    json=body_params,
                    timeout=aiohttp.ClientTimeout(total=60 * 10),  # 等待10分钟超时
                ) as response:
                    logging.info(f"dify response status: {response.status}")
                    if response.status == 200:
                        # await self.res_begin(res, chat_id)
                        data_type = ""
                        bus_data = ""
                        while True:
                            reader = response.content
                            reader._high_water = 10 * 1024 * 1024  # 设置为10MB
                            chunk = await reader.readline()
                            if not chunk:
                                break
                            str_chunk = chunk.decode("utf-8")
                            # print(str_chunk)
                            if str_chunk.startswith("data"):
                                str_data = str_chunk[5:]
                                data_json = json.loads(str_data)
                                event_name = data_json.get("event")
                                conversation_id = data_json.get("conversation_id")
                                message_id = data_json.get("message_id")
                                task_id = data_json.get("task_id")

                                if DiFyCodeEnum.MESSAGE.value[0] == event_name:
                                    answer = data_json.get("answer")
                                    if answer and answer.startswith("dify_"):
                                        event_list = answer.split("_")
                                        if event_list[1] == "0":
                                            # 输出开始
                                            data_type = event_list[2]
                                            if data_type == DataTypeEnum.ANSWER.value[0]:
                                                await self.send_message(
                                                    res,
                                                    {"data": {"messageType": "begin"}, "dataType": data_type},
                                                    answer,
                                                )
                                        elif event_list[1] == "1":
                                            # 输出结束
                                            data_type = event_list[2]
                                            if data_type == DataTypeEnum.ANSWER.value[0]:
                                                await self.send_message(
                                                    res,
                                                    {"data": {"messageType": "end"}, "dataType": data_type},
                                                    answer,
                                                )

                                            # 输出业务数据
                                            elif bus_data and data_type == DataTypeEnum.BUS_DATA.value[0]:
                                                res_data = process(json.loads(bus_data)["data"])
                                                await self.send_message(
                                                    res,
                                                    {"data": res_data, "dataType": data_type},
                                                    answer,
                                                )
                                                t04_answer_data = {"data": res_data, "dataType": data_type}

                                            data_type = ""

                                    elif len(data_type) > 0:
                                        # 这里输出 t02之间的内容
                                        if data_type == DataTypeEnum.ANSWER.value[0]:
                                            await self.send_message(
                                                res,
                                                {
                                                    "data": {"messageType": "continue", "content": answer},
                                                    "dataType": data_type,
                                                    "task_id": task_id,
                                                },
                                                answer,
                                            )

                                            t02_answer_data.append(await self.format_answer(answer))

                                        # 这里设置业务数据
                                        if data_type == DataTypeEnum.BUS_DATA.value[0]:
                                            bus_data = answer

                                elif DiFyCodeEnum.MESSAGE_ERROR.value[0] == event_name:
                                    # 输出异常情况日志
                                    error_msg = data_json.get("message")
                                    logging.error(f"Error 调用dify失败错误信息: {data_json}")
                                    await res.write(
                                        "data:"
                                        + json.dumps(
                                            {
                                                "data": {
                                                    "messageType": "error",
                                                    "content": "调用失败请查看dify日志,错误信息: " + error_msg,
                                                },
                                                "dataType": DataTypeEnum.ANSWER.value[0],
                                            },
                                            ensure_ascii=False,
                                        )
                                        + "\n\n"
                                    )

                                elif DiFyCodeEnum.MESSAGE_END.value[0] == event_name:
                                    t02_message_json = {
                                        "data": {"messageType": "continue", "content": "".join(t02_answer_data)},
                                        "dataType": DataTypeEnum.ANSWER.value[0],
                                    }

                                    if t02_message_json:
                                        await self._save_message(
                                            t02_message_json,
                                            qa_context,
                                            conversation_id,
                                            message_id,
                                            task_id,
                                            qa_type,
                                            uuid_str,
                                            file_list,
                                        )
                                    if t04_answer_data:
                                        await self._save_message(
                                            t04_answer_data,
                                            qa_context,
                                            conversation_id,
                                            message_id,
                                            task_id,
                                            qa_type,
                                            uuid_str,
                                            file_list,
                                        )

                                    t02_answer_data = []
                                    t04_answer_data = {}

        except Exception as e:
            logging.error(f"Error during get_answer: {e}")
            traceback.print_exception(e)
            return {"error": str(e)}  # 返回错误信息作为字典
        finally:
            await self.res_end(res)

    @staticmethod
    async def _save_message(message, qa_context, conversation_id, message_id, task_id, qa_type, uuid_str, file_list):
        """
            保存消息记录并发送SSE数据
        :param message:
        :param qa_context:
        :param conversation_id:
        :param message_id:
        :param task_id:
        :param qa_type:
        :param file_list 用户上传的附件信息
        :return:
        """
        # 保存用户问答记录 1.保存用户问题 2.保存用户答案 t02 和 t04
        if "content" in message["data"]:
            await add_question_record(
                uuid_str,
                qa_context.token,
                conversation_id,
                message_id,
                task_id,
                qa_context.chat_id,
                qa_context.question,
                message,
                "",
                qa_type,
                file_list,
            )
        elif message["dataType"] == DataTypeEnum.BUS_DATA.value[0]:
            await add_question_record(
                uuid_str,
                qa_context.token,
                conversation_id,
                message_id,
                task_id,
                qa_context.chat_id,
                qa_context.question,
                "",
                message,
                qa_type,
                file_list,
            )

    @staticmethod
    async def format_answer(answer) -> dict:
        """
        格式化大模型输出的Token
        如果是思考型大模型则格式化思考过程消息
        :param answer:
        :return:
        """
        if answer and ("<think>" in answer):
            # 尝试提取<think>标签内容
            think_content = answer.replace("<think>", "").replace("</think>", "")
            # 构造思考过程的HTML格式
            think_html = """<details style="color:gray;background-color: #f8f8f8;padding: 2px;border-radius: 
                6px;margin-top:5px" open>
                    <summary> Thinking... </summary>"""

            formatted_message = think_html + think_content.replace("\n", "")
        elif answer and ("</think>" in answer):
            formatted_message = "</details>\n" + answer.replace("<think>", "").replace("</think>", "")
        else:
            formatted_message = answer

        return formatted_message

    async def send_message(self, response, message, answer):
        """
        SSE 格式发送数据，每一行以 data: 开头

        :param response: HTTP响应对象
        :param message: 要发送的消息数据
        :param answer: 原始回答内容
        """
        # await response.write("data:" + json.dumps(message, ensure_ascii=False) + "\n\n")

        # 检查是否需要特殊处理 < think > 标签
        if answer and ("<think>" in answer):
            try:
                # 尝试提取<think>标签内容
                think_content = answer.replace("<think>", "").replace("</think>", "")
                # 构造思考过程的HTML格式
                think_html = """<details style="color:gray;background-color: #f8f8f8;padding: 2px;border-radius:
                6px;margin-top:5px" open>
                    <summary> Thinking... </summary>"""

                # 组装完整消息
                formatted_message = {
                    "data": {
                        "messageType": "continue",
                        "content": think_html + think_content.replace("\n", ""),
                    },
                    "dataType": "t02",
                }
                await response.write("data:" + json.dumps(formatted_message, ensure_ascii=False) + "\n\n")
            except Exception as e:
                # 处理异常情况
                logging.warning(f"处理<think>标签时出错: {e}")
                await response.write("data:" + json.dumps(message, ensure_ascii=False) + "\n\n")
        else:
            # # 只有在 content 存在时才添加 </details>
            if answer and ("</think>" in answer):
                think_content = answer.replace("<think>", "").replace("</think>", "")
                await response.write(
                    "data:"
                    + json.dumps(
                        {
                            "data": {
                                "messageType": "continue",
                                "content": "</details>\n" + think_content,
                            },
                            "dataType": "t02",
                        },
                        ensure_ascii=False,
                    )
                    + "\n\n"
                )
            else:
                await response.write("data:" + json.dumps(message, ensure_ascii=False) + "\n\n")

    @staticmethod
    async def res_begin(res, chat_id):
        """

        :param res:
        :param chat_id:
        :return:
        """
        await res.write(
            "data:"
            + json.dumps(
                {
                    "data": {"id": chat_id},
                    "dataType": DataTypeEnum.TASK_ID.value[0],
                }
            )
            + "\n\n"
        )

    @staticmethod
    async def res_end(res):
        """
        :param res:
        :return:
        """
        await res.write(
            "data:"
            + json.dumps(
                {
                    "data": "DONE",
                    "dataType": DataTypeEnum.STREAM_END.value[0],
                }
            )
            + "\n\n"
        )

    @staticmethod
    def _build_request(chat_id, query, app_key, qa_type):
        """
        构建请求参数
        :param chat_id: 对话id
        :param app_key: api key
        :param query: 用户问题
        :param qa_type: 问答类型
        :return:
        """

        # 通用问答时，使用上次会话id 实现多轮对话效果
        conversation_id = ""
        if qa_type == DiFyAppEnum.COMMON_QA.value[0]:
            qa_record = query_user_qa_record(chat_id)
            if qa_record and len(qa_record) > 0:
                conversation_id = qa_record[0]["conversation_id"]

        body_params = {
            "query": query,
            "inputs": {"qa_type": qa_type},
            "response_mode": "streaming",
            "conversation_id": conversation_id,
            "user": "abc-123",
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {app_key}",
        }

        dify_service_url = DiFyRestApi.build_url(DiFyRestApi.DIFY_REST_CHAT)
        return dify_service_url, body_params, headers

    @staticmethod
    def _get_authorization_token(qa_type: str):
        """
            根据请求类别获取api/token
            固定走一个dify流
             app-IzudxfuN8uO2bvuCpUHpWhvH master分支默认的数据问答key
            :param qa_type
        :return:
        """
        # 遍历枚举成员并检查第一个元素是否与测试字符串匹配
        for member in DiFyAppEnum:
            if member.value[0] == qa_type:
                return os.getenv("DIFY_DATABASE_QA_API_KEY")
        else:
            raise ValueError(f"问答类型 '{qa_type}' 不支持")


async def query_dify_suggested(chat_id) -> dict:
    """
    发送反馈给指定的消息ID。

    :param chat_id: 消息的唯一标识符。
    :return: 返回服务器响应。
    """
    # 查询对话记录
    qa_record = query_user_qa_record(chat_id)
    url = DiFyRestApi.replace_path_params(DiFyRestApi.DIFY_REST_SUGGESTED, {"message_id": qa_record[0]["message_id"]})
    logger.info(f"query dify suggested url: {url}")
    api_key = os.getenv("DIFY_DATABASE_QA_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    response = requests.get(url + "?user=abc-123", headers=headers)

    # 检查请求是否成功
    if response.status_code == 200:
        logger.info("Feedback successfully sent.")
        return response.json()
    else:
        logger.error(f"Failed to send feedback. Status code: {response.status_code},Response body: {response.text}")
        raise


async def stop_dify_chat(request, task_id, qa_type) -> dict:
    """
    停止dify对话流输出

    :param task_id: 任务id。
    :param qa_type: 问答类型
    :param request
    :return: 返回服务器响应。
    """
    # 获取登录用户信息
    token = request.headers.get("Authorization")
    if not token:
        raise MyException(SysCodeEnum.c_401)
    if token.startswith("Bearer "):
        token = token.split(" ")[1]

    # 通用问答和数据问答停止任务
    if qa_type == DiFyAppEnum.COMMON_QA.value[0]:
        user_dict = await decode_jwt_token(token)
        task_id = user_dict["id"]
        success = await common_agent.cancel_task(task_id)
        return {"success": success, "message": "任务已停止" if success else "未找到任务"}

    elif qa_type == DiFyAppEnum.DATABASE_QA.value[0]:
        user_dict = await decode_jwt_token(token)
        task_id = user_dict["id"]
        success = await sql_agent.cancel_task(task_id)
        return {"success": success, "message": "任务已停止" if success else "未找到任务"}

    else:
        # 文件问答和报告问答停止任务走默认的dify接口

        # 查询对话记录
        url = DiFyRestApi.replace_path_params(DiFyRestApi.DIFY_REST_STOP, {"task_id": task_id})

        api_key = os.getenv("DIFY_DATABASE_QA_API_KEY")
        # 行业报告走的是 报告问答的key
        if DiFyAppEnum.FILEDATA_QA.value[0] == qa_type:
            api_key = os.getenv("DIFY_ENTERPRISE_REPORT_API_KEY")

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        body = {"user": "abc-123"}

        logger.info(url)

        """
        data：若传入字典或元组列表，requests 库会把数据编码为表单数据格式（key1=value1&key2=value2）；若传入字节或类文件对象，则直接发送。
        json：requests 库会自动把传入的 Python 对象序列化为 JSON 字符串，然后发送。
        """
        response = requests.post(url, json=body, headers=headers)

        # 检查请求是否成功
        if response.status_code == 200:
            logger.info("Stop chat successfully sent.")
            return response.json()
        else:
            logger.error(f"Failed to stop chat. Status code: {response.status_code},Response body: {response.text}")
            raise
