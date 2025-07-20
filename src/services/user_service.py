import json
import logging
import os
import traceback
from datetime import datetime, timedelta

import jwt
import requests

from common.exception import MyException
from common.mysql_util import MysqlUtil
from constants.code_enum import SysCodeEnum, DiFyAppEnum
from constants.dify_rest_api import DiFyRestApi

logger = logging.getLogger(__name__)

mysql_client = MysqlUtil()


async def authenticate_user(username, password):
    """验证用户凭据并返回用户信息或 None"""
    sql = f"""select * from t_user where userName='{username}' and password='{password}'"""
    report_dict = MysqlUtil().query_mysql_dict(sql)
    if len(report_dict) > 0:
        return report_dict[0]
    else:
        return False


async def generate_jwt_token(user_id, username):
    """生成 JWT token"""
    payload = {"id": str(user_id), "username": username, "exp": datetime.utcnow() + timedelta(hours=24)}  # Token 过期时间
    token = jwt.encode(payload, os.getenv("JWT_SECRET_KEY"), algorithm="HS256")
    return token


async def decode_jwt_token(token):
    """解析 JWT token 并返回 payload"""
    try:
        # 使用与生成 token 时相同的密钥和算法来解码 token
        payload = jwt.decode(token, key=os.getenv("JWT_SECRET_KEY"), algorithms=["HS256"])
        # 检查 token 是否过期
        if "exp" in payload and datetime.utcfromtimestamp(payload["exp"]) < datetime.utcnow():
            raise jwt.ExpiredSignatureError("Token has expired")
        return payload
    except jwt.ExpiredSignatureError as e:
        # 处理过期的 token
        return None, 401, str(e)
    except jwt.InvalidTokenError as e:
        # 处理无效的 token
        return None, 400, str(e)
    except Exception as e:
        # 处理其他可能的错误
        return None, 500, str(e)


async def get_user_info(request) -> dict:
    """获取登录用户信息"""
    token = request.headers.get("Authorization")

    # 检查 Authorization 头是否存在
    if not token:
        logging.error("Authorization header is missing")
        raise MyException(SysCodeEnum.c_401)

    # 检查 Authorization 头格式是否正确
    if not token.startswith("Bearer "):
        logging.error("Invalid Authorization header format")
        raise MyException(SysCodeEnum.c_400)

    # 提取 token
    token = token.split(" ")[1].strip()

    # 检查 token 是否为空
    if not token:
        logging.error("Token is empty or whitespace")
        raise MyException(SysCodeEnum.c_400)

    try:
        # 解码 JWT token
        user_info = await decode_jwt_token(token)
    except Exception as e:
        logging.error(f"Failed to decode JWT token: {e}")
        raise MyException(SysCodeEnum.c_401)

    return user_info


async def add_question_record(uuid_str, user_token, conversation_id, message_id, task_id, chat_id, question, t02_answer, t04_answer, qa_type):
    """
    @:param uuid_str: 唯一ID
    @param user_token: 用户token
    @param conversation_id: dify会话ID
    @param message_id: 消息ID
    @param task_id: 任务ID
    @param chat_id: 聊天ID
    @param question: 问题
    @param t02_answer: 回答
    @param t04_answer: 回答
    @param qa_type: 问答类型
    记录用户问答记录，如果记录已存在，则更新之；否则，创建新记录。
    """
    try:
        # 解析token信息
        user_dict = await decode_jwt_token(user_token)
        user_id = user_dict["id"]

        # 文件问答时保存 minio/key
        file_key = ""
        if qa_type == DiFyAppEnum.FILEDATA_QA.value[0]:
            file_key = question.split("|")[0]
            question = question.split("|")[1]

        sql = f"select * from t_user_qa_record where user_id={user_id} and chat_id='{chat_id}' and message_id='{message_id}'"
        log_dict = mysql_client.query_mysql_dict(sql)

        # 根据 message_id 判断是否是同一个问题
        if len(log_dict) > 0:
            sql = f"""update t_user_qa_record set to4_answer='{json.dumps(t04_answer, ensure_ascii=False)}' 
                    where user_id={user_id} and chat_id='{chat_id}' and message_id='{message_id}'"""
            mysql_client.update(sql)
        else:
            insert_params = [
                uuid_str,
                user_id,
                conversation_id,
                message_id,
                task_id,
                chat_id,
                question,
                json.dumps(t02_answer, ensure_ascii=False),
                qa_type,
                file_key,
            ]
            sql = (
                f" insert into t_user_qa_record(uuid,user_id,conversation_id, message_id, task_id,chat_id,question,to2_answer,qa_type,file_key) "
                f"values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
            )
            mysql_client.insert(sql=sql, params=insert_params)

    except Exception as e:
        traceback.print_exception(e)
        logger.error(f"保存用户问答日志失败: {e}")


async def delete_user_record(user_id, record_ids):
    """
    删除用户问答记录
    :param user_id: 用户ID
    :param record_ids: 要删除的记录ID列表
    :return: None
    """
    # 确保 record_ids 是一个非空列表
    if not isinstance(record_ids, list) or not record_ids:
        raise ValueError("record_ids 必须是非空列表")

    # 创建 IN 子句和对应的参数列表
    in_clause = ", ".join(["%s"] * len(record_ids))
    sql = f"""
        DELETE FROM t_user_qa_record
        WHERE user_id = %s AND id IN ({in_clause})
    """

    # 将 user_id 添加到参数列表的开头
    params = [user_id] + record_ids

    # 执行更新操作
    mysql_client.update_params(sql=sql, params=params)


async def query_user_record(user_id, page, limit, search_text, chat_id):
    """
    根据用户id查询用户问答记录
    :param page
    :param limit
    :param user_id
    :param search_text
    :param chat_id
    :return:
    """
    conditions = []
    if chat_id:
        conditions.append(f"chat_id = '{chat_id}'")
    if search_text:
        conditions.append(f"question LIKE '%{search_text}%'")
    elif user_id:
        conditions.append(f"user_id = {user_id}")

    count_sql = "select count(1) as count from t_user_qa_record"
    if conditions:
        count_sql += " where " + " and ".join(conditions)
    total_count = mysql_client.query_mysql_dict(count_sql)[0]["count"]
    total_pages = (total_count + limit - 1) // limit  # 计算总页数

    # 计算偏移量
    offset = (page - 1) * limit
    records_sql = f"""select * from t_user_qa_record"""
    if conditions:
        records_sql += " where " + " and ".join(conditions)
    records_sql += " order by id desc LIMIT {limit} OFFSET {offset}".format(limit=limit, offset=offset)
    records = mysql_client.query_mysql_dict(records_sql)

    return {"records": records, "current_page": page, "total_pages": total_pages, "total_count": total_count}


def query_user_qa_record(chat_id):
    """
    根据chat_id查询对话记录
    :param chat_id:
    :return:
    """
    sql = f"select * from t_user_qa_record where chat_id='{chat_id}' order by id desc limit 1"
    return mysql_client.query_mysql_dict(sql)


async def send_dify_feedback(chat_id, rating):
    """
    发送反馈给指定的消息ID。

    :param chat_id: 消息的唯一标识符。
    :param rating: 反馈评级（例如："like" 或 "dislike"）。
    :return: 返回服务器响应。
    """
    # 查询对话记录
    qa_record = query_user_qa_record(chat_id)
    url = DiFyRestApi.replace_path_params(DiFyRestApi.DIFY_REST_FEEDBACK, {"message_id": qa_record[0]["message_id"]})
    api_key = os.getenv("DIFY_DATABASE_QA_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"rating": rating, "user": "abc-123"}

    response = requests.post(url, headers=headers, json=payload)

    # 检查请求是否成功
    if response.status_code == 200:
        logger.info("Feedback successfully sent.")
    else:
        logger.error(f"Failed to send feedback. Status code: {response.status_code},Response body: {response.text}")
        raise
