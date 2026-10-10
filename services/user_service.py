import json
import logging
import os
import traceback
from datetime import datetime, timedelta
from typing import List, Any

import bcrypt
import jwt
import requests
from sqlalchemy import text
from sqlalchemy.orm import Session

from common.exception import MyException
from constants.code_enum import SysCodeEnum, IntentEnum, DataTypeEnum
from constants.dify_rest_api import DiFyRestApi
from model.db_connection_pool import get_db_pool
from model.db_models import TUserQaRecord, TUser
from model.serializers import model_to_dict
from model.schemas import PaginatedResponse

logger = logging.getLogger(__name__)

# 固定盐值，确保相同的密码生成相同的哈希值（注意：生产环境通常建议随机盐值以提高安全性，但根据需求此处固定）
PASSWORD_SALT = b'$2b$12$rmnFss1KlnSgcRKCv/Q8e.'

pool = get_db_pool()


def execute_sql_dict(sql: str, params: tuple = None) -> List[dict]:
    """
    执行 SQL 查询并返回字典列表
    :param sql: SQL 查询语句（支持 %s 占位符）
    :param params: 参数元组（可选）
    :return: 字典列表
    """
    with pool.get_session() as session:
        if params:
            # 将 %s 占位符转换为命名参数
            param_dict = {f"param_{i}": val for i, val in enumerate(params)}
            # 替换 %s 为命名参数
            sql_with_params = sql
            for i in range(len(params)):
                sql_with_params = sql_with_params.replace("%s", f":param_{i}", 1)
            result = session.execute(text(sql_with_params), param_dict)
        else:
            result = session.execute(text(sql))
        rows = result.fetchall()
        columns = result.keys()

        result_list = []
        for row in rows:
            row_dict = {}
            for i, col in enumerate(columns):
                value = row[i]
                # 处理日期时间类型
                if isinstance(value, datetime):
                    value = value.strftime("%Y-%m-%d %H:%M:%S")
                row_dict[col] = value
            result_list.append(row_dict)
        return result_list


def execute_sql_update(sql: str, params: tuple = None):
    """
    执行 SQL 更新/插入/删除操作
    :param sql: SQL 语句（支持 %s 占位符）
    :param params: 参数元组（可选）
    :return: 影响的行数
    """
    with pool.get_session() as session:
        if params:
            # 将 %s 占位符转换为命名参数
            param_dict = {f"param_{i}": val for i, val in enumerate(params)}
            # 替换 %s 为命名参数
            sql_with_params = sql
            for i in range(len(params)):
                sql_with_params = sql_with_params.replace("%s", f":param_{i}", 1)
            result = session.execute(text(sql_with_params), param_dict)
        else:
            result = session.execute(text(sql))
        session.commit()
        return result.rowcount


async def authenticate_user(username, password):
    """验证用户凭据并返回用户信息或 None"""
    with pool.get_session() as session:
        session: Session = session
        user = session.query(TUser).filter(TUser.userName == username).first()
        if user and user.password:
            # 1. 优先尝试使用固定盐值验证
            try:
                if bcrypt.hashpw(password.encode('utf-8'), PASSWORD_SALT).decode('utf-8') == user.password:
                    return model_to_dict(user)
            except Exception:
                pass

            # 2. 兼容性验证：尝试标准bcrypt验证 (适用于旧的随机盐值)
            try:
                if bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
                    return model_to_dict(user)
            except ValueError:
                pass

            # 3. 兼容性验证：明文密码
            if user.password == password:
                return model_to_dict(user)
        return False
    # sql = f"""select * from t_user where userName='{username}' and password='{password}'"""
    # report_dict = MysqlUtil().query_mysql_dict(sql)
    # if len(report_dict) > 0:
    #     return report_dict[0]
    # else:
    #     return False


async def generate_jwt_token(user_id, username, role="user"):
    """生成 JWT token"""
    payload = {
        "id": str(user_id),
        "username": username,
        "role": role,
        "exp": datetime.utcnow() + timedelta(days=7),
    }  # Token 过期时间
    jwt_secret_key = os.getenv("JWT_SECRET_KEY")
    if not jwt_secret_key:
        raise RuntimeError("JWT_SECRET_KEY environment variable is not set")
    token = jwt.encode(payload, jwt_secret_key, algorithm="HS256")
    return token


async def decode_jwt_token(token):
    """解析 JWT token 并返回 payload"""
    try:
        # 使用与生成 token 时相同的密钥和算法来解码 token
        jwt_secret_key = os.getenv("JWT_SECRET_KEY")
        if not jwt_secret_key:
            raise RuntimeError("JWT_SECRET_KEY environment variable is not set")
        payload = jwt.decode(token, key=jwt_secret_key, algorithms=["HS256"])
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


async def add_question_record(
    uuid_str, user_token, conversation_id, message_id, task_id, chat_id, question, t02_answer, t04_answer, qa_type
):
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
        if qa_type == IntentEnum.FILEDATA_QA.value[0]:
            file_key = question.split("|")[0]
            question = question.split("|")[1]

        sql = f"select * from t_user_qa_record where user_id={user_id} and chat_id='{chat_id}' and message_id='{message_id}'"
        log_dict = execute_sql_dict(sql)

        # 根据 message_id 判断是否是同一个问题
        if len(log_dict) > 0:
            sql = f"""update t_user_qa_record set to4_answer='{json.dumps(t04_answer, ensure_ascii=False)}' 
                    where user_id={user_id} and chat_id='{chat_id}' and message_id='{message_id}'"""
            execute_sql_update(sql)
        else:
            insert_params = (
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
            )
            sql = (
                f" insert into t_user_qa_record(uuid,user_id,conversation_id, message_id, task_id,chat_id,question,to2_answer,qa_type,file_key) "
                f"values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
            )
            execute_sql_update(sql, insert_params)

    except Exception as e:
        traceback.print_exception(e)
        logger.error(f"保存用户问答日志失败: {e}")


async def add_user_record(
    uuid_str: str,
    chat_id: int,
    question: str,
    to2_answer: List[str],
    to4_answer: dict[str, Any],
    qa_type: str,
    user_token: str,
    file_list: dict[str, Any] = None,
    datasource_id: int = None,
    sql_statement: str = "",
):
    """
    新增用户问答记录
    :param sql_statement: SQL语句（数据问答时保存，其他类型使用默认值空字符串）
    """
    try:
        # 1. 解析用户信息
        user_info = await decode_jwt_token(user_token)
        user_id = user_info.get("id")
        if not user_id:
            raise ValueError("Invalid user token: missing user_id")

        # 2. 组装 answer 数据 - 修复部分：确保所有元素转换为字符串
        t02_content = "".join(str(item) for item in (to2_answer or []))
        t02_message_json = {
            "data": {"messageType": "continue", "content": t02_content},
            "dataType": DataTypeEnum.ANSWER.value[0],
        }
        t02_answer_str = json.dumps(t02_message_json, ensure_ascii=False)

        # 3. 插入数据库并返回插入的记录ID
        insert_sql = """
            INSERT INTO t_user_qa_record
            (uuid, user_id, chat_id, question, to2_answer,to4_answer, qa_type,file_key, datasource_id, sql_statement)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        insert_params = (
            uuid_str,
            user_id,
            chat_id,
            question,
            t02_answer_str,
            json.dumps(to4_answer, ensure_ascii=False),
            qa_type,
            json.dumps(file_list, ensure_ascii=False) if file_list and len(file_list) > 0 else "",
            datasource_id,
            sql_statement or "",  # 确保是字符串，非数据问答类型使用空字符串
        )

        # 执行插入并获取返回的ID
        result = execute_sql_dict(sql=insert_sql, params=insert_params)
        record_id = result[0]["id"] if result else None
        
        return record_id

    except Exception as e:
        # 建议替换成项目的日志系统
        logger.error(f"Failed to insert user QA record: {e}", exc_info=True)
        raise


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
        WHERE user_id = %s AND chat_id IN ({in_clause})
    """

    # 将 user_id 添加到参数列表的开头
    params = tuple([user_id] + record_ids)

    # 执行更新操作
    execute_sql_update(sql=sql, params=params)


async def query_user_record(user_id, page, size, search_text, chat_id):
    """
    根据用户id查询用户问答记录
    如果chat_id不为空，则查询该chat_id的所有记录；否则根据chat_id去重，取id最小的那条
    :param page
    :param size
    :param user_id
    :param search_text
    :param chat_id
    :return:
    """
    conditions = []
    if chat_id:
        conditions.append(f"chat_id = '{chat_id}'")
    if search_text:
        search_text = search_text.strip()  # 去除search_text首尾空格
        conditions.append(f"question LIKE '%{search_text}%'")
    elif user_id:
        conditions.append(f"user_id = {user_id}")

    # 计算偏移量
    offset = (page - 1) * size

    # 如果chat_id不为空，则不需要去重，直接查询
    if chat_id:
        count_sql = "SELECT COUNT(1) as count FROM t_user_qa_record"
        if conditions:
            count_sql += " WHERE " + " AND ".join(conditions)
        total_count_result = execute_sql_dict(count_sql)
        total_count = total_count_result[0]["count"] if total_count_result else 0
        total_pages = (total_count + size - 1) // size

        records_sql = f"SELECT t.*, d.name as datasource_name FROM t_user_qa_record t LEFT JOIN t_datasource d ON t.datasource_id = d.id"
        if conditions:
            # Note: We need to adjust column references if they are ambiguous, but here conditions are simple
            # However, since we aliased t_user_qa_record as t, we should probably update conditions or just use the table name in WHERE if not ambiguous
            # Actually, the conditions constructed earlier use simple column names. 
            # To be safe, let's prefix them with 't.' in the WHERE clause or just rely on them being unique enough (except id which is in both)
            # The conditions construction was: conditions.append(f"chat_id = '{chat_id}'") etc.
            # To avoid ambiguity with 'id' or 'name' (though name is in datasource), let's just append WHERE clause.
            # But wait, conditions uses `question LIKE` and `user_id =`.
            # `chat_id` is in t_user_qa_record. `datasource_id` is in t_user_qa_record.
            # `id` is in both. `name` is in datasource.
            # The conditions list is built before.
            # Let's rebuild conditions with 't.' prefix or just use table alias in query.
            where_clause = " WHERE " + " AND ".join([f"t.{c}" if "id" in c or "question" in c or "chat_id" in c or "user_id" in c else c for c in conditions])
            records_sql += where_clause
        records_sql += f" ORDER BY t.id ASC LIMIT {size} OFFSET {offset}"
        records = execute_sql_dict(records_sql)
    else:
        # 如果chat_id为空，则需要去重，根据chat_id取id最小的记录
        base_condition = ""
        if conditions:
            base_condition = " WHERE " + " AND ".join(conditions)

        count_sql = f"""
            SELECT COUNT(1) as count FROM (
                SELECT chat_id, MIN(id) as min_id 
                FROM t_user_qa_record 
                {base_condition}
                GROUP BY chat_id
            ) as distinct_chats
        """
        total_count_result = execute_sql_dict(count_sql)
        total_count = total_count_result[0]["count"] if total_count_result else 0
        total_pages = (total_count + size - 1) // size

        # 查询去重后的记录，根据chat_id分组并取id最小的记录
        records_sql = f"""
            SELECT t.*, d.name as datasource_name FROM t_user_qa_record t
            INNER JOIN (
                SELECT chat_id, MIN(id) as min_id 
                FROM t_user_qa_record 
                {base_condition}
                GROUP BY chat_id
            ) tm ON t.chat_id = tm.chat_id AND t.id = tm.min_id
            LEFT JOIN t_datasource d ON t.datasource_id = d.id
            ORDER BY t.id DESC 
            LIMIT {size} OFFSET {offset}
        """
        records = execute_sql_dict(records_sql)

    return PaginatedResponse(
        records=records,
        current_page=page,
        total_count=total_count,
        total_pages=total_pages,
    )


async def query_user_record_list(user_id, page, size, search_text):
    """
    查询用户对话历史列表（简化版，只返回必要字段，用于登录渲染优化）
    根据chat_id去重，取id最小的那条
    :param page: 页码
    :param size: 每页大小
    :param user_id: 用户ID
    :param search_text: 搜索关键词
    :return: 包含必要字段的记录列表
    """
    conditions = []
    if search_text:
        search_text = search_text.strip()  # 去除search_text首尾空格
        conditions.append(f"question LIKE '%{search_text}%'")
    if user_id:
        conditions.append(f"user_id = {user_id}")

    # 计算偏移量
    offset = (page - 1) * size

    base_condition = ""
    if conditions:
        base_condition = " WHERE " + " AND ".join(conditions)

    # 计算总数（根据chat_id去重）
    count_sql = f"""
        SELECT COUNT(1) as count FROM (
            SELECT chat_id, MIN(id) as min_id 
            FROM t_user_qa_record 
            {base_condition}
            GROUP BY chat_id
        ) as distinct_chats
    """
    total_count_result = execute_sql_dict(count_sql)
    total_count = total_count_result[0]["count"] if total_count_result else 0
    total_pages = (total_count + size - 1) // size

    # 查询去重后的记录，只选择必要字段
    records_sql = f"""
        SELECT 
            t.uuid,
            t.question,
            t.chat_id,
            t.qa_type,
            t.datasource_id,
            d.name as datasource_name
        FROM t_user_qa_record t
        INNER JOIN (
            SELECT chat_id, MIN(id) as min_id 
            FROM t_user_qa_record 
            {base_condition}
            GROUP BY chat_id
        ) tm ON t.chat_id = tm.chat_id AND t.id = tm.min_id
        LEFT JOIN t_datasource d ON t.datasource_id = d.id
        ORDER BY t.id DESC 
        LIMIT {size} OFFSET {offset}
    """
    records = execute_sql_dict(records_sql)

    return PaginatedResponse(
        records=records,
        current_page=page,
        total_count=total_count,
        total_pages=total_pages,
    )


def query_user_qa_record(chat_id):
    """
    根据chat_id查询对话记录
    :param chat_id:
    :return:
    """
    with pool.get_session() as session:
        session: Session = session
        records = (
            session.query(TUserQaRecord)
            .filter(TUserQaRecord.chat_id == chat_id)
            .order_by(TUserQaRecord.id.desc())
            .all()
        )
        return model_to_dict(records)
    # sql = f"select * from t_user_qa_record where chat_id='{chat_id}' order by id desc limit 1"
    # return mysql_client.query_mysql_dict(sql)


async def get_record_sql(record_id: int, user_id: int) -> dict:
    """
    根据记录ID查询SQL语句
    :param record_id: 记录ID
    :param user_id: 用户ID（用于权限验证）
    :return: 包含SQL语句的字典，如果记录不存在或用户无权限则返回空字符串
    """
    try:
        with pool.get_session() as session:
            session: Session = session
            record = (
                session.query(TUserQaRecord)
                .filter(
                    TUserQaRecord.id == record_id,
                    TUserQaRecord.user_id == user_id
                )
                .first()
            )
            if record:
                return {"sql_statement": record.sql_statement or ""}
            else:
                return {"sql_statement": ""}
    except Exception as e:
        logger.error(f"查询记录SQL失败: {e}", exc_info=True)
        return {"sql_statement": ""}


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


async def query_user_list(page, size, name=None):
    """
    查询用户列表
    :param page: 页码
    :param size: 每页大小
    :param name: 用户名搜索
    :return:
    """
    with pool.get_session() as session:
        query = session.query(TUser)
        if name:
            query = query.filter(TUser.userName.like(f"%{name}%"))

        total_count = query.count()
        total_pages = (total_count + size - 1) // size

        users = query.order_by(TUser.createTime.desc()).offset((page - 1) * size).limit(size).all()

        user_list = []
        for user in users:
            u_dict = model_to_dict(user)
            # if u_dict.get("createTime"):
            #     u_dict["createTime"] = u_dict["createTime"].strftime("%Y-%m-%d %H:%M:%S")
            # if u_dict.get("updateTime"):
            #     u_dict["updateTime"] = u_dict["updateTime"].strftime("%Y-%m-%d %H:%M:%S")
            user_list.append(u_dict)

        return PaginatedResponse(
            records=user_list,
            current_page=page,
            total_count=total_count,
            total_pages=total_pages,
        )


async def add_user(username, password, mobile, role="user"):
    """
    添加用户
    :param username: 用户名
    :param password: 密码
    :param mobile: 手机号
    :param role: 角色
    :return:
    """
    with pool.get_session() as session:
        exist = session.query(TUser).filter(TUser.userName == username).first()
        if exist:
            raise MyException(SysCodeEnum.PARAM_ERROR, "用户名已存在")

        hashed_password = bcrypt.hashpw(password.encode('utf-8'), PASSWORD_SALT).decode('utf-8')

        new_user = TUser(
            userName=username,
            password=hashed_password,
            mobile=mobile,
            role=role,
            createTime=datetime.now(),
            updateTime=datetime.now(),
        )
        session.add(new_user)
        session.commit()
        return True


async def init_super_admin():
    """初始化超级管理员"""
    admin_name = "admin"
    # 默认密码，实际应从配置读取或更安全的方式
    admin_pass = "admin123" 
    try:
        with pool.get_session() as session:
            # 检查是否存在 admin 角色或名为 admin 的用户
            # 这里简单检查用户名
            exist = session.query(TUser).filter(TUser.userName == admin_name).first()
            if not exist:
                print(f"Initializing super admin: {admin_name}")
                hashed_password = bcrypt.hashpw(admin_pass.encode('utf-8'), PASSWORD_SALT).decode('utf-8')
                admin_user = TUser(
                    userName=admin_name,
                    password=hashed_password,
                    mobile="",
                    role="admin",
                    createTime=datetime.now(),
                    updateTime=datetime.now(),
                )
                session.add(admin_user)
                session.commit()
                print("Super admin initialized.")
            else:
                # 确保已存在的 admin 用户有 admin 角色
                if exist.role != 'admin':
                    exist.role = 'admin'
                    session.commit()
                    print("Updated existing user 'admin' to role 'admin'.")
                else:
                    print("Super admin already exists.")
    except Exception as e:
        print(f"Failed to init super admin: {e}")



async def update_user(user_id, username, mobile, password=None):
    """
    更新用户
    :param user_id: 用户ID
    :param username: 用户名
    :param mobile: 手机号
    :param password: 密码（可选）
    :return:
    """
    with pool.get_session() as session:
        user = session.query(TUser).filter(TUser.id == user_id).first()
        if not user:
            raise MyException(SysCodeEnum.PARAM_ERROR, "用户不存在")

        if user.userName != username:
            exist = session.query(TUser).filter(TUser.userName == username).first()
            if exist:
                raise MyException(SysCodeEnum.PARAM_ERROR, "用户名已存在")

        user.userName = username
        user.mobile = mobile
        if password:
            user.password = bcrypt.hashpw(password.encode('utf-8'), PASSWORD_SALT).decode('utf-8')
        user.updateTime = datetime.now()
        session.commit()
        return True


async def delete_user(user_id):
    """
    删除用户
    :param user_id: 用户ID
    :return:
    """
    with pool.get_session() as session:
        user = session.query(TUser).filter(TUser.id == user_id).first()
        if not user:
            raise MyException(SysCodeEnum.PARAM_ERROR, "用户不存在")
        session.delete(user)
        session.commit()
        return True
