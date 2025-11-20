from enum import Enum


class SysCodeEnum(Enum):
    """
    系统状态码定义
    """

    c_200 = (200, "ok", "ok")

    c_401 = (401, "登录异常", "登录异常")

    c_400 = (401, "无效Token", "无效Token")

    c_9999 = (9999, "系统异常", "系统异常")


class DiFyAppEnum(Enum):
    """
    DiFy app-key 枚举
    """

    DATABASE_QA = ("DATABASE_QA", "数据问答")

    FILEDATA_QA = ("FILEDATA_QA", "表格问答")

    COMMON_QA = ("COMMON_QA", "智能问答")

    REPORT_QA = ("REPORT_QA", "深度搜索")


def get_qatype_name(member_name):
    """
    根据DiFyAppEnum枚举成员名称获取对应的中文名称

    :param member_name: 枚举成员名称字符串，如 "REPORT_QA"
    :return: 对应的中文名称字符串
    """
    try:
        enum_member = DiFyAppEnum[member_name]
        return enum_member.value[1]
    except KeyError:
        raise ValueError(f"'{member_name}' 不是有效的DiFyAppEnum枚举成员")


class DataTypeEnum(Enum):
    """
    自定义数据类型枚举
    """

    ANSWER = ("t02", "答案")

    LOCATION = ("t03", "溯源")

    BUS_DATA = ("t04", "业务数据")

    TASK_ID = ("t11", "任务ID,方便后续点赞等操作")

    STREAM_END = ("t99", "流式推流结束")


class DiFyCodeEnum(Enum):
    """
    DiFy 返回数据流定义
    """

    MESSAGE = ("message", "答案")

    MESSAGE_END = ("message_end", "结束")

    MESSAGE_ERROR = ("error", "错误")
