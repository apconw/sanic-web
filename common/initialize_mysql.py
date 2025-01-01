import pymysql
import os

from pymysql import MySQLError

"""
Mysql 初始化脚本工具类
"""

# 配置信息
MYSQL_ROOT_PASSWORD = "1"  # MySQL root 用户的密码
SQL_FILE = "../docker/init_sql.sql"  # SQL 文件路径
HOST = "localhost"  # MySQL 服务器地址
PORT = 13006  # MySQL 服务器端口


def check_sql_file(file_path***REMOVED***:
    """
    检查 SQL 文件是否存在
    :param file_path:
    :return:
    """
    if not os.path.isfile(file_path***REMOVED***:
        raise FileNotFoundError(f"Error: SQL file {file_path***REMOVED*** not found."***REMOVED***


def execute_sql_file(file_path***REMOVED***:
    """
    执行 SQL 文件
    :param file_path:
    :return:
    """
    try:
        # 创建数据库连接
        connection = pymysql.connect(
            host=HOST, user="root", password=MYSQL_ROOT_PASSWORD, port=PORT, charset="utf8mb4", cursorclass=pymysql.cursors.DictCursor
        ***REMOVED***
        with connection.cursor(***REMOVED*** as cursor:
            print(f"Initializing MySQL with {file_path***REMOVED*** on port {PORT***REMOVED***..."***REMOVED***

            # 读取 SQL 文件
            with open(file_path, "r", encoding="utf-8"***REMOVED*** as file:
                sql_script = file.read(***REMOVED***

            # 分割 SQL 命令并执行
            commands = sql_script.split(";"***REMOVED***
            for command in commands:
                if command.strip(***REMOVED***:  # 忽略空命令
                    cursor.execute(command***REMOVED***

            # 提交事务
            connection.commit(***REMOVED***
            print("MySQL initialization completed successfully."***REMOVED***
    except pymysql.MySQLError as e:
        print(f"Error: {e***REMOVED***"***REMOVED***
    finally:
        if connection.open:
            connection.close(***REMOVED***


def execute_user_qa_record_sql(***REMOVED***:
    """
    使用 pymysql 连接数据库并执行 SQL 语句。
    初始化特殊结构json数据
    """

    sql_insert_data = """
    INSERT INTO chat_db.t_user_qa_record
    (id, user_id, conversation_id,message_id,task_id,chat_id, question, to2_answer, to4_answer, qa_type,file_key,create_time***REMOVED***
    VALUES(%s, %s, %s,%s, %s, %s, %s, %s, %s, %s,%s,%s***REMOVED***;
    """

    data_to_insert = [
        (
            301,
            1,
            "35be68e9-c7f7-4b36-862b-70efa54cc220",
            "4d838aed-bad5-4fff-8807-fb1b4a9108e2",
            "92c98e0d-1cc0-4121-84d0-97526117b401",
            "7b549d5e-8a71-4312-837e-68d75151d72f",
            "统计案件数量按分局分组柱状图",
            '{"data": {"messageType": "continue", "content": "从数据可以看出，诈骗案件数量呈现出一定的地域差异。徐汇分局的诈骗案件最多，共计10宗；其次是浦东分局、朝阳分局和天河分局，分别有7宗、4宗和4宗。福田分局、海淀分局、白云分局各有2至3宗不等的诈骗案件。而武侯分局和锦江分局较少，仅有1宗。整体来看，部分区域如徐汇分局的诈骗案件数量较多，其他区域则相对较少。详情如下图:"***REMOVED***, "dataType": "t02"***REMOVED***',
            '{"data": {"chart_type": "柱状图", "template_code": "temp03", "data": [["product", "总数"], ["浦东分局", "7"], ["朝阳分局", "4"], ["天河分局", "4"], ["南山分局", "3"], ["武侯分局", "1"], ["锦江分局", "2"], ["徐汇分局", "10"], ["海淀分局", "4"], ["白云分局", "3"], ["福田分局", "2"]], "note": "数据来源: xxx数据库，以上数据仅供参考，具体情况可能会根据xx进一步调查和统计而有所变化"***REMOVED***, "dataType": "t04"***REMOVED***',
            "DATABASE_QA",
            "",
            "2025-01-01 08:54:40",
        ***REMOVED***,
        (
            302,
            1,
            "8a929e01-e085-41ee-a875-02b5fbd0f93b",
            "7524b3e5-a1be-42a8-b421-4fe26ca929bb",
            "3060905a-da43-4e4d-baa2-56c632b12cf7",
            "213f5c32-dd38-479e-b3e4-91600392c73e",
            "诈骗金额最高的前十条数据",
            '{"data": {"messageType": "continue", "content": "共计10宗诈骗案件。从时间上看，案件集中在2024年6月至12月期间，且有3宗发生在上海市徐汇区。诈骗类型主要为冒充公检法及投资理财，其中冒充公检法占一半以上。受害人多为经理或工程师职业，多数年龄在30至40岁之间。涉案资金以1万至2万元为主。反诈宣传方面，大多数受害者表示看过相关宣传。详情如下图:"***REMOVED***, "dataType": "t02"***REMOVED***',
            '{"data": {"chart_type": "表格", "template_code": "temp01", "data": [{"案发地点": "上海市徐汇区龙阳路", "所属分局": "徐汇分局", "来电类别": "电话报警", "报警人姓名": "黄梅", "性别": "女", "年龄": "38", "文化程度": "本科", "受害人职业": "经理", "手机品牌": "华为", "涉案资金": "21000", "是否电诈": "是", "诈骗类型": "冒充公检法", "引流方式": "电话", "引流地址": "8613700004444", "引流联系时间": "2024-06-03 18:00:00", "是否看过反诈宣传": "是", "登记时间": "2024-06-15 18:30:00"***REMOVED***, {"案发地点": "深圳市福田区福华路", "所属分局": "福田分局", "来电类别": "电话报警", "报警人姓名": "黄梅", "性别": "女", "年龄": "38", "文化程度": "本科", "受害人职业": "经理", "手机品牌": "华为", "涉案资金": "21000", "是否电诈": "是", "诈骗类型": "冒充公检法", "引流方式": "电话", "引流地址": "8613700004444", "引流联系时间": "2024-09-01 18:00:00", "是否看过反诈宣传": "是", "登记时间": "2024-09-05 18:30:00"***REMOVED***, {"案发地点": "上海市徐汇区龙阳路", "所属分局": "徐汇分局", "来电类别": "电话报警", "报警人姓名": "黄梅", "性别": "女", "年龄": "38", "文化程度": "本科", "受害人职业": "经理", "手机品牌": "华为", "涉案资金": "21000", "是否电诈": "是", "诈骗类型": "冒充公检法", "引流方式": "电话", "引流地址": "8613700004444", "引流联系时间": "2024-12-03 18:00:00", "是否看过反诈宣传": "是", "登记时间": "2024-12-15 18:30:00"***REMOVED***, {"案发地点": "北京市海淀区学院路", "所属分局": "海淀分局", "来电类别": "电话报警", "报警人姓名": "周涛", "性别": "男", "年龄": "31", "文化程度": "大专", "受害人职业": "工程师", "手机品牌": "VIVO", "涉案资金": "16000", "是否电诈": "是", "诈骗类型": "投资理财", "引流方式": "电话", "引流地址": "8613700002222", "引流联系时间": "2024-07-04 14:00:00", "是否看过反诈宣传": "是", "登记时间": "2024-07-20 14:30:00"***REMOVED***, {"案发地点": "成都市武侯区武侯祠大街", "所属分局": "武侯分局", "来电类别": "电话报警", "报警人姓名": "周涛", "性别": "男", "年龄": "31", "文化程度": "大专", "受害人职业": "工程师", "手机品牌": "VIVO", "涉案资金": "16000", "是否电诈": "是", "诈骗类型": "投资理财", "引流方式": "电话", "引流地址": "8613700002222", "引流联系时间": "2024-04-02 14:00:00", "是否看过反诈宣传": "是", "登记时间": "2024-04-10 14:30:00"***REMOVED***, {"案发地点": "上海市浦东新区花木路", "所属分局": "浦东分局", "来电类别": "电话报警", "报警人姓名": "周杰", "性别": "男", "年龄": "30", "文化程度": "大专", "受害人职业": "工程师", "手机品牌": "VIVO", "涉案资金": "15000", "是否电诈": "是", "诈骗类型": "投资理财", "引流方式": "电话", "引流地址": "8613700002222", "引流联系时间": "2024-02-02 14:00:00", "是否看过反诈宣传": "是", "登记时间": "2024-02-10 14:30:00"***REMOVED***, {"案发地点": "上海市徐汇区漕溪路", "所属分局": "徐汇分局", "来电类别": "电话报警", "报警人姓名": "张明", "性别": "男", "年龄": "36", "文化程度": "硕士", "受害人职业": "产品经理", "手机品牌": "华为", "涉案资金": "13000", "是否电诈": "是", "诈骗类型": "冒充亲友", "引流方式": "电话", "引流地址": "8613700003333", "引流联系时间": "2024-12-01 16:00:00", "是否看过反诈宣传": "是", "登记时间": "2024-12-05 16:30:00"***REMOVED***, {"案发地点": "广州市白云区金沙洲", "所属分局": "白云分局", "来电类别": "电话报警", "报警人姓名": "张明", "性别": "男", "年龄": "36", "文化程度": "硕士", "受害人职业": "产品经理", "手机品牌": "华为", "涉案资金": "13000", "是否电诈": "是", "诈骗类型": "冒充亲友", "引流方式": "电话", "引流地址": "8613700003333", "引流联系时间": "2024-08-02 16:00:00", "是否看过反诈宣传": "是", "登记时间": "2024-08-10 16:30:00"***REMOVED***, {"案发地点": "上海市徐汇区漕溪路", "所属分局": "徐汇分局", "来电类别": "电话报警", "报警人姓名": "张明", "性别": "男", "年龄": "36", "文化程度": "硕士", "受害人职业": "产品经理", "手机品牌": "华为", "涉案资金": "13000", "是否电诈": "是", "诈骗类型": "冒充亲友", "引流方式": "电话", "引流地址": "8613700003333", "引流联系时间": "2024-06-01 16:00:00", "是否看过反诈宣传": "是", "登记时间": "2024-06-05 16:30:00"***REMOVED***, {"案发地点": "广州市天河区珠江新城", "所属分局": "天河分局", "来电类别": "现场报警", "报警人姓名": "李娜", "性别": "女", "年龄": "28", "文化程度": "硕士", "受害人职业": "设计师", "手机品牌": "苹果", "涉案资金": "12000", "是否电诈": "是", "诈骗类型": "冒充公检法", "引流方式": "电话", "引流地址": "8613800138000", "引流联系时间": "2024-02-04 11:00:00", "是否看过反诈宣传": "否", "登记时间": "2024-02-20 11:30:00"***REMOVED***], "note": "数据来源: xxx数据库，以上数据仅供参考，具体情况可能会根据xx进一步调查和统计而有所变化"***REMOVED***, "dataType": "t04"***REMOVED***',
            "DATABASE_QA",
            "",
            "2025-01-01 08:55:36",
        ***REMOVED***,
        (
            303,
            1,
            "8743388c-ba20-4cb7-b4d6-a7a8a48d552a",
            "8e222e16-ec3c-4e9e-9d9a-b64550f798e8",
            "69d3c422-c240-4501-aa38-f7d2487d5a74",
            "dcf58bbd-a057-46b5-8a59-2e84a1bcfd1d",
            "统计案件数量按分局分组饼图",
            '{"data": {"messageType": "continue", "content": "浦东分局的诈骗案件数量最多，为7宗；徐汇分局次之，有10宗。其他分局如朝阳分局、天河分局等均为4宗或更少。福田分局和锦江分局分别为2宗，南山分局、武侯分局、海淀分局、白云分局和福田分局均为3宗或更少。整体来看，案件数量分布不均，浦东分局和徐汇分局较为突出。详情如下图:"***REMOVED***, "dataType": "t02"***REMOVED***',
            '{"data": {"chart_type": "饼图", "template_code": "temp02", "data": [{"name": "浦东分局", "value": "7", "percent": false***REMOVED***, {"name": "朝阳分局", "value": "4", "percent": false***REMOVED***, {"name": "天河分局", "value": "4", "percent": false***REMOVED***, {"name": "南山分局", "value": "3", "percent": false***REMOVED***, {"name": "武侯分局", "value": "1", "percent": false***REMOVED***, {"name": "锦江分局", "value": "2", "percent": false***REMOVED***, {"name": "徐汇分局", "value": "10", "percent": false***REMOVED***, {"name": "海淀分局", "value": "4", "percent": false***REMOVED***, {"name": "白云分局", "value": "3", "percent": false***REMOVED***, {"name": "福田分局", "value": "2", "percent": false***REMOVED***], "note": "数据来源: xxx数据库，以上数据仅供参考，具体情况可能会根据xx进一步调查和统计而有所变化"***REMOVED***, "dataType": "t04"***REMOVED***',
            "DATABASE_QA",
            "",
            "2025-01-01 08:56:11",
        ***REMOVED***,
        (
            304,
            1,
            "cee6f9c9-e6da-4747-8b5c-f7ac78e48d34",
            "fd2ea932-2b5b-4751-a886-97f44f3bb560",
            "32aea8c4-77ae-41bc-a4a6-f5c46303b94b",
            "e666f35b-7678-4626-a41a-01a04fbf689b",
            "统计案件数量按月分组折线图",
            '{"data": {"messageType": "continue", "content": "从数据可以看出，诈骗案件数量呈现先增后减再增的趋势。具体来说，2024年1月至6月案件数量逐渐减少至最低点1起（2024-10），之后在11月和12月又回升至5起。整体来看，案件数量波动较大但总体呈上升趋势。详情如下图:"***REMOVED***, "dataType": "t02"***REMOVED***',
            '{"data": {"chart_type": "折线图", "template_code": "temp04", "data": [["2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06", "2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12"], ["5", "4", "3", "2", "1", "5", "4", "3", "2", "1", "5", "5"]], "note": "数据来源: xxx数据库，以上数据仅供参考，具体情况可能会根据xx进一步调查和统计而有所变化"***REMOVED***, "dataType": "t04"***REMOVED***',
            "DATABASE_QA",
            "",
            "2025-01-01 08:56:45",
        ***REMOVED***,
***REMOVED***

    # 创建数据库连接
    connection = pymysql.connect(
        host=HOST, user="root", password=MYSQL_ROOT_PASSWORD, port=PORT, db="chat_db", charset="utf8mb4", cursorclass=pymysql.cursors.DictCursor
    ***REMOVED***

    try:
        with connection.cursor(***REMOVED*** as cursor:
            # 使用 executemany 方法插入多条数据
            cursor.executemany(sql_insert_data, data_to_insert***REMOVED***
        # 提交事务
        connection.commit(***REMOVED***
    except MySQLError as e:
        print(f"Error executing query: {e***REMOVED***"***REMOVED***
        # 如果出现错误，则回滚事务
        if connection.open:
            connection.rollback(***REMOVED***
    finally:
        # 关闭数据库连接
        connection.close(***REMOVED***


if __name__ == "__main__":
    check_sql_file(SQL_FILE***REMOVED***
    execute_sql_file(SQL_FILE***REMOVED***
    execute_user_qa_record_sql(***REMOVED***
