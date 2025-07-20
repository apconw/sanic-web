import json
import logging
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import re
import traceback
from typing import Dict, Any, List

from common.mysql_util import MysqlUtil


class ChartTypeEnum(Enum):
    """
    图表类别枚举
    """

    TABLE_CHART = ("response_table", "表格", "temp01")
    PIE_CHART = ("response_pie_chart", "饼图", "temp02")
    BAR_CHART = ("response_bar_chart", "柱状图", "temp03")
    LINE_CHART = ("response_line_chart", "折线图", "temp04")

    @classmethod
    def get_enum_by_code(cls, code):
        """

        :param code:
        :return:
        """
        if code == cls.TABLE_CHART.value[0]:
            return cls.TABLE_CHART
        elif code == cls.PIE_CHART.value[0]:
            return cls.PIE_CHART
        elif code == cls.BAR_CHART.value[0]:
            return cls.BAR_CHART
        elif code == cls.LINE_CHART.value[0]:
            return cls.LINE_CHART
        else:
            raise ValueError(f"Unsupported chart type: {code}")


def is_number(str_num):
    """

    :param str_num:
    :return:
    """
    if not str_num:  # 检查字符串是否为空或None
        return False
    try:
        float(str_num)  # 尝试转换为float
        return True
    except ValueError:  # 捕获转换失败的异常
        return False


# 匹配COUNT(*)*100的情况
count_pattern = r"\* (100(\.0)?)"


def format_value(type_enum, llm_info, key, value_str):
    """
        格式化样式
    :param type_enum:
    :param llm_info:
    :param key:
    :param value_str:
    :return:
    """
    if is_number(value_str):
        decimal = Decimal(value_str)
        if any(kw in key for kw in ["比例", "占比", "比率", "百分比", "概率"]):
            if re.search(count_pattern, llm_info.get("sql", "")):
                decimal = decimal.quantize(Decimal(".01"), rounding=ROUND_HALF_UP)
            else:
                decimal *= Decimal("100")
                decimal = decimal.quantize(Decimal(".01"), rounding=ROUND_HALF_UP)
            if type_enum in [ChartTypeEnum.TABLE_CHART, ChartTypeEnum.PIE_CHART]:
                return f"{decimal}%"
            else:
                return str(decimal)
        else:
            decimal = decimal.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        return str(decimal)
    return value_str or "0"


# 定义正则表达式模式，用于匹配包含中文字符的字符串
patternStr = ".*[\u4e00-\u9fa5]+.*"
pattern = re.compile(patternStr)


def process(data):
    """
        数据处理
    :param data:
    :return:
    """

    try:
        default_result = {
            "chart_type": ChartTypeEnum.TABLE_CHART,
            "template_code": ChartTypeEnum.TABLE_CHART.value[2],
            "data": [],
            "note": "数据来源: xxx数据库，以上数据仅供参考，具体情况可能会根据xx进一步调查和统计而有所变化",
        }

        if not data.strip():
            return default_result

        json_data = json.loads(data)
        llm_info = json_data.get("llm")
        chart_type = llm_info.get("type")
        type_enum = ChartTypeEnum.get_enum_by_code(chart_type)

        data_obj = json_data.get("data")
        if data_obj is None:
            return default_result

        chart_data = data_obj.get("result")
        column_data = data_obj.get("column")

        # 使用字典推导创建schema map schema_map = {item["column"]: item["comment"] for item in json_data.get("schema",
        # [{}])[0].get("schemaData", [])}

        # 处理图表类型
        if len(column_data) == 1:
            type_enum = ChartTypeEnum.TABLE_CHART
        elif type_enum == ChartTypeEnum.PIE_CHART and len(column_data) > 2:
            type_enum = ChartTypeEnum.TABLE_CHART
        elif type_enum == ChartTypeEnum.BAR_CHART:
            # 如果大模型返回柱状图且如果包含多列中文，则使用表格展示
            if chart_data:
                json_object = chart_data[0]
                if json_object is not None:
                    count = sum(1 for key in json_object.keys() if (item := json_object.get(key)) and pattern.match(str(item)))
                    if count > 1:
                        type_enum = ChartTypeEnum.TABLE_CHART

            # 如果 columnData 列表中的元素数量大于3，则更改图表类型为表格
            if len(column_data) > 3:
                type_enum = ChartTypeEnum.TABLE_CHART
        elif type_enum == ChartTypeEnum.LINE_CHART and len(column_data) > 2:
            type_enum = ChartTypeEnum.TABLE_CHART

        handlers = {
            ChartTypeEnum.TABLE_CHART: process_table_chart,
            ChartTypeEnum.PIE_CHART: process_pie_chart,
            ChartTypeEnum.BAR_CHART: process_bar_chart,
            ChartTypeEnum.LINE_CHART: process_line_chart,
        }

        handler = handlers.get(type_enum)
        if handler:
            processed_data = handler(llm_info, column_data, chart_data)
            return {"chart_type": type_enum.value[1], "template_code": type_enum.value[2], "data": processed_data, "note": default_result["note"]}
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")
    except Exception as e:
        logging.error(f"Error processing data: {e}")
        traceback.print_exception(e)


def process_table_chart(llm_info, column_data, chart_data):
    """
        表格数据处理
    :param llm_info:
    :param column_data:
    :param chart_data:
    :return:
    """
    # 简化处理，实际可能需要根据具体需求调整
    return [dict((col, format_value(ChartTypeEnum.TABLE_CHART, llm_info, col, data.get(col, ""))) for col in column_data) for data in chart_data]


def process_pie_chart(llm_info: dict, column_data: list, chart_data: list) -> List[dict]:
    """
    饼图数据处理
    :param llm_info:
    :param column_data:
    :param chart_data:
    :return:
    """
    pie_data_list = []
    for data_map in chart_data:
        pie_data = {}
        for key, value in data_map.items():
            if key == column_data[0]:
                pie_data["name"] = "" if value is None else value
            else:
                pie_value = format_value(ChartTypeEnum.PIE_CHART, llm_info, key, value)
                if pie_value and "%" in pie_value:
                    pie_data["value"] = pie_value.split("%")[0]
                    pie_data["percent"] = True
                else:
                    pie_data["value"] = pie_value
                    pie_data["percent"] = False
        pie_data_list.append(pie_data)

    return pie_data_list


def is_numeric(value: str) -> bool:
    """
    判断是否是数字
    :param value:
    :return:
    """
    try:
        float(value)
        return True
    except ValueError:
        return False


def is_valid_date(value: str) -> bool:
    """
    判断是否为日期
    :param value:
    :return:
    """
    # @todo支持更多的时间格式
    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    return bool(date_pattern.match(value))


def process_bar_chart(llm_info: Dict[str, Any], column_data: List[str], chart_data: List[Dict[str, Any]]) -> List[List[Any]]:
    """
    柱状图
    :param llm_info:
    :param column_data:
    :param chart_data:
    :return:
    """
    data_list = []

    if chart_data:
        column_array = column_data.copy()
        column_array[0] = "product"
        pattern_str = r"^[\u4e00-\u9fa5]+$"  # Pattern to match Chinese characters

        for k in range(1, len(column_array)):
            column_str = column_array[k]
            if column_str and not re.match(pattern_str, column_str):
                first_item = chart_data[0]
                for key in first_item.keys():
                    item_value = first_item[key]
                    if is_numeric(item_value):
                        column_array[k] = "数量"
                    elif is_valid_date(item_value):
                        column_array[k] = "日期"

        data_list.append(column_array)

        for item in chart_data:
            item_data = []
            for column_key in column_data:
                value = item.get(column_key, "")
                if not is_numeric(value):
                    item_data.append("未知" if not value else value)
                else:
                    item_data.append(format_value(ChartTypeEnum.BAR_CHART, llm_info, column_key, value))
            data_list.append(item_data)

    return data_list


def process_line_chart(llm_info: dict, column_data: list, chart_data: list) -> List[list]:
    """
    折线图
    :param llm_info:
    :param chart_data:
    :param column_data:
    :return:
    """
    data_list = []

    if chart_data:
        item_date = []
        item_value = []
        for result in chart_data:
            for key, value in result.items():
                # 查询列第一个则是折线的x轴
                if key == column_data[0]:
                    item_date.append(value)
                else:
                    item_value.append(format_value(ChartTypeEnum.LINE_CHART, llm_info, key, value))
        data_list.append(item_date)
        data_list.append(item_value)

    return data_list


async def select_report_by_title(title: str) -> str:
    """
    报告查询
    :param title:
    :return:
    """
    sql = f"""select markdown from t_report_info where title like '%{title}%'  order by create_time desc limit 1"""
    report_dict = MysqlUtil().query_mysql_dict(sql)
    if len(report_dict) > 0:
        return report_dict[0]["markdown"].replace("```markdown", "").replace("```", "")
    else:
        return ""
