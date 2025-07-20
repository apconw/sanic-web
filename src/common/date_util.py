import decimal
import json
from datetime import date, datetime


class DateEncoder(json.JSONEncoder):
    """
    日期格式
    """

    def default(self, obj):
        """

        :param obj:
        :return:
        """
        # 处理返回数据中有date类型的数据
        if isinstance(obj, date):
            return obj.strftime("%Y-%m-%d")
        # 处理返回数据中有datetime类型的数据
        elif isinstance(obj, datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        # 处理返回数据中有decimal类型的数据
        elif isinstance(obj, decimal.Decimal):
            return float(obj)
        else:
            return json.JSONEncoder.default(self, obj)
