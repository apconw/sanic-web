from typing import Any

import pandas as pd
from pandasai.responses.response_parser import ResponseParser
from pandasai.responses.response_serializer import ResponseSerializer


class PandasaiCustomResponse(ResponseParser):
    """
    自定义返回值解析类型
    """

    def _init__(self, context) -> None:
        super().__init__(context)

    def parse(self, result: dict) -> Any:
        """

        :param result:
        :return:
        """
        if not isinstance(result, dict) or any(
            key not in result for key in ["type", "value"]
        ):
            raise ValueError("Unsupported result format")

        return ResponseSerializer.serialize(result)
