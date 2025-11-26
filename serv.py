import os

from sanic import Sanic
from sanic.response import empty

import controllers
from common.route_utility import autodiscover
from config.load_env import load_env

# 加载配置文件
load_env()

app = Sanic("sanic-web")
autodiscover(
    app,
    controllers,
    recursive=True,
)
# 添加api docs
app.extend(config={"OAS_PATH_TO_REDOC_HTML": "docs/redoc.html", "OAS_PATH_TO_SWAGGER_HTML": "docs/swagger.html"})

app.route("/")(lambda _: empty())


def get_server_config():
    """获取服务器配置参数"""
    return {
        "host": os.getenv("SERVER_HOST", "0.0.0.0"),
        "port": int(os.getenv("SERVER_PORT", 8088)),
        "workers": int(os.getenv("SERVER_WORKERS", 2)),
    }


if __name__ == "__main__":
    config = get_server_config()
    app.run(**config)
