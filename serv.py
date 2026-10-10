import os

# 避免本地向量/Embedding 库重复初始化 OpenMP 导致崩溃或卡死
# 参考错误: "OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized."
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from sanic import Sanic
from sanic.response import empty
from sanic.worker.manager import WorkerManager

# 设置 worker 启动超时时间（单位：0.1秒）
# 设置为 180 秒（1800 * 0.1秒），允许 workers 有足够时间完成启动
startup_timeout_seconds = int(os.getenv("SANIC_WORKER_STARTUP_TIMEOUT", 180))
WorkerManager.THRESHOLD = startup_timeout_seconds * 10  # 转换为 0.1 秒单位

import controllers
from common.route_utility import autodiscover
from config.load_env import load_env

# 加载配置文件
load_env()

# 校验必需的安全配置：JWT_SECRET_KEY 必须显式设置，不允许使用内置默认值
# 在此处（进程启动、worker fork 之前）校验一次即可覆盖所有 worker 进程
if not os.getenv("JWT_SECRET_KEY"):
    raise RuntimeError(
        "JWT_SECRET_KEY environment variable is not set. Set it before starting "
        "the server (see README for how to generate and configure one)."
    )

# 确保日志配置在 Sanic 启动前已正确加载
import logging

root_logger = logging.getLogger()
if not root_logger.handlers:
    # 如果 root logger 没有 handlers，说明配置加载失败，使用备用配置
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s | %(asctime)s | [PID:%(process)d] | %(filename)s:%(lineno)d | %(funcName)s() | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

app = Sanic("Aix-DB", configure_logging=False)


# 确保在每个 worker 启动时都重新加载日志配置
@app.before_server_start
async def ensure_logging_config(app, loop):
    """
    在每个 worker 启动时确保日志配置正确
    Sanic 使用多进程模式时，每个 worker 都会重新加载代码，需要确保日志配置正确
    """
    import logging

    from config.load_env import load_env

    # 重新加载日志配置（如果被覆盖了）
    root_logger = logging.getLogger()

    # 如果 root logger 没有 handlers 或者级别不对，重新加载配置
    if not root_logger.handlers or root_logger.level > logging.INFO:
        # 重新加载日志配置
        load_env()

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        logging.info(
            "✅ [SERV] Logging configuration reloaded in worker - handlers: %d, level: %s",
            len(root_logger.handlers),
            root_logger.level,
        )


@app.main_process_start
async def init_minio(app, loop):
    """
    在主进程启动时初始化 MinIO bucket（只执行一次）
    """
    import logging

    logger = logging.getLogger(__name__)

    # 获取默认 bucket 名称
    default_bucket = os.getenv("MINIO_DEFAULT_BUCKET", "filedata")

    try:
        from common.minio_util import MinioUtils

        minio_utils = MinioUtils()
        minio_utils.ensure_bucket(default_bucket)
        logger.info(f"✅ [SERV] MinIO bucket '{default_bucket}' initialized successfully")
    except Exception as e:
        # MinIO 初始化失败不阻止服务启动，只记录警告
        logger.warning(f"⚠️ [SERV] MinIO initialization failed: {e}. File upload features may not work.")


autodiscover(
    app,
    controllers,
    recursive=True,
)

# 设置 worker 状态 TTL，优先使用环境变量
app.config.SANIC_WORKER_STATE_TTL = int(os.getenv("SANIC_WORKER_STATE_TTL", 120))

# SSE 流式响应超时设置（DeepAgent 报告生成可能耗时较长）
# 超时链路：LLM(15min) < TASK(30min) < RESPONSE_TIMEOUT(35min) < 前端 fetch(36min)
app.config.RESPONSE_TIMEOUT = int(os.getenv("SANIC_RESPONSE_TIMEOUT", 2100))  # 35分钟
app.config.REQUEST_TIMEOUT = int(os.getenv("SANIC_REQUEST_TIMEOUT", 300))  # 5分钟
app.config.KEEP_ALIVE_TIMEOUT = int(os.getenv("SANIC_KEEP_ALIVE_TIMEOUT", 120))  # 2分钟

# 添加api docs
app.extend(
    config={
        "OAS_PATH_TO_REDOC_HTML": "docs/redoc.html",
        "OAS_PATH_TO_SWAGGER_HTML": "docs/swagger.html",
        "OAS_UI_DEFAULT": "swagger",
        "OAS_VERSION": "3.1.0",
        "OAS_TITLE": "Aix-DB API",
        "OAS_DESCRIPTION": "Aix-DB API 接口文档",
        "OAS_VERSION_STRING": "1.0.0",
    }
)

app.route("/")(lambda _: empty())


def get_server_config():
    """获取服务器配置参数"""
    workers = int(os.getenv("SERVER_WORKERS", 2))
    config = {
        "host": os.getenv("SERVER_HOST", "0.0.0.0"),
        "port": int(os.getenv("SERVER_PORT", 8088)),
        "auto_reload": False,
    }
    if workers == 1:
        # 单进程模式：开发环境推荐，PyCharm 可正确管理进程生命周期
        config["single_process"] = True
    else:
        config["workers"] = workers
    return config


if __name__ == "__main__":
    config = get_server_config()
    app.run(**config)
