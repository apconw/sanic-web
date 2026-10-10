import logging
import os
from datetime import datetime
from functools import wraps

import jwt
from sanic import response

logger = logging.getLogger(__name__)


def check_token(f):
    """
    jwt token 校验注解
    """

    @wraps(f)
    async def wrapper(request, *args, **kwargs):
        token = request.headers.get("Authorization")
        if not token:
            return response.json({"message": "无效Token", "code": 401}, status=401)
        try:
            # 去掉 Bearer 前缀（如果有的话）
            if token.startswith("Bearer "):
                token = token.split(" ")[1]

            jwt_secret_key = os.getenv("JWT_SECRET_KEY")
            if not jwt_secret_key:
                raise RuntimeError("JWT_SECRET_KEY environment variable is not set")

            # 解码 JWT token
            payload = jwt.decode(token, key=jwt_secret_key, algorithms=["HS256"])
            # 检查 token 是否过期
            if "exp" in payload and datetime.utcfromtimestamp(payload["exp"]) < datetime.utcnow():
                return response.json({"message": "Token已过期", "code": 401}, status=401)

            request.ctx.user_payload = payload
        except jwt.ExpiredSignatureError as e:
            return response.json({"message": "Token已过期", "code": 401}, status=401)
        except RuntimeError as e:
            # 服务端配置缺失（JWT_SECRET_KEY 未设置），不能当作客户端的无效 Token 处理
            logger.critical(f"JWT auth misconfigured: {e}")
            return response.json({"message": "服务器配置错误", "code": 500}, status=500)
        except Exception as e:
            return response.json({"message": "无效Token", "code": 401}, status=401)

        # 继续处理请求
        return await f(request, *args, **kwargs)

    return wrapper
