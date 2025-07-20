import os
from datetime import datetime

import jwt
from sanic import response


def check_token(f):
    """
    jwt token 校验注解
    """

    async def wrapper(request, *args, **kwargs):
        token = request.headers.get("Authorization")
        if not token:
            return response.json({"message": "无效Token", "code": 401}, status=401)
        try:
            # 去掉 Bearer 前缀（如果有的话）
            if token.startswith("Bearer "):
                token = token.split(" ")[1]

            # 解码 JWT token
            payload = jwt.decode(token, key=os.getenv("JWT_SECRET_KEY"), algorithms=["HS256"])
            # 检查 token 是否过期
            if "exp" in payload and datetime.utcfromtimestamp(payload["exp"]) < datetime.utcnow():
                return response.json({"message": "Token已过期", "code": 401}, status=401)

            request.ctx.user_payload = payload
        except jwt.ExpiredSignatureError as e:
            return response.json({"message": "Token已过期", "code": 401}, status=401)
        except Exception as e:
            return response.json({"message": "无效Token", "code": 401}, status=401)

        # 继续处理请求
        return await f(request, *args, **kwargs)

    return wrapper
