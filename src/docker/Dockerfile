# 构建阶段
FROM python:3.10-slim-bookworm AS builder
WORKDIR /sanic-web

# 设置清华源 + 添加 PATH
ENV PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple" \
    PIP_EXTRA_INDEX_URL="https://pypi.doubanio.com/simple" \
    UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple" \
    PATH="/root/.local/bin:${PATH}"

# 复制项目文件
COPY . .

# 安装依赖工具并安装依赖
RUN set -eux; \
    apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    pip install --no-cache-dir pipx && \
    pipx install "uv==0.8.0" && \
    uv venv --clear && \
    . .venv/bin/activate && \
    uv sync --no-cache && \
    apt-get purge -y gcc python3-dev && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# 最终运行阶段
FROM python:3.10-slim-bookworm
WORKDIR /sanic-web

# 复制构建结果
COPY --from=builder /sanic-web /sanic-web

# 设置环境变量
ENV PATH="/sanic-web/.venv/bin:${PATH}"


# 暴露端口
EXPOSE 8088

# 启动服务
CMD [".venv/bin/python","serv.py"]