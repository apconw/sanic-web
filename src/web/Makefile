# 项目名称
PROJECT_NAME = chat-vue3-mvp

# Docker 镜像标签
DOCKER_IMAGE = apconw/$(PROJECT_NAME):1.1.4

# Docker 容器名称
DOCKER_WEB_CONTAINER = $(PROJECT_NAME)

# Node.js 版本
NODE_VERSION = 18-alpine

# Nginx 配置文件路径
NGINX_CONF = ./docker/nginx.conf

# 构建 Vue 3 项目
build:
	npm run build

# 构建 Docker 镜像
docker-build: clean build
	docker build -t $(DOCKER_IMAGE) -f ./docker/Dockerfile .

# 构建 前端arm64/amd64架构镜像并推送
docker-build-web-multi: clean build
	docker buildx build --platform linux/amd64,linux/arm64 --push -t $(DOCKER_IMAGE) -f ./docker/Dockerfile .

# 清理构建产物
clean:
	rm -rf dist

.PHONY:  build docker-build