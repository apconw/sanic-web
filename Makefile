# 导入子模块web的 Makefile
include src/web/Makefile

# 服务端项目名称
SERVER_PROJECT_NAME = sanic-web

# 服务端 Docker 镜像标签
SERVER_DOCKER_IMAGE = apconw/$(SERVER_PROJECT_NAME):1.1.4


# 构建 Vue 3 前端项目镜像
web-build:
	$(MAKE) -C src/web docker-build

# 构建服务端镜像
service-build:
		docker build --no-cache -t $(SERVER_DOCKER_IMAGE) -f ./src/docker/Dockerfile .


# 构建 服务端arm64/amd64架构镜像并推送
docker-build-server-multi:
	docker buildx build --platform linux/amd64,linux/arm64 --push -t $(SERVER_DOCKER_IMAGE) -f ./src/docker/Dockerfile .


.PHONY: web-build service-build