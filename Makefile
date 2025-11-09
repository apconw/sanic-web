# 导入子模块web的 Makefile
include web/Makefile

# 服务端项目名称
SERVER_PROJECT_NAME = sanic-web

# 服务端 Docker 镜像标签
SERVER_DOCKER_IMAGE = apconw/$(SERVER_PROJECT_NAME):1.2.0

# 阿里云镜像仓库地址 (需要根据实际情况修改)
ALIYUN_REGISTRY = crpi-7xkxsdc0iki61l0q.cn-hangzhou.personal.cr.aliyuncs.com
ALIYUN_NAMESPACE = apconw
ALIYUN_IMAGE_NAME = $(ALIYUN_REGISTRY)/$(ALIYUN_NAMESPACE)/$(SERVER_PROJECT_NAME):1.2.0

# 构建 Vue 3 前端项目镜像
web-build:
	$(MAKE) -C web docker-build

# 构建服务端镜像
service-build:
	docker build --no-cache -t $(SERVER_DOCKER_IMAGE) -f ./docker/Dockerfile .


# 构建 服务端arm64/amd64架构镜像并推送docker-hub
docker-build-server-multi:
	docker buildx build --platform linux/amd64,linux/arm64 --push -t $(SERVER_DOCKER_IMAGE) -f ./docker/Dockerfile .


# 构建服务端arm64/amd64架构镜像并推送至阿里云镜像仓库
docker-build-aliyun-server-multi:
	docker buildx build --platform linux/amd64,linux/arm64 --push -t $(ALIYUN_IMAGE_NAME) -f ./docker/Dockerfile .

.PHONY: web-build service-build