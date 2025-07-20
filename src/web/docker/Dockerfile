
# 使用Nginx作为运行时的轻量级Web服务器
FROM nginx:stable-alpine

# 设置工作目录
WORKDIR /app
#
## 将package.json和pnpm-lock.yaml复制到工作目录
#COPY package.json ./
#
## 复制项目源代码到容器中
#COPY . .

# 删除默认的Nginx配置
RUN rm /etc/nginx/conf.d/default.conf

# # 添加自定义的Nginx配置
COPY ./docker/nginx.conf /etc/nginx/conf.d

# 将构建输出复制到Nginx的HTML目录
COPY dist /usr/share/nginx/html

# 暴露80端口
EXPOSE 80

# 启动Nginx服务
CMD ["nginx", "-g", "daemon off;"]