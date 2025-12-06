
确保上一步[环境配置](environment.md)已配置好



## 1. 后端依赖安装  
   - uv安装 [参考uv官方文档](https://docs.astral.sh/uv/getting-started/installation/)
```bash
# 安装uv
curl -LsSf https://astral.sh/uv/install.sh | sh
   
#进入项目目录
cd sanic-web

# 创建虚拟环境
uv venv --clear

# 激活虚拟环境
   
# Mac or Linux 用户执行
source .venv/bin/activate

# Windows 用户执行
.venv\Scripts\activate
   
# 安装依赖
uv sync --no-cache
   
# pycharm 配置虚拟环境
Settings -> Project: sanic-web -> Project Interpreter -> Add -> Existing environment
选择.venv目录
```

## 2. 修改.env.dev配置文件
- 根据实际情况修改一下配置信息
- **以下配置本机启动默认不用修改在服务器上启动时localhost需修改为实际IP地址**
    - **必须修改TAVILY_API_KEY** Tavily搜索配置
    - **必须修改MINIO_ACCESS_KEY** MinIO服务Key
    - **必须修改MINIO_SECRET_KEY** MinIO服务密钥
   
  
## 3. 初始化数据库
- 如果使用已安装的mysql,初始化数据时需修改源码initialize_mysql里面的连接信息
```bash
# Mac or Linux 用户执行
cd docker
./init_data.sh

# Windows 用户执行
cd common
python initialize_mysql.py
```

## 4. 前端依赖安装  
- 前端是基于开源项目[可参考chatgpt-vue3-light-mvp安装](https://github.com/pdsuwwz/chatgpt-vue3-light-mvp)二开
 
```bash
# 安装前端依赖&启动服务
cd web
   
#安装依赖
npm install -g pnpm

pnpm i
   
#启动服务
pnpm dev
```

## 5. 启动后端服务
```bash
#启动后端服务
python serv.py
```

## 6. langgraph(可选)
- langsmith studio 方式启动
```angular2html
 langgraph dev 
```

## 7. 访问服务
- 前端服务：http://localhost:2048


## 7. 构建镜像

- 执行构建命令：
```bash
# 构建前端镜像 
make web-build
  
# 构建后端镜像
make service-build
```