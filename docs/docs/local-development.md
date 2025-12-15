
确保上一步[环境配置](environment.md)已配置好
> 本地开发依赖**环境配置**中的间件服务、所以先使用docker启动中间件服务

# 项目骨架
```angular2html
sanic-web/
├── agent/              # 智能体模块
│   ├── context/       # 上下文管理
│   ├── deepagent/     # DeepAgent 智能体
│   ├── excel/         # Excel 处理
│   ├── mcp/           # MCP 协议适配
│   ├── middleware/    # 中间件
│   ├── text2sql/      # Text2SQL 智能体
│   └── common_react_agent.py
├── controllers/        # 控制器层 (API 接口)
│   ├── common_chat_api.py
│   ├── db_chat_api.py
│   ├── dify_chat_api.py
│   ├── file_chat_api.py
│   ├── ta_assistant_api.py
│   └── user_service_api.py
├── services/          # 服务层 (业务逻辑)
│   ├── db_qadata_process.py
│   ├── dify_service.py
│   ├── file_chat_service.py
│   ├── search_service.py
│   ├── ta_assistant_service.py
│   ├── text2_sql_service.py
│   └── user_service.py
├── model/             # 数据模型层
│   ├── db_connection_pool.py
│   ├── db_models.py
│   └── serializers.py
├── common/            # 通用工具类
│   ├── neo4j/        # Neo4j 工具
│   ├── date_util.py
│   ├── duckdb_util.py
│   ├── llm_util.py
│   ├── mcp_client.py
│   ├── redis_tool.py
│   └── ...
├── config/            # 配置管理
├── constants/         # 常量定义
└── serv.py           # 应用入口
```

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
- **1.以下配置本机启动默认不用修改在服务器上启动时localhost需修改为实际IP地址**
    - **必须修改TAVILY_API_KEY** Tavily搜索配置
    - **必须修改MINIO_ACCESS_KEY** MinIO服务Key
    - **必须修改MINIO_SECRET_KEY** MinIO服务密钥
- **2.修改大模型/嵌入模型/重排模型密钥**
   
  
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