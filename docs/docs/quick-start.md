确保上一步[环境配置](environment.md)已配置好

## 1. **环境配置**
> 获取上一步骤MINIO密钥和MCP-HUB工具集配置ID编辑.env文件

### 1.1. **服务IP修改**
- 本地部署不要修改、服务器部署时配置成实际IP地址
```angular2html
# 中间件容器内部访问地址 - 服务器部署时配置成实际IP地址
DOCKER_HOST_INTERNAL=host.docker.internal

# 中间件本地访问地址 - 服务器部署时配置成实际IP地址
PUBLIC_DOMAIN_HOST=localhost
```
### 1.2. **MINIO密钥配置**
- 从上一步MINIO环境配置中获取
```angular2html
MINIO_ACCESS_KEY=QfIx8FgdpgKtmbFMbKVb
MiNIO_SECRET_KEY=DsitWZJT3pecrg020Y2NKCETVpsIc3h2PrKTqONA
```

### 1.3. **大模型密钥配置**
- 大模型密钥配置
  - **本地ollama模式 MODEL_TYPE需配置成ollama**
```angular2html
MODEL_TYPE="openai"
MODEL_NAME="qwen3-max"
MODEL_API_KEY="sk-xxx"
MODEL_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_TEMPERATURE=0.75
```
- 小模型配置
```angular2html
SMALL_MODEL_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
SMALL_MODEL_API_KEY="sk-xxx"
RERANK_MODEL_NAME="gte-rerank-v2"
EMBEDDING_MODEL_NAME="text-embedding-v4"
```

### 1.4. **MCP-HUB工具集配置**
- 从上步MCP-HUB工具集配置中获取**只修改ID值**
- **http://${DOCKER_HOST_INTERNAL}:3300/mcp/** 前段部分保持不变
```angular2html
# MCP-HUB工具集配置
MCP_HUB_COMMON_QA_GROUP_URL="http://${DOCKER_HOST_INTERNAL}:3300/mcp/d7af20c7-1b08-4963-82b6-41affc54a20d"
MCP_HUB_DATABASE_QA_GROUP_URL="http://${DOCKER_HOST_INTERNAL}:3300/mcp/71a21b11-d684-462d-9005-79bc62934d88"
```

### 1.5. **Tavily 配置**
- 用于深度搜索智能体
- 获取Tavily密钥配置(https://www.tavily.com/)
```angular2html
TAVILY_API_KEY="sk-xxx"
```


## 2. **重起服务**
```angular2html
docker compose down && docker compose up -d
```
   
## 3. **访问服务**
- 前端服务：http://localhost:8081