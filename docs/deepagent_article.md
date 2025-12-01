# LangGraph + DeepAgents 多智能体协作实战：让 AI 像团队一样思考

## 前言

在大模型应用开发中，单一 Agent 往往难以应对复杂任务。就像一个人很难同时做好规划、执行和总结一样，我们需要让 AI 也学会"分工协作"。

本文将深入剖析基于 **LangGraph** 和 **DeepAgents** 框架的多智能体协作方案，通过实际代码展示如何构建一个由规划师、研究员、分析师组成的"AI 团队"，让它们像真实团队一样高效协作，完成深度研究任务。

## 一、为什么需要多智能体协作？

### 单一 Agent 的困境

传统的单一 Agent 架构存在以下问题：

| 问题类型 | 具体表现 | 影响 |
|---------|---------|------|
| **角色混乱** | 同时承担规划、执行、分析等多重职责 | 输出质量不稳定，逻辑混乱 |
| **上下文爆炸** | 所有信息堆积在一个对话中 | Token 消耗大，响应变慢 |
| **缺乏专业性** | 无法针对特定任务优化提示词 | 结果泛化，不够深入 |
| **难以调试** | 问题出现时无法定位具体环节 | 迭代优化困难 |

### 多智能体协作的优势

通过将复杂任务拆解给多个专业化的子智能体，可以实现：

- ✅ **职责清晰**：每个 Agent 专注于自己擅长的领域
- ✅ **流程可控**：明确的工作流程，便于监控和优化
- ✅ **质量提升**：专业化分工带来更高质量的输出
- ✅ **可扩展性**：轻松添加新的子智能体扩展能力

## 二、DeepAgents 架构设计

### 核心架构图

```
用户问题
   ↓
┌─────────────────────────────────────┐
│         DeepAgent 主控制器           │
│  (基于 LangGraph + InMemorySaver)   │
└─────────────────────────────────────┘
   ↓
┌──────────────────────────────────────────────────┐
│              三个子智能体协作流程                  │
├──────────────────────────────────────────────────┤
│  1. Planner (规划师)                              │
│     ├─ 分析问题                                   │
│     ├─ 识别关键要素                               │
│     └─ 制定研究计划                               │
│                                                   │
│  2. Researcher (研究员)                           │
│     ├─ 执行搜索任务                               │
│     ├─ 收集信息                                   │
│     └─ 验证数据可靠性                             │
│                                                   │
│  3. Analyst (分析师)                              │
│     ├─ 整合信息                                   │
│     ├─ 深度分析                                   │
│     └─ 生成结构化报告                             │
└──────────────────────────────────────────────────┘
   ↓
结构化 Markdown 报告
```

### 技术栈对比

| 技术组件 | 作用 | 为什么选它 |
|---------|------|-----------|
| **LangGraph** | 工作流编排 | 支持复杂的状态管理和条件分支，比 LangChain 更灵活 |
| **DeepAgents** | 多智能体框架 | 专为深度研究场景设计，内置子智能体协作机制 |
| **InMemorySaver** | 对话状态持久化 | 支持多轮对话记忆，用户体验更连贯 |
| **Tavily/Brave Search** | 网络搜索工具 | 提供实时、准确的网络信息检索能力 |

## 三、核心代码实现解析

### 3.1 DeepAgent 主类设计

```python
class DeepAgent:
    """
    基于DeepAgents的智能体，支持多轮对话记忆
    """
    
    def __init__(self):
        # 初始化LLM
        self.llm = get_llm()
        
        # 全局checkpointer用于持久化所有用户的对话状态
        self.checkpointer = InMemorySaver()
        
        # 存储运行中的任务
        self.running_tasks = {}
        
        # 配置参数
        self.RECURSION_LIMIT = int(os.getenv("RECURSION_LIMIT", 25))
        
        # 加载核心指令
        with open(os.path.join(current_dir, "instructions.md"), "r", encoding="utf-8") as f:
            self.CORE_INSTRUCTIONS = f.read()
        
        # 加载子智能体配置
        with open(os.path.join(current_dir, "subagents.json"), "r", encoding="utf-8") as f:
            self.subagents_config = json.load(f)
        
        # 提取三个子智能体的配置
        self.planner = self.subagents_config["planner"]
        self.researcher = self.subagents_config["researcher"]
        self.analyst = self.subagents_config["analyst"]
        
        # 定义智能体可以使用的工具
        self.tools = [search_web]
```

**设计亮点：**

1. **配置文件分离**：将系统提示词和子智能体配置分离到独立文件，便于调整和版本管理
2. **状态持久化**：使用 `InMemorySaver` 实现多轮对话记忆
3. **任务管理**：通过 `running_tasks` 字典管理并发任务，支持任务取消

### 3.2 子智能体配置详解

#### Planner（规划师）

```json
{
  "name": "planner",
  "description": "负责分析问题并制定研究策略",
  "system_prompt": "你是一个专业的研究规划师。你的任务是：
    1. 深入理解用户的问题和需求
    2. 识别问题的关键要素和研究方向
    3. 制定清晰、可执行的研究计划
    4. 确定需要收集哪些信息才能全面回答问题
    5. 考虑多个角度和维度，确保研究的全面性"
}
```

**职责定位：** 
- 🎯 问题拆解专家
- 🎯 研究路线设计师
- 🎯 不直接回答问题，而是为后续工作提供清晰指导

#### Researcher（研究员）

```json
{
  "name": "researcher",
  "description": "负责执行信息收集和数据获取",
  "system_prompt": "你是一个专业的研究员。你的任务是：
    1. 根据规划师的计划或用户的直接请求，使用可用工具收集信息
    2. 确保搜索关键词准确、相关，能够获取高质量的信息
    3. 如果初次搜索结果不够充分，从不同角度进行多次搜索
    4. 验证信息的可靠性和时效性
    5. 整理和归纳收集到的信息，便于后续分析"
}
```

**职责定位：**
- 🔍 信息收集执行者
- 🔍 多角度搜索策略实施者
- 🔍 数据质量把关者

#### Analyst（分析师）

```json
{
  "name": "analyst",
  "description": "负责分析信息并生成最终报告",
  "system_prompt": "你是一个专业的分析师和报告撰写专家。你的任务是：
    1. 仔细阅读和理解研究员收集的所有信息
    2. 对信息进行深度分析、对比、归纳和总结
    3. 识别关键发现、趋势和洞察
    4. 如果信息不足，明确指出需要补充哪些信息
    5. 撰写结构清晰、逻辑严密的中文研究报告"
}
```

**职责定位：**
- 📊 信息整合专家
- 📊 深度分析师
- 📊 报告撰写者

### 3.3 工作流程实现

```python
async def run_agent(
    self,
    query: str,
    response,
    session_id: Optional[str] = None,
    uuid_str: str = None,
    user_token=None,
    file_list: dict = None,
):
    """
    运行智能体，支持多轮对话记忆和实时思考过程输出
    """
    # 获取用户信息 标识对话状态
    user_dict = await decode_jwt_token(user_token)
    task_id = user_dict["id"]
    task_context = {"cancelled": False}
    self.running_tasks[task_id] = task_context
    
    try:
        # 使用用户会话ID作为thread_id，如果未提供则使用默认值
        thread_id = session_id if session_id else "default_thread"
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}
        
        # 创建深度智能体
        agent = create_deep_agent(
            tools=self.tools,
            system_prompt=self.CORE_INSTRUCTIONS,
            subagents=[self.planner, self.researcher, self.analyst],
            model=self.llm,
            backend=self.checkpointer,
        ).with_config({"recursion_limit": self.RECURSION_LIMIT})
        
        # 流式输出处理
        async for message_chunk, metadata in agent.astream(
            input={"messages": [HumanMessage(content=formatted_query)]},
            config=config,
            stream_mode="messages",
        ):
            # 检查是否已取消
            if self.running_tasks[task_id]["cancelled"]:
                await response.write(
                    self._create_response("\n> ⚠️ 任务已被用户取消", "info")
                )
                break
            
            # 获取当前节点信息
            node_name = metadata.get("langgraph_node", "unknown")
            
            # 工具调用输出
            if node_name == "tools":
                tool_name = message_chunk.name or "未知工具"
                
                if tool_name == "search_web":
                    search_content = message_chunk.content
                    content_json = json.loads(search_content)
                    think_html = f"""\n > ✅ 搜索{content_json["query"]}\n\n"""
                    await response.write(self._create_response(think_html, "info"))
                
                continue
            
            # 输出智能体的思考和回答内容
            if message_chunk.content:
                content = message_chunk.content
                await response.write(self._create_response(content))
```

**核心特性：**

| 特性 | 实现方式 | 价值 |
|------|---------|------|
| **多轮对话** | 通过 `thread_id` 区分不同会话 | 支持上下文连续对话 |
| **流式输出** | 使用 `astream` 实时返回结果 | 提升用户体验，减少等待感 |
| **任务取消** | 检查 `cancelled` 标志 | 用户可随时中断长时间任务 |
| **过程可视化** | 区分工具调用和内容输出 | 让用户看到 AI 的思考过程 |

### 3.4 搜索工具实现

```python
@tool
def search_web(query: str) -> str:
    """
    网络搜索工具
    
    参数:
        query: 搜索查询字符串
    
    返回:
        JSON 格式的搜索结果,包含查询内容和搜索结果列表
    """
    if not web_search:
        return json.dumps({"error": "搜索引擎未配置"}, ensure_ascii=False)
    
    try:
        # 执行搜索
        results = web_search(query)
        
        # 返回 JSON 格式的结果
        return json.dumps(
            {"query": query, "results": results}, 
            indent=2, 
            ensure_ascii=False
        )
    except Exception as e:
        logging.error(f"搜索失败: {str(e)}")
        return json.dumps({"error": f"搜索失败: {str(e)}"}, ensure_ascii=False)
```

**搜索引擎选择策略：**

```python
# 优先使用 Brave Search(如果配置了 API 密钥)
if BRAVE_SEARCH_API_KEY:
    brave_search = BraveSearch.from_api_key(
        api_key=BRAVE_SEARCH_API_KEY, 
        search_kwargs={"count": 5}
    )
    web_search = brave_search.run
    SEARCH_PROVIDER = "brave"

# 如果没有 Brave Search,则使用 Tavily Search
elif TAVILY_API_KEY:
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    web_search = tavily_client.search
    SEARCH_PROVIDER = "tavily"
```

**对比分析：**

| 搜索引擎 | 优势 | 适用场景 |
|---------|------|---------|
| **Brave Search** | 隐私保护好，无广告干扰 | 注重数据隐私的企业应用 |
| **Tavily** | 专为 AI 优化，结果结构化 | 需要高质量结构化数据的研究任务 |

## 四、实战应用场景

### 场景 1：深度行业研究

**用户输入：**
```
分析 2024 年 AI Agent 技术发展趋势，包括主要技术路线、代表性产品和未来方向
```

**执行流程：**

1. **Planner 规划阶段**
   ```
   研究计划：
   1. 搜索 "2024 AI Agent 技术趋势"
   2. 搜索 "多智能体协作框架对比"
   3. 搜索 "LangGraph AutoGen CrewAI 对比"
   4. 搜索 "AI Agent 商业应用案例"
   ```

2. **Researcher 执行阶段**
   ```
   ✅ 搜索 2024 AI Agent 技术趋势
   ✅ 搜索 多智能体协作框架对比
   ✅ 搜索 LangGraph AutoGen CrewAI 对比
   ✅ 搜索 AI Agent 商业应用案例
   ```

3. **Analyst 分析阶段**
   ```markdown
   # 2024 年 AI Agent 技术发展趋势分析报告
   
   ## 一、技术路线概览
   
   ### 主流框架对比
   
   | 框架 | 核心特点 | 适用场景 |
   |------|---------|---------|
   | LangGraph | 状态图编排，灵活度高 | 复杂工作流 |
   | AutoGen | 多智能体对话 | 协作任务 |
   | CrewAI | 角色化设计 | 团队模拟 |
   
   ## 二、代表性产品
   ...
   ```

### 场景 2：竞品分析

**用户输入：**
```
对比分析 ChatGPT、Claude 和 Gemini 在代码生成能力上的差异
```

**输出示例：**

```markdown
# ChatGPT vs Claude vs Gemini 代码生成能力对比报告

## 核心能力对比

| 维度 | ChatGPT | Claude | Gemini |
|------|---------|--------|--------|
| **代码理解** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **多语言支持** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **调试能力** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **上下文长度** | 128K | 200K | 2M |

## 详细分析
...
```

### 场景 3：技术选型建议

**用户输入：**
```
我要开发一个企业知识库问答系统，应该选择什么技术栈？
```

**工作流程：**

1. **Planner** 识别关键要素：企业级、知识库、问答系统
2. **Researcher** 搜索：RAG 技术方案、向量数据库对比、企业级部署方案
3. **Analyst** 输出：技术选型建议 + 架构设计 + 实施路线图

## 五、性能优化与最佳实践

### 5.1 提示词优化策略

| 优化点 | 优化前 | 优化后 | 效果提升 |
|-------|--------|--------|---------|
| **角色定位** | "你是一个助手" | "你是一个专业的研究规划师，擅长..." | 输出更专业 |
| **任务拆解** | "分析这个问题" | "1. 识别关键要素 2. 制定研究计划 3. ..." | 流程更清晰 |
| **输出格式** | "给我一个报告" | "使用 Markdown 格式，包含摘要、分析、结论..." | 结构更规范 |

### 5.2 并发控制

```python
# 任务管理机制
self.running_tasks = {}

# 启动任务
task_context = {"cancelled": False}
self.running_tasks[task_id] = task_context

# 检查取消
if self.running_tasks[task_id]["cancelled"]:
    await response.write(self._create_response("\n> ⚠️ 任务已被用户取消"))
    break

# 清理任务
if task_id in self.running_tasks:
    del self.running_tasks[task_id]
```

**优势：**
- ✅ 支持多用户并发
- ✅ 任务可随时取消
- ✅ 自动资源清理

### 5.3 成本控制

| 策略 | 实现方式 | 节省比例 |
|------|---------|---------|
| **缓存搜索结果** | 相同查询 24 小时内复用 | ~30% |
| **递归限制** | `RECURSION_LIMIT=25` | 防止无限循环 |
| **流式输出** | 使用 `astream` 而非 `ainvoke` | 提升体验，无额外成本 |
| **模型选择** | Planner 用小模型，Analyst 用大模型 | ~40% |

### 5.4 错误处理

```python
try:
    # 主流程
    async for message_chunk, metadata in agent.astream(...):
        ...
        
except asyncio.CancelledError:
    await response.write(self._create_response("\n> ⚠️ 任务已被取消"))
    
except Exception as e:
    logger.error(f"Agent运行异常: {e}")
    traceback.print_exception(e)
    error_msg = f"❌ **错误**: 智能体运行异常\n\n```\n{str(e)}\n```\n"
    await response.write(self._create_response(error_msg, "error"))
    
finally:
    # 清理任务记录
    if task_id in self.running_tasks:
        del self.running_tasks[task_id]
```

## 六、与其他方案对比

### 主流多智能体框架对比

| 框架 | 编排方式 | 学习曲线 | 灵活性 | 适用场景 |
|------|---------|---------|--------|---------|
| **LangGraph + DeepAgents** | 状态图 | 中等 | ⭐⭐⭐⭐⭐ | 复杂研究任务 |
| **AutoGen** | 对话驱动 | 较低 | ⭐⭐⭐ | 多轮对话协作 |
| **CrewAI** | 角色编排 | 较低 | ⭐⭐⭐⭐ | 团队模拟场景 |
| **MetaGPT** | 软件工程 | 较高 | ⭐⭐⭐⭐ | 代码生成项目 |
| **Dify Workflow** | 可视化拖拽 | 最低 | ⭐⭐ | 快速原型验证 |

### 为什么选择 LangGraph + DeepAgents？

**核心优势：**

1. **状态管理强大**
   - LangGraph 的状态图机制支持复杂的条件分支
   - 可以精确控制每个节点的执行逻辑

2. **专为研究场景优化**
   - DeepAgents 内置了 Planner-Researcher-Analyst 三角色模型
   - 提示词模板针对深度研究任务优化

3. **可观测性好**
   - 通过 `metadata` 可以追踪每个节点的执行
   - 便于调试和性能分析

4. **生产级特性**
   - 支持持久化（可替换为 PostgreSQL/Redis）
   - 支持流式输出
   - 支持任务取消

## 七、部署与集成

### 7.1 环境配置

```bash
# 安装依赖
pip install langgraph langchain-core deepagents tavily-python langchain-community

# 配置环境变量
export TAVILY_API_KEY="your_tavily_key"
export BRAVE_SEARCH_API_KEY="your_brave_key"  # 可选
export RECURSION_LIMIT=25
```

### 7.2 与 Sanic Web 集成

```python
from sanic import Sanic, response
from agent.deepagent.deep_research_agent import DeepAgent

app = Sanic("DeepAgentApp")
deep_agent = DeepAgent()

@app.route("/api/research", methods=["POST"])
async def research_endpoint(request):
    query = request.json.get("query")
    session_id = request.json.get("session_id")
    
    async def streaming_response(response):
        await deep_agent.run_agent(
            query=query,
            response=response,
            session_id=session_id,
            user_token=request.token
        )
    
    return response.stream(streaming_response, content_type="text/event-stream")
```

### 7.3 Docker 部署

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "serv.py"]
```

## 八、未来展望

### 即将支持的功能

| 功能 | 预期价值 | 实现难度 |
|------|---------|---------|
| **多模态支持** | 支持图片、PDF 等文件分析 | ⭐⭐⭐ |
| **知识库集成** | 结合企业私有知识库 | ⭐⭐ |
| **自定义子智能体** | 用户可配置专属 Agent | ⭐⭐⭐⭐ |
| **协作可视化** | 实时展示智能体协作过程 | ⭐⭐⭐ |
| **成本分析** | 详细的 Token 消耗统计 | ⭐⭐ |

### 技术演进方向

1. **更智能的规划**
   - 引入强化学习优化规划策略
   - 根据历史任务自动调整研究路线

2. **更高效的协作**
   - 支持子智能体并行执行
   - 动态调整智能体数量

3. **更好的可观测性**
   - 集成 LangSmith 进行全链路追踪
   - 提供详细的性能分析报告

## 九、总结

通过 **LangGraph + DeepAgents** 构建的多智能体协作系统，我们实现了：

✅ **专业化分工**：规划师、研究员、分析师各司其职  
✅ **流程化协作**：清晰的工作流程，可控可追溯  
✅ **高质量输出**：结构化的 Markdown 报告，专业且易读  
✅ **生产级特性**：多轮对话、流式输出、任务管理一应俱全  

这套方案已在实际项目中验证，能够稳定处理各类深度研究任务。如果你也在构建复杂的 AI 应用，不妨试试这种"让 AI 像团队一样思考"的方式。

---

## 附录：完整配置文件

### instructions.md（核心指令）

```markdown
你是一个深度研究智能体，由多个子智能体（规划师、研究员、分析师）协作完成任务。

# 核心能力
- **信息收集**: 通过网络搜索获取最新、最准确的信息
- **深度分析**: 对收集到的信息进行多维度分析和综合
- **报告生成**: 根据用户需求生成结构化、专业的研究报告

# 标准工作流程
1. **理解需求**: 规划师深入分析用户问题
2. **制定计划**: 规划师提出具体的研究策略
3. **信息收集**: 研究员根据计划执行搜索
4. **质量检查**: 分析师评估收集到的信息
5. **深度分析**: 分析师对信息进行综合分析
6. **报告输出**: 分析师撰写结构化的最终报告

## 输出要求
- 必须使用 Markdown 格式
- 使用清晰的标题层级
- 适当使用列表、表格、引用等格式
- 必须使用中文
```

### subagents.json（子智能体配置）

```json
{
  "planner": {
    "name": "planner",
    "description": "负责分析问题并制定研究策略",
    "system_prompt": "你是一个专业的研究规划师..."
  },
  "researcher": {
    "name": "researcher",
    "description": "负责执行信息收集和数据获取",
    "system_prompt": "你是一个专业的研究员..."
  },
  "analyst": {
    "name": "analyst",
    "description": "负责分析信息并生成最终报告",
    "system_prompt": "你是一个专业的分析师..."
  }
}
```

---

**关于作者**

专注于大模型应用开发，擅长 LangChain/LangGraph、RAG、Text2SQL 等技术栈。本文基于实际项目经验总结，代码已在生产环境验证。

**项目地址**: [sanic-web](https://github.com/apconw/sanic-web)

**技术交流**: 欢迎关注公众号，获取更多大模型应用开发实战内容。

---

*本文所有代码示例均来自开源项目 sanic-web，采用 MIT 协议开源。*
