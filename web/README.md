# chatgpt-vue3-light-mvp

💭 一个可二次开发 Chat Bot 对话 Web 端原型模板, 基于 Vue3、Vite 5、TypeScript、Naive UI 、UnoCSS 等主流技术构建, 🧤简单集成大模型 API, 采用单轮 AI 问答对话模式, 每次提问独立响应, 无需上下文, 支持打字机效果流式输出, 集成 markdown-it, highlight.js 语法高亮预览, 💼 易于定制和快速搭建 Chat 类大语言模型产品


__[🌈 Live Demo 在线体验](https://pdsuwwz.github.io/chatgpt-vue3-light-mvp)__


## 🎉 特性

* 🛠️ **核心技术栈**：__Vite 5 + Vue 3 + TypeScript + ESLint (v9)__
* 🎨 **UI 框架**：__Naive UI 2.x__
* 🪄 **解放双手**：内置 **Unplugin Auto Import**，支持组件按需自动导入，提升开发效率
* 🌟 **图标支持**：内置 **UnoCSS + Iconify**，实现原子化样式内联和图标按需自动导入
* 💬 **AI 对话**：支持单轮对话，用户输入即得 AI 独立响应回复，无需上下文
* 📝 **Markdown 预览**：支持多种编程语言代码高亮，集成 `markdown-it` 和 `highlight.js`
* 🧪 **模拟开发模式**：提供本地模拟开发模式，无需真实 API 即可开始开发
* 🔑 **环境变量管理**：通过 `.env` 文件管理 API 密钥，支持不同大模型的配置
* 🌍 **大语言模型 API**：兼容 Spark 星火认知大模型、SiliconFlow、Ollama 等，允许自由扩展
* 🚀 **灵活扩展**：轻量级模块化 MVP 设计，纯前端开发，项目结构清晰，快速搭建 AI 对话原型


## 前置条件

* Vue 3.x
* Node >= 18.12.x
* Pnpm 9.x
* **VS Code 插件 `dbaeumer.vscode-eslint` >= v3.0.5 (pre-release)**

## 运行效果

![image](https://github.com/user-attachments/assets/6a066e66-1501-4816-880f-df37db658934)
![image](https://github.com/user-attachments/assets/71515bdc-1831-4421-b8df-127577d8a24d)


https://github.com/user-attachments/assets/42e8254f-d1bc-42e0-88dd-90279d040ae0


## 安装和运行

* 安装依赖

```bash
pnpm i
```

* 本地开发

```bash
pnpm dev
```

本地运行后，可以通过访问 `http://localhost:2048` 来查看应用。


## 🔑 配置 API 密钥

在运行项目之前，需要设置大语言模型的 API 密钥：

1. 执行以下命令，自动创建环境变量模板文件 `.env` 文件：
    ```sh
    cp .env.template .env
    ```

2. 编辑 `.env` 文件，填入你的 API 密钥

  ```sh
  VITE_SPARK_KEY=你的_星火_API_Key # 需要用冒号拼接key和secret，格式如 `key123456:secret123456`
  VITE_SILICONFLOW_KEY=你的_SiliconFlow_API_Key # 通常以 `sk-` 开头，如 `sk-xxxxxx`
  ```

## 🌍 模拟/真实 API 模式切换

本项目提供了一个模拟开发模式，用于在本地开发环境或 Github 等部署环境中模拟调用大模型相关策略，无需调用真实 API 接口。该模式在 [src/config/env.ts](src/config/env.ts) 文件中定义，由以下代码控制：

```ts
// src/config/env.ts

const isDev = import.meta.env.DEV

/**
 * TODO: 若是本地开发环境、Github 部署环境，则模拟大模型相关策略，不调接口
 */
export const isMockDevelopment = isDev
  || process.env.VITE_ROUTER_MODE === 'hash'

// 打开此行，则会调用真实的大模型接口（需提前配置好 Key）
// export const isMockDevelopment = false
```
### 默认配置

默认情况下，在开发环境或使用 `hash` 路由模式时, `isMockDevelopment` 会被设置为 `true`, 这意味着应用将使用模拟数据而不是真实的大模型 API 接口。

### 切换至真实 API

如果想在所有环境中使用真实的 API，你有两个选择：

1. **取消注释**：将最后一行的代码注释取消，设置 `isMockDevelopment = false`

2. **修改逻辑**：全局搜索 `isMockDevelopment`, 并修改相应的 if 判断逻辑，使其默认使用真实接口

**请注意，无论选择哪种方式，都需要确保项目已经正确配置了 `.env` API 密钥**

---

## 🦙 接入大语言模型 API

### 国内在线大模型

**1. Spark 星火认知大模型**：

- **注册**：访问 [星火大模型 API](https://xinghuo.xfyun.cn/sparkapi) 进行注册并登录
- **获取 API 密钥**：访问 [控制台](https://console.xfyun.cn/services/bm4) 获取 `APIKey` 和 `APISecret`

![image](https://github.com/user-attachments/assets/8761d59d-b4c3-41d5-9c58-14a5b0f4389c)

- **接口说明**：[spark HTTP 调用文档](https://www.xfyun.cn/doc/spark/HTTP%E8%B0%83%E7%94%A8%E6%96%87%E6%A1%A3.html#_1-%E6%8E%A5%E5%8F%A3%E8%AF%B4%E6%98%8E)

![image](https://github.com/user-attachments/assets/71334353-b6c1-4778-ae21-95e98860d2b1)

- **配置到本仓库**：将创建的 `APIKey` 和 `APISecret` 密钥用冒号 `:` 拼接填入到 `.env` 文件中的 `VITE_SPARK_KEY` 环境变量


**2. SiliconFlow 大模型**：
- **注册**：访问 [SiliconFlow 官网](https://siliconflow.cn/zh-cn/siliconcloud) 进行注册登录并创建 API 密钥
- **创建 API 密钥**：访问 [账户管理 - API 密钥](https://cloud.siliconflow.cn/account/ak) 创建新 API 密钥

![image](https://github.com/user-attachments/assets/31e1ef13-869a-4695-a7c0-054d2c3e877f)

- **接口说明**：[官方 Chat Completions 在线调试](https://docs.siliconflow.cn/reference/chat-completions-3)
- **配置到本仓库**：将创建的 API 密钥填入 `.env` 文件中的 `VITE_SILICONFLOW_KEY` 环境变量
- **SiliconFlow现已支持的大模型**：[模型列表](https://siliconflow.cn/zh-cn/models)

![image](https://github.com/user-attachments/assets/f320f495-cb17-48ff-99c4-aaedbf87fc84)


### 使用本地 Ollama 大模型

**Ollama3 大模型**：
- **安装**：Ollama3 不需要 API 密钥，只需要在本地安装并运行 Ollama 即可。请参考 Ollama 官方文档进行安装：[Ollama GitHub](https://github.com/ollama/ollama)
- **运行**：运行 Ollama3 服务，直接执行 `ollama run llama3`, 运行后确保其在 `http://localhost:11434` 运行

![image](https://github.com/user-attachments/assets/f3955060-a22d-4db8-b162-7393c10403f6)

- **查看运行状态**：执行 `ollama list`命令可查看当前正在运行的 Ollama 模型

![image](https://github.com/user-attachments/assets/8c6cf637-fd5b-45b5-93c2-f58927b7110c)

---

## 🔌 大模型响应处理

由于不同大模型的响应结果结构有所差异，本项目封装了一个 `model` 字段，用于控制响应结果的转换和字段提取。

### 🧠 已支持的模型

- **（默认类型）模拟数据模型**：`standard`
- **Spark 星火大模型**：`spark`
- **llama 3 大模型**：`ollama3`
- **SiliconFlow 硅基流动大模型**：`siliconflow`

### 🔬 主要实现

- **LLMTypes**: 定义了支持的大模型列表及其对应的 modelName，[详见代码](src/components/MarkdownPreview/transform/index.ts#L39)
- **TransformStreamModelTypes**: 基于 LLMTypes 推导出来的定义的模型名称类型，[详见代码](src/components/MarkdownPreview/transform/index.ts#L58)
- **transformStreamValue**: 包含了针对各种模型的响应结果转换函数，[详见代码](src/components/MarkdownPreview/transform/index.ts#L63)
- **MarkdownPreview 组件**: 接收 `model` props 属性，根据不同模型类型处理流式响应，[详见代码](src/components/MarkdownPreview/index.vue#L15)

### 📚 使用示例

在使用 [`MarkdownPreview`](src/components/MarkdownPreview/index.vue) 组件时，通过设置 `model` 属性来指定当前使用的大模型类型：

```html
<MarkdownPreview
  v-model:reader="outputTextReader"
  :model="defaultLLMTypeName"
  @failed="onFailedReader"
  @completed="onCompletedReader"
/>
```

其中 [`defaultLLMTypeName`](src/views/chat.vue#L15) 会根据映射自动选择对应的模型（也可具体指定一个模型）：

```ts
const defaultLLMTypeName: TransformStreamModelTypes = isMockDevelopment
  ? 'standard'
  : 'spark'
```

默认情况下，会处理 `spark` 模型，在模拟开发环境下，使用 standard 模型。具体的模型类型可以根据需求进行配置。

#### 💡 提示

> `defaultLLMTypeName` 会根据模型映射自动选择合适的模型，也可以手动指定模型
>
> 如果后端返回的是可直接渲染的纯字符串（而非 JSON），standard 模型将适用于所有这种情况

## 🌹 支持

如果你喜欢这个项目或发现有用，可以点右上角 [`Star`](https://github.com/pdsuwwz/chatgpt-vue3-light-mvp) 支持一下，你的支持是我们不断改进的动力，感谢！ ^_^


## 😎 Awesome

* [awesome-vite](https://github.com/pdsuwwz/awesome-vite)

## License

[MIT](./LICENSE) License | Copyright © 2020-PRESENT [Wisdom](https://github.com/pdsuwwz)
