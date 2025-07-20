# **💡说明**

> - 请注意，本次展示回复内容为**模拟 Event Stream 静态输出**，非真实接口调用。
>
>- 如果需要调用真实的 GPT 接口，请拉取本仓库，在本地开发环境中配置相应的 `API` 密钥和`接口地址`。

## <a target="_blank" href="https://github.com/pdsuwwz/chatgpt-vue3-light-mvp">关于本仓库</a>

![GitHub Pages](https://img.shields.io/badge/gh--pages-passing-brightgreen)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Thanks](https://img.shields.io/badge/thanks-%E2%9D%A4-pink)
![License](https://img.shields.io/badge/license-MIT-blue)


<a target="_blank" href="https://github.com/pdsuwwz/chatgpt-vue3-light-mvp">本仓库</a>是一个可二次开发的 ChatBot 对话 Web 端 MVP 原型模板，基于 `Vue3`、`TypeScript`、`Naive UI`、`UnoCSS` 等主流技术构建。项目特点如下：

- **大语言模型 API 集成**: 简单集成大语言模型 API，支持单轮 AI 对话模式。每次提问都会独立响应，无需上下文，提供简洁高效的对话体验。
- **Markdown 预览**: 集成 Markdown 预览功能，使用户可以发送和接收格式化的 Markdown 内容，增强用户交流的灵活性和展示效果。
- **易于定制**: 项目结构清晰，代码简洁，易于根据实际需求进行定制和扩展，快速搭建符合特定需求的 Chat 产品。

**主要特性**:
- 使用 Vue 3 的 `setup` 语法和 TypeScript 构建，提供现代化的开发体验。
- 集成 Naive UI 和 UnoCSS，提供优雅的用户界面和简洁的样式管理。
- 支持 markdown-it 预览，用户可以发送和接收格式化内容，提升交互效果。

**如何使用**:
1. **克隆仓库**: 将本仓库克隆到本地。
    ```bash
    git clone https://github.com/pdsuwwz/chatgpt-vue3-light-mvp.git
    ```
2. **安装依赖**: 进入项目目录并安装依赖。
    ```bash
    cd chatgpt-vue3-light-mvp
    pnpm i
    ```
3. **本地运行**: 启动本地开发服务器。
    ```bash
    pnpm dev
    ```
---


以下是一个使用 Vue 3 `setup` 语法的示例组件。


```vue
<script setup lang="ts">
// 创建一个响应式变量
const message = ref('欢迎使用 Vue 3 组件示例！')

// 定义一个方法来更新 message 的值
const updateMessage = () => {
  message.value = '你点击了按钮！'
}
</script>

<template>
  <div>
    <h1>{{ message }}</h1>
    <button @click="updateMessage">点击我</button>
  </div>
</template>

<style lang="scss" scoped>
/* 添加一些样式以美化组件 */
h1 {
  color: #42b983;
}
button {
  padding: 10px 20px;
  font-size: 16px;
  background-color: #42b983;
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
}
button:hover {
  background-color: #35495e;
}
</style>
```

**模拟接口说明**:
- **接口请求**: 当用户发送请求“请展示一个 Vue 3 的示例组件”时，本模拟系统返回上述代码和说明。
- **接口响应**: 响应中包括 Vue 3 示例组件的代码和详细说明，用于演示如何使用 `setup` 语法构建组件。

**注意**: 本模拟回复仅为演示用途，真实的接口调用及其效果需在本地运行相应代码来体验。


**组件细节描述**:
- **功能描述**: 该组件包含一个标题和一个按钮，点击按钮会触发更新消息的功能。
- **技术细节**: 使用 Vue 3 的 `setup` 语法来定义响应式数据和方法。`message` 是一个使用 `ref` 创建的响应式变量，`updateMessage` 方法用于更新该变量的值。
- **样式说明**: 组件的样式使用了 `scss` 语法，并且 `scoped` 确保样式只应用于该组件。标题的颜色设置为绿色，按钮在悬停时会变为深色。

**适用场景**:
- **学习和演示**: 该组件是一个简单的示例，非常适合用于学习和演示 Vue 3 的基本用法和 `setup` 语法。
- **实际应用**: 可以根据项目需求，对该组件进行扩展和修改，应用到实际项目中。


**如果您有其他问题或需要进一步的帮助，请随时告知！**

## 🌹 说明

> * 如果此开源对您有帮助，您可以点<a target="_blank" href="https://github.com/pdsuwwz/chatgpt-vue3-light-mvp">开源仓库</a>右上角 \`Star\` 支持一下 谢谢！ ^_^ ⭐️
>
> * 或者您可以 <a target="_blank" href="https://github.com/pdsuwwz">Follow</a> 一下, 我会不断开源更多有趣和实用的项目
>
> * 开发环境 MacOS Ventura, VSCode, Chrome
>
> * 推荐一个 `Element Plus` + `Vite5` + `Vue3` + `TS` + `UnoCSS` 开源入门模板项目, 对 Element Plus UI 库感兴趣的朋友可以去看看。<a target="_blank" href="https://github.com/pdsuwwz/vite-ts-starter">地址在这里</a>
>
> * 另外一个 `Naive UI` 开源版本的，基于 `Vite5` + `Vue3` + `TS` + `UnoCSS` 的入门模板项目, 非常适合入门练习和二次开发。<a target="_blank" href="https://github.com/pdsuwwz/vue3-tab-demo">地址在这里</a>
