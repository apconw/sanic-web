<script lang="tsx" setup>
import { Client } from '@modelcontextprotocol/sdk/client/index.js'
import { SSEClientTransport } from '@modelcontextprotocol/sdk/client/sse.js' // 假设存在 SseClientTransport

const message = ref('')
const messages = ref('')
const connectionStatus = ref('Disconnected')
// https://github.com/modelcontextprotocol/typescript-sdk
// StdioClientTransport
// 如果是本地开发，需要使用 localhost 来访问宿主机的mcp服务
const transport = new SSEClientTransport(
  new URL('http://host.docker.internal:8000/sse'),
)

const client = new Client(
  {
    name: 'mcp-client',
    version: '1.0.0',
  },
  {
    capabilities: {
      prompts: {},
      resources: {},
      tools: {},
    },
  },
)

onMounted(async () => {
  await client.connect(transport)
  connectionStatus.value = 'Connected'
  const tools = await client.listTools()
  console.log(tools)

  // const prompts = await client.listPrompts()
  // console.log(prompts)
})

const sendMessage = async () => {
  const result = await client.callTool({
    name: 'maps_weather',
    arguments: {
      city: message.value,
    },
  })
  messages.value = JSON.stringify(result.content)
  console.log(result.content)
}

// List prompts
// const prompts = await client.listPrompts()

// // Get a prompt
// const prompt = await client.getPrompt('example-prompt', {
//     arg1: 'value'
// })

// List resources
// const resources = await client.listResources()

// // Read a resource
// const resource = await client.readResource('file:///example.txt')
</script>

<template>
  <LayoutCenterPanel>
    <n-layout class="main-layout" style="height: 100vh">
      <!-- 移动说明区块到 n-layout 外部 -->
      <div class="note-box">
        <p>MCP功能说明：</p>
        <ul>
          <li>
            目前该功能只做了简单的客户端对接示例;
            下一版本对接qwen3mcp能力实现多工具协调调用
          </li>
          <li>
            第一步 申请高德API-KEY 并启动高德 MCP Server，命令如下：
          </li>
          <li>
            npx -y supergateway --stdio
            "AMAP_MAPS_API_KEY=高德API-KEY npx -y
            @amap/amap-maps-mcp-server" --cors
          </li>
          <li>
            通过 maps_weather 工具查询实时天气,输出城市名称如:杭州
          </li>
        </ul>
      </div>

      <div class="mcp-client">
        <h1>MCP Client</h1>
        <p>Connection Status: {{ connectionStatus }}</p>

        <!-- 输入框和按钮 -->
        <div>
          <input
            v-model="message"
            style="width: 280px; height: 25px"
            placeholder="Enter message to send"
          >
          <button
            style="height: 25px; margin-left: 2px; width: 80px"
            @click="sendMessage"
          >
            Send
          </button>
        </div>

        <!-- 消息日志 -->
        <div class="message-log">
          <h3>Message Log:</h3>
          <ul>
            {{
              messages
            }}
          </ul>
        </div>
      </div>
    </n-layout>
  </LayoutCenterPanel>
</template>

<style scoped>
/* 全局样式 */

.note-box {
  max-width: 600px;
  margin: 20px auto 0;
  background: #f8f9fa;
  border-left: 4px solid #3b82f6;
  border-radius: 4px;
  padding: 15px;
}

.note-box ul {
  margin: 10px 0 0 20px;
}

.note-box li {
  list-style-type: disc;
  margin: 5px 0;
}

/* 保持原有 scoped 样式 */

.mcp-client {
  font-family: Arial, sans-serif;
  max-width: 600px;
  margin: 0 auto;
  padding: 20px;
  border: 1px solid #ccc;
  border-radius: 8px;
  margin-top: 20px;
}

.message-log {
  margin-top: 20px;
}

ul {
  list-style-type: none;
  padding: 0;
}

li {
  background: #f9f9f9;
  margin: 5px 0;
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 4px;
}

.main-layout,
.sub-layout {
  height: 100%;
  border-radius: 10px;
  margin-bottom: 10px;
  margin-right: 2px;
  background-color: #fff;
}
</style>
