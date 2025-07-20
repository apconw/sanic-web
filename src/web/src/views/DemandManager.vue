<script setup>
import { marked } from 'marked' // 引入 marked 库
import { NLayout, NLayoutContent, NLayoutHeader } from 'naive-ui'
import * as GlobalAPI from '@/api'

const router = useRouter()

// 文件上传
const uploadDocRef = ref()
const finish_upload = (res) => {
  if (res.event.target.responseText) {
    const json_data = JSON.parse(res.event.target.responseText)
    const file_key = json_data.data.object_key
    if (json_data.code === 200) {
      window.$ModalMessage.success(`文件上传成功`)
      projectForm.value.file_key = file_key
      projectForm.value.doc_name = file_key.split('.')[0]
      projectForm.value.doc_desc = file_key.split('.')[0]
    } else {
      window.$ModalMessage.error(`文件上传失败`)
    }
  }
}

// 抽取信息
const showAbModal = ref(false)
const progress = ref(null)
const messages = ref([])
const realTimeContent = ref(null) // 引用容器元素

function startExtraction(itemId) {
  showAbModal.value = true
  progress.value = 0 // 初始化进度为0
  const eventSource = new EventSource(
    `${location.origin}/sanic/ta/abstract_doc_func/${itemId}`,
  )

  eventSource.onmessage = function (event) {
    const data = JSON.parse(event.data)
    if (data.type === 'progress') {
      // 更新进度条
      progress.value = data.progress
      // messages.value.push(`进度: ${data.progress}%`)
    } else if (data.type === 'log') {
      // 显示日志信息
      messages.value.push(data.message)
    } else if (data.type === 'complete') {
      // 关闭模态框
      messages.value.push('任务完成')
      eventSource.close()
      setTimeout(() => {
        showAbModal.value = false
        messages.value = []
        query_demand_records()
      }, 1000)
    }
    scrollToBottom() // 每次收到消息后滚动到底部
  }

  eventSource.onerror = function (error) {
    console.error('EventSource failed:', error)
    messages.value = []
    // messages.value.push('发生错误，请稍后再试')
    eventSource.close()
    showAbModal.value = false
  }
}

// 滚动到底部的函数
function scrollToBottom() {
  if (realTimeContent.value) {
    realTimeContent.value.scrollTop
            = realTimeContent.value.scrollHeight + 20
  }
}

onMounted(() => {
  // 页面加载时也可以调用一次滚动到底部
  scrollToBottom()
})

// Form表单
const showModal = ref(false)
const items = ref([])

const projectForm = ref({
  doc_name: '',
  doc_desc: '',
  file_key: '',
})
const submitProject = async () => {
  const res = await GlobalAPI.insert_demand_manager(projectForm.value)
  const json = await res.json()
  if (json?.data !== undefined && json?.data) {
    window.$ModalMessage.success(`项目创建成功`)
    closeModal()
  }

  query_demand_records()
}

const closeModal = () => {
  showModal.value = false
  // 清空表单
  projectForm.value = {
    doc_name: '',
    doc_desc: '',
    file_key: '',
  }
}

const dropdownOptions = [
  {
    label: '抽取',
    key: 'abstract',
  },
  {
    label: '编辑',
    key: 'edit',
  },
  {
    label: '删除',
    key: 'delete',
  },
]

const handleSelect = async (key, index) => {
  switch (key) {
    // 抽取功能
    case 'abstract':
      console.log(`Editing item at index ${index}`)
      startExtraction(index)
      break
    case 'edit':
      console.log(`Editing item at index ${index}`)
      // 编辑项目的逻辑
      break
    case 'delete':
      GlobalAPI.delete_demand_records(`${index}`)
      await query_demand_records()
      break
    default:
      console.log(`Selected option not handled: ${key}`)
  }
}

const query_demand_records = async () => {
  const res = await GlobalAPI.query_demand_records(1, 999999)
  const json = await res.json()
  if (json?.data !== undefined) {
    items.value = json.data.records
  } else {
    items.value = []
  }
}

onMounted(() => {
  query_demand_records()
})

function navigateToDetail(id) {
  router.push({ name: 'UaDetail', params: { id } })
}
</script>

<template>
  <n-layout
    class="h-full"
  >
    <n-layout-header class="header">
      <div class="header-content">
        <!-- 这里可以放置一些顶部的内容或导航 -->
      </div>
      <button class="create-project-btn" @click="showModal = true">
        + 创建项目
      </button>
    </n-layout-header>
    <n-layout-content>
      <div class="container">
        <div
          v-for="(item, index) in items"
          :key="index"
          class="card"
          @click="navigateToDetail(item.id)"
        >
          <div class="card-header">
            <n-icon style="margin-right: 5px" size="18">
              <div class="i-formkit:filedoc"></div>
            </n-icon>
            <span class="card-title">需求</span>
          </div>
          <div class="card-body">
            <p>{{ item.doc_desc }}</p>
          </div>
          <div class="card-footer">
            <span class="card-info">功能点: {{ item.fun_num }}</span>
            <span class="card-date">{{
              item.update_time
            }}</span>
            <!-- 使用 n-dropdown 组件替换原有的按钮 -->
            <n-dropdown
              trigger="click"
              :options="dropdownOptions"
              @select="(key) => handleSelect(key, item.id)"
            >
              <button class="card-button" @click.stop>
                ...
              </button>
            </n-dropdown>
          </div>
        </div>
      </div>
    </n-layout-content>
  </n-layout>

  <!-- 模态框 -->
  <n-modal
    v-model:show="showModal"
    preset="dialog"
    title="创建新项目"
    style="width: 600px"
    @close="closeModal"
  >
    <n-form :model="projectForm">
      <n-form-item label="项目名称" required>
        <n-input
          v-model:value="projectForm.doc_name"
          placeholder="请输入项目名称"
        />
      </n-form-item>
      <n-form-item label="项目描述" required>
        <n-input
          v-model:value="projectForm.doc_desc"
          type="textarea"
          placeholder="请输入项目描述"
        />
      </n-form-item>
      <n-form-item label="项目附件" hidden>
        <n-input v-model:value="projectForm.file_key" />
      </n-form-item>
      <n-upload
        ref="uploadDocRef"
        multiple
        :show-file-list="true"
        action="sanic/file/upload_file"
        accept=".doc, .docx"
        @finish="finish_upload"
      >
        <n-button>上传附件</n-button>
      </n-upload>
    </n-form>
    <template #action>
      <n-button @click="submitProject">提交</n-button>
      <n-button @click="closeModal">取消</n-button>
    </template>
  </n-modal>

  <n-modal
    v-model:show="showAbModal"
    :closable="false"
    preset="dialog"
    title="抽取功能"
    :mask-closable="false"
    style="width: 800px"
  >
    <div v-if="progress !== null">
      <n-progress type="line" :percentage="progress" />
    </div>
    <div v-else>正在准备...</div>

    <!-- 实时显示推送的内容 -->
    <div ref="realTimeContent" class="real-time-content">
      <p
        v-for="(message, index) in messages"
        :key="index"
        v-html="marked(message)"
      ></p>
    </div>
    <div
      class="i-svg-spinners:pulse-2 c-#26244c"
      style="width: 30px; height: 30px; margin-left: -8px"
    ></div>
  </n-modal>
</template>

<style scoped>
.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 20px; /* 调整padding以适应设计 */
  background-color: #f6f7fb; /* 根据需要调整背景颜色 */
}

.header-content {
  /* 这里可以添加任何必要的样式，比如logo或导航链接 */
}

.create-project-btn {
  background-color: #2c7be5;
  color: #fff;
  border: none;
  padding: 10px 20px;
  border-radius: 20px;
  cursor: pointer;
  font-size: 14px;
}

.container {
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
  padding: 20px;
}

.card {
  width: 250px;
  margin-top: 10px;
  border-radius: 8px;
  box-shadow: 0 4px 6px rgb(0 0 0 / 10%);
  background-color: #fff;
  overflow: hidden;
}

.card-header {
  display: flex;
  align-items: center;
  padding: 10px;
  background-color: #f9f9f9;
}

.card-icon {
  width: 20px;
  height: 20px;
  margin-right: 10px;
}

.card-title {
  font-weight: bold;
}

.card-body {
  padding: 10px;
}

.card-footer {
  display: flex;
  justify-content: space-between;
  padding: 10px;
  background-color: #fff;
}

.card-info,
.card-date {
  font-size: 12px;
  color: #666;
}

.card-button {
  background-color: #e0e0e0;
  border: none;
  padding: 5px 10px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
}

form-item-inline {
  display: flex;
  align-items: center;
}

.form-item-inline .n-form-item__label {
  width: 120px; /* 设置标签宽度 */
  margin-right: 15px; /* 设置标签与输入框之间的间距 */
}

/* 滚动条整体部分 */

::-webkit-scrollbar {
  width: 8px; /* 竖向滚动条宽度 */
  height: 8px; /* 横向滚动条高度 */
}

/* 滚动条的轨道 */

::-webkit-scrollbar-track {
  background: #fff; /* 轨道背景色 */
}

/* 滚动条的滑块 */

::-webkit-scrollbar-thumb {
  background: #cac9f9; /* 滑块颜色 */
  border-radius: 10px; /* 滑块圆角 */
}

/* 滚动条的滑块在悬停状态下的样式 */

::-webkit-scrollbar-thumb:hover {
  background: #cac9f9; /* 悬停时滑块颜色 */
}

.real-time-content {
  margin-top: 10px;
  max-height: 300px;
  overflow-y: hidden;
  border: 0 solid #ccc;
  padding-top: 10px;
  background-color: #fff; /* 黑色背景 */
  color: #26244c;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
    'Helvetica Neue', Arial, sans-serif;
}

/* 当鼠标悬停时改变overflow-y属性 */

.real-time-content:hover {
  overflow-y: auto;
}
</style>
