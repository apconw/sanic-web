<script lang="tsx" setup>
import type { InputInst } from 'naive-ui'
import { init } from 'echarts'
import { UAParser } from 'ua-parser-js'
import * as GlobalAPI from '@/api'
import { isMockDevelopment } from '@/config'
import DefaultPage from './DefaultPage.vue'
import SuggestedView from './SuggestedPage.vue'
import TableModal from './TableModal.vue'

const route = useRoute()
const router = useRouter()
const message = useMessage()

// 显示默认页面
const showDefaultPage = ref(true)

// 全局存储
const businessStore = useBusinessStore()

// 是否是刚登录到系统 批量渲染对话记录
const isInit = ref(false)

// 是否查看历史消息标识
const isView = ref(false)

// 使用 onMounted 生命周期钩子加载历史对话
// 新增：加载历史对话的状态
const isLoadingHistory = ref(false)

// 使用 onMounted 生命周期钩子加载历史对话
onBeforeMount(() => {
  try {
    // 开始加载历史对话
    isLoadingHistory.value = true
    isInit.value = true
    fetchConversationHistory(isInit, conversationItems, tableData, currentRenderIndex, null, '')
  } catch (error) {
    console.error('加载历史对话失败:', error)
    window.$ModalMessage.error('加载历史对话失败，请重试')
  } finally {
    // 加载完成
    isLoadingHistory.value = false
  }
})

// 管理对话
const isModalOpen = ref(false)
function openModal() {
  isModalOpen.value = true
}
// 模态框关闭
function handleModalClose(value) {
  isModalOpen.value = value
  isInit.value = true
  // 重新加载对话记录
  fetchConversationHistory(
    isInit,
    conversationItems,
    tableData,
    currentRenderIndex,
    null,
    '',
  )
}

// 新建对话
function newChat() {
  if (showDefaultPage.value) {
    window.$ModalMessage.success(`已经是最新对话`)
    return
  }
  showDefaultPage.value = true
  isInit.value = false
  conversationItems.value = []
  stylizingLoading.value = false
  suggested_array.value = []

  // 新增：生成当前问答类型的新uuid
  uuids.value[qa_type.value] = uuidv4()
}

/**
 * 默认大模型
 */
const defaultLLMTypeName = 'qwen2'
const currentChatId = computed(() => {
  return route.params.chatId
})


// 对话等待提示词图标
const stylizingLoading = ref(false)

// 输入字符串
const inputTextString = ref('')
const refInputTextString = ref<InputInst | null>()

// 输出字符串 Reader 流（风格化的）
const outputTextReader = ref<ReadableStreamDefaultReader | null>()

// markdown对象
const refReaderMarkdownPreview = ref<any>()

// 主内容区域
const messagesContainer = ref<HTMLElement | null>(null)

// 读取失败
const onFailedReader = (index: number) => {
  if (conversationItems.value[index]) {
    conversationItems.value[index].reader = null
    stylizingLoading.value = false
    if (refReaderMarkdownPreview.value) {
      refReaderMarkdownPreview.value.initializeEnd()
    }
    window.$ModalMessage.error('请求失败，请重试')
    setTimeout(() => {
      if (refInputTextString.value) {
        refInputTextString.value.focus()
      }
    })
  }
}

// 读取完成
const onCompletedReader = (index: number) => {
  if (conversationItems.value[index]) {
    stylizingLoading.value = false
    setTimeout(() => {
      if (refInputTextString.value) {
        refInputTextString.value.focus()
      }
    })
  }

  // 查询是推荐列表
  if (isView.value == false) {
    query_dify_suggested()
  }
}

// 当前索引位置
const currentRenderIndex = ref(0)
// 图表子组件渲染完毕
const onChartReady = (index) => {
  if (index < conversationItems.value.length) {
    // console.log('onChartReady', index)
    currentRenderIndex.value = index
    stylizingLoading.value = false
  }
}

const onRecycleQa = async (index: number) => {
  // 设置当前选中的问答类型
  const item = conversationItems.value[index]
  onAqtiveChange(item.qa_type)

  if (item.qa_type === 'FILEDATA_QA') {
    businessStore.update_file_url(item.file_key)
  }

  // 清空推荐列表
  suggested_array.value = []
  // 发送问题重新生成
  handleCreateStylized(item.question)
  scrollToBottom()
}

// 赞 结果反馈
const onPraiseFeadBack = async (index: number) => {
  const item = conversationItems.value[index]
  const res = await GlobalAPI.fead_back(item.chat_id, 'like')
  if (res.ok) {
    window.$ModalMessage.destroyAll()
    window.$ModalMessage.success('感谢反馈', {
      duration: 1500,
    })
  }
}

// 开始输出时隐藏加载提示
const onBeginRead = async (index: number) => {
  // 设置最上面的滚动提示图标隐藏
  contentLoadingStates.value[currentRenderIndex.value - 1] = false
}

// 踩 结果反馈
const onBelittleFeedback = async (index: number) => {
  const item = conversationItems.value[index]
  const res = await GlobalAPI.fead_back(item.chat_id, 'dislike')
  if (res.ok) {
    window.$ModalMessage.destroyAll()
    window.$ModalMessage.success('感谢反馈', {
      duration: 1500,
    })
  }
}

// 侧边栏对话历史
interface TableItem {
  uuid: string
  key: string
  chat_id: string
}
const tableData = ref<TableItem[]>([])
const tableRef = ref(null)

// 保存对话历史记录
const conversationItems = ref<
  Array<{
    uuid: string
    chat_id: string
    qa_type: string
    question: string
    file_key: string
    role: 'user' | 'assistant'
    reader: ReadableStreamDefaultReader | null
  }>
>([])

// 这里子组件 chart渲染慢需要子组件渲染完毕后通知父组件
const visibleConversationItems = computed(() => {
  return conversationItems.value.slice(0, currentRenderIndex.value + 2)
})
// 这里控制内容加载状态
const contentLoadingStates = ref(
  visibleConversationItems.value.map(() => false),
)

// chat_id定义
const uuids = ref<Record<string, string>>({}) // 改为对象存储不同问答类型的uuid

// 提交对话
const handleCreateStylized = async (send_text = '') => {
  // 滚动到底部
  scrollToBottom()

  // 设置初始化数据标识为false
  isInit.value = false

  // 设置查看历史消息标识为false
  isView.value = false

  // 清空推荐列表
  suggested_array.value = []

  // 若正在加载，则点击后恢复初始状态
  if (stylizingLoading.value) {
      // 停止dify 对话
    // console.log('停止dify 对话', businessStore.$state.task_id)
    await GlobalAPI.stop_chat(businessStore.$state.task_id, qa_type.value)
    onCompletedReader(conversationItems.value.length - 1)
    return
  }

  // 如果输入为空，则直接返回
  if (send_text === '') {
    if (refInputTextString.value && !inputTextString.value.trim()) {
      inputTextString.value = ''
      refInputTextString.value.focus()
      return
    }
  }

  // 如果没有上传文件 表格问答直接提示并返回
  if (
    qa_type.value === 'FILEDATA_QA'
    && `${businessStore.$state.file_url}` === ''
  ) {
    window.$ModalMessage.success('请先上传文件')
    return
  }

  if (showDefaultPage.value) {
    // 新建对话 时输入新问题 清空历史数据
    conversationItems.value = []
    showDefaultPage.value = false
  }

  // 自定义id
  const uuid_str = uuidv4()
  // 加入对话历史用于左边表格渲染
  const newItem = {
    uuid: uuid_str, // 或者根据你的需求计算新的索引
    key: inputTextString.value ? inputTextString.value : send_text,
    chat_id: uuids.value[qa_type.value],
  }
  // 使用 unshift 方法将新元素添加到数组的最前面
  tableData.value.unshift(newItem)

  // 调用大模型后台服务接口
  stylizingLoading.value = true
  const textContent = inputTextString.value
    ? inputTextString.value
    : send_text
  inputTextString.value = ''

  if (!uuids.value[qa_type.value]) {
    uuids.value[qa_type.value] = uuidv4()
  }


  if (textContent) {
    // 存储该轮用户对话消息
    conversationItems.value.push({
      uuid: uuid_str,
      chat_id: uuids.value[qa_type.value],
      qa_type: qa_type.value,
      question: textContent,
      file_key: '',
      role: 'user',
      reader: null,
    })
    // 更新 currentRenderIndex 以包含新添加的项
    currentRenderIndex.value = conversationItems.value.length - 1
    contentLoadingStates.value[currentRenderIndex.value] = true
  }

  const { error, reader, needLogin }
    = await businessStore.createAssistantWriterStylized(
      uuid_str,
      uuids.value[qa_type.value],
      currentChatId.value,
      {
        text: textContent,
        writer_oid: currentChatId.value,
      },
    )

  if (needLogin) {
    message.error('登录已失效，请重新登录')

    // 跳转至登录页面
    setTimeout(() => {
      router.push('/login')
    }, 2000)
  }

  if (error) {
    stylizingLoading.value = false
    onCompletedReader(conversationItems.value.length - 1)
    return
  }

  if (reader) {
    outputTextReader.value = reader
    // 存储该轮AI回复的消息
    conversationItems.value.push({
      uuid: uuid_str,
      chat_id: uuids.value[qa_type.value],
      qa_type: qa_type.value,
      question: textContent,
      file_key: `${businessStore.$state.file_url}`,
      role: 'assistant',
      reader,
    })
    // 更新 currentRenderIndex 以包含新添加的项
    currentRenderIndex.value = conversationItems.value.length - 1
  }

  // 滚动到底部
  scrollToBottom()
}

// 滚动到底部
const scrollToBottom = () => {
  if (isView.value == false) {
    nextTick(() => {
      if (messagesContainer.value) {
        messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
      }
    })
  }
}

const keys = useMagicKeys()
const enterCommand = keys.Enter
const enterCtrl = keys.Enter

const activeElement = useActiveElement()
const notUsingInput = computed(
  () => activeElement.value?.tagName !== 'TEXTAREA',
)

const parser = new UAParser()
const isMacos = parser.getOS().name.includes('Mac')

const placeholder = computed(() => {
  if (stylizingLoading.value) {
    return `输入任意问题...`
  }
  return `输入任意问题, 按 ${
    isMacos ? 'Command' : 'Ctrl'
  } + Enter 键快捷开始...`
})

const generateRandomSuffix = function () {
  return Math.floor(Math.random() * 10000) // 生成0到9999之间的随机整数
}

watch(
  () => enterCommand.value,
  () => {
    if (!isMacos || notUsingInput.value) {
      return
    }

    if (stylizingLoading.value) {
      return
    }

    if (!enterCommand.value) {
      handleCreateStylized()
    }
  },
  {
    deep: true,
  },
)

watch(
  () => enterCtrl.value,
  () => {
    if (isMacos || notUsingInput.value) {
      return
    }

    if (stylizingLoading.value) {
      return
    }

    if (!enterCtrl.value) {
      handleCreateStylized()
    }
  },
  {
    deep: true,
  },
)

// 重置状态
const handleResetState = () => {
  if (isMockDevelopment) {
    inputTextString.value = ''
  } else {
    inputTextString.value = ''
  }

  stylizingLoading.value = false
  nextTick(() => {
    refInputTextString.value?.focus()
  })
  refReaderMarkdownPreview.value?.abortReader()
  refReaderMarkdownPreview.value?.resetStatus()
}
handleResetState()

// 文件上传
const file_name = ref('')
const finish_upload = (res) => {
  file_name.value = res.file.name
  if (res.event.target.responseText) {
    const json_data = JSON.parse(res.event.target.responseText)
    const file_url = json_data.data.object_key
    if (json_data.code === 200) {
      onAqtiveChange('FILEDATA_QA')
      businessStore.update_file_url(file_url)
      window.$ModalMessage.success(`文件上传成功`)
    } else {
      window.$ModalMessage.error(`文件上传失败`)
      return
    }
    const file_name_without_extension = file_name.value.slice(0, file_name.value.lastIndexOf('.')) || file_name.value
    const query_text = `分析${file_name_without_extension}表格数据`
    handleCreateStylized(query_text)
  }
}

// 下面方法用于左侧对话列表点击 右侧内容滚动
// 用于存储每个 MarkdownPreview 容器的引用
// const markdownPreviews = ref<Array<HTMLElement | null>>([]) // 初始化为空数组
const markdownPreviews = ref<Map<string, HTMLElement | null>>(new Map())


// 表格行点击事件
const currentIndex = ref<number | null>(null)
const rowProps = (row: any) => {
  return {
    class: [
      'cursor-pointer select-none',
      currentIndex.value === row.uuid && '[&_.n-data-table-td]:bg-#d5dcff',
    ].join(' '),
    onClick: async () => {
      currentIndex.value = row.uuid
      suggested_array.value = []

      isInit.value = false
      isView.value = true

      // 这里根据chat_id 过滤同一轮对话数据
      fetchConversationHistory(
        isInit,
        conversationItems,
        tableData,
        currentRenderIndex,
        row,
        '',
      )

      // 关闭默认页面
      showDefaultPage.value = false

      //   等待 DOM 更新完成
      await nextTick()
      //  滚动到指定位置
      scrollToItem(row.uuid)
    },
  }
}

// 递归查找最底层的元素
const findDeepestElement = (element: HTMLElement): HTMLElement => {
  if (element.children.length === 0) {
    return element
  }
  return findDeepestElement(element.lastElementChild as HTMLElement)
}

// 设置 markdownPreviews 数组中的元素
const setMarkdownPreview = (uuid: string, role: string, el: any) => {
  if (role === 'user') {
    if (el && el instanceof HTMLElement) {
      // 查找最下面的元素
      const deepestElement = findDeepestElement(el)
      markdownPreviews.value.set(uuid, deepestElement)
    }
  }
}

// 滚动到指定位置的方法
const scrollToItem = async (uuid: string) => {
  // 等待 DOM 更新完成
  await nextTick()
  await nextTick()

  const element = markdownPreviews.value.get(uuid)

  if (element && element instanceof HTMLElement) {
    try {
      // 强制重排，确保元素位置和尺寸正确
      console.log('UUID:', uuid)
      console.log('Element:', element)
      void element.offsetWidth
      element.scrollIntoView({
        behavior: 'smooth',
        block: 'nearest',
        inline: 'nearest',
      })
    } catch (error) {
      console.error('滚动到指定元素时出错:', error)
    }
  }
}

// 默认选中的对话类型
const qa_type = ref('COMMON_QA')
const onAqtiveChange = (val) => {
  qa_type.value = val
  businessStore.update_qa_type(val)

  // 新增：切换类型时生成新uuid
  uuids.value[val] = uuidv4()

  // 清空文件上传历史url
  if (val === 'FILEDATA_QA') {
    businessStore.update_file_url('')
  }
}

// 获取建议问题
const suggested_array = ref([])
const query_dify_suggested = async () => {
  if (!isInit.value) {
    const res = await GlobalAPI.dify_suggested(uuids.value[qa_type.value])
    const json = await res.json()
    if (json?.data?.data !== undefined) {
      suggested_array.value = json.data.data
    }
  }

  // 滚动到底部
  scrollToBottom()
}
// 建议问题点击事件
const onSuggested = (index: number) => {
  // 如果是报告问答的建议问题点击后切换到通用对话
  if (qa_type.value === 'REPORT_QA') {
    onAqtiveChange('COMMON_QA')
  }
  handleCreateStylized(suggested_array.value[index])
}

// 下拉菜单的选项
const options = [
  {
    label: () => h('span', null, '上传文档'),
    icon: () =>
      h('div', {
        class: 'i-vscode-icons:file-type-excel2',
        style: 'inline-block:none',
      }),
    key: 'excel',
  },
  {
    label: () => h('span', null, '上传图片'),
    icon: () =>
      h('div', {
        class: 'i-vscode-icons:file-type-image',
        style: 'inline-block:none',
      }),
    key: 'image',
  },
]

// 下拉菜单选项选择事件处理程序
const uploadRef = useTemplateRef('uploadRef')
function handleSelect(key: string) {
  if (key === 'excel') {
    // 使用 nextTick 确保 DOM 更新完成后执行
    nextTick(() => {
      if (uploadRef.value) {
        // 尝试直接调用 n-upload 的点击方法
        // 如果 n-upload 没有提供这样的方法，可以查找内部的 input 并调用 click 方法
        const fileInput
                    = uploadRef.value.$el.querySelector('input[type="file"]')
        if (fileInput) {
          fileInput.click()
        }
      }
    })
  } else {
    window.$ModalMessage.success('功能开发中', {
      duration: 1500,
    })
  }
}

// 侧边表格滚动条数 动态显示隐藏设置
const scrollableContainer = useTemplateRef('scrollableContainer')

const showScrollbar = () => {
  if (
    scrollableContainer.value
    && scrollableContainer.value.$el
    && scrollableContainer.value.$el.firstElementChild
  ) {
    scrollableContainer.value.$el.firstElementChild.style.overflowY = 'auto'
  }
}

const hideScrollbar = () => {
  if (
    scrollableContainer.value
    && scrollableContainer.value.$el
    && scrollableContainer.value.$el.firstElementChild
  ) {
    scrollableContainer.value.$el.firstElementChild.style.overflowY
            = 'hidden'
  }
}

const searchText = ref('')
const searchChatRef = useTemplateRef('searchChatRef')
const isFocusSearchChat = ref(false)
const onFocusSearchChat = () => {
  isFocusSearchChat.value = true
  nextTick(() => {
    searchChatRef.value?.focus()
  })
}
const onBlurSearchChat = () => {
  if (searchText.value) {
    return
  }
  isFocusSearchChat.value = false
}

// 在script部分添加搜索处理函数
const handleSearch = () => {
  tableData.value = []
  fetchConversationHistory(
    isInit,
    conversationItems,
    tableData,
    currentRenderIndex,
    null,
    searchText.value,
  )
}

const handleClear = () => {
  if (!showDefaultPage.value) {
    newChat()
  }
}

const collapsed = useLocalStorage(
  'collapsed-chat-menu',
  ref(false),
)
</script>

<template>
  <div
    class="flex justify-between items-center h-full"
  >
    <n-layout
      ref="scrollableContainer"
      class="custom-layout h-full"
      has-sider
      :native-scrollbar="true"
      @mouseenter="showScrollbar"
      @mouseleave="hideScrollbar"
    >
      <n-layout-sider
        v-model:collapsed="collapsed"
        collapse-mode="width"
        :collapsed-width="0"
        :width="260"
        :show-collapsed-content="false"
        show-trigger="bar"
        bordered
      >
        <div
          h-full
          class="content"
          flex="~ col"
        >
          <div
            class="header p-20"
            :style="{
              'display': `flex`, /* 使用Flexbox布局 */
              'align-items': `center`, /* 垂直居中对齐 */
              'justify-content': `start`, /* 水平分布空间 */
              'flex-shrink': `0`,
              'position': `sticky`,
              'top': `0`,
              'z-index': `1`,
            }"
          >
            <div
              class="create-chat-box"
              :class="{
                hide: isFocusSearchChat,
              }"
            >
              <n-button
                type="primary"
                icon-placement="left"
                color="#5e58e7"
                strong
                class="create-chat"
                @click="newChat"
              >
                <template #icon>
                  <n-icon>
                    <div class="i-hugeicons:add-01"></div>
                  </n-icon>
                </template>
                新建对话
              </n-button>
            </div>
            <n-input
              ref="searchChatRef"
              v-model:value="searchText"
              placeholder="搜索"
              class="search-chat"
              clearable
              :class="{
                focus: isFocusSearchChat,
              }"
              @click="onFocusSearchChat()"
              @blur="onBlurSearchChat()"
              @input="handleSearch()"
              @keyup.enter="handleSearch()"
              @clear="handleClear()"
            >
              <template #prefix>
                <div class="i-hugeicons:search-01"></div>
              </template>
            </n-input>
          </div>
          <div flex="1 ~ col" class="scrollable-table-container">
            <n-data-table
              ref="tableRef"
              class="custom-table"
              :style="{
                'font-size': `14px`,
                '--n-td-color-hover': `#d5dcff`,
                'font-family': `-apple-system, BlinkMacSystemFont,
              'Segoe UI', Roboto, 'Helvetica Neue', Arial,
              sans-serif`,
              }"
              size="small"
              :bordered="false"
              :bottom-bordered="false"
              :single-line="false"
              :columns="[
                {
                  key: 'key',
                  align: 'left',
                  ellipsis: { tooltip: false },
                },
              ]"
              :data="tableData"
              :loading="isLoadingHistory"
              :row-props="rowProps"
            />
          </div>
          <div
            class="footer"
            style="flex-shrink: 0; height: 200"
          >
            <n-divider style="width: 100%" />
            <n-button
              quaternary
              icon-placement="left"
              type="primary"
              strong
              :style="{
                'width': `200px`,
                'height': `38px`,
                'margin-left': `20px`,
                'margin-bottom': `10px`,
                'align-self': `center`,
                'text-align': `center`,
                'font-family': `-apple-system, BlinkMacSystemFont,
            'Segoe UI', Roboto, 'Helvetica Neue', Arial,
            sans-serif`,
                'font-size': `14px`,
              }"
              @click="openModal"
            >
              <template #icon>
                <n-icon>
                  <div class="i-hugeicons:voice-id"></div>
                </n-icon>
              </template>
              管理对话
            </n-button>

            <TableModal
              :show="isModalOpen"
              @update:show="handleModalClose"
            />
          </div>
        </div>
      </n-layout-sider>
      <n-layout-content class="content">
        <!-- 内容区域 -->
        <div
          flex="~ 1 col"
          min-w-0
          h-full
        >
          <div flex="~ justify-between items-center">
            <NavigationNavBar />
          </div>

          <!-- 这里循环渲染即可实现多轮对话 -->
          <div
            ref="messagesContainer"
            flex="1 ~ col"
            min-h-0
            pb-20
            class="scrollable-container"
            style="background-color: #f6f7fb"
          >
            <!-- 默认对话页面 -->
            <transition name="fade">
              <div v-if="showDefaultPage">
                <DefaultPage />
              </div>
            </transition>

            <template
              v-if="!showDefaultPage"
            >
              <div
                v-for="(item, index) in visibleConversationItems"
                :key="index"
                :ref="(el) => setMarkdownPreview(item.uuid, item.role, el)"
                class="mb-4"
              >
                <div v-if="item.role === 'user'">
                  <div
                    whitespace-break-spaces
                    text-right
                    :style="{
                      'margin-left': `10%`,
                      'margin-right': `10%`,
                      'padding': `15px`,
                      'border-radius': `5px`,
                      'text-align': `center`,
                      'float': `right`,
                    }"
                  >
                    <n-space>
                      <n-tag
                        size="large"
                        :bordered="false"
                        :round="true"
                        :style="{
                          'fontSize': '15px',
                          'fontFamily': 'PMingLiU !important',
                          'color': '#26244c',
                          'max-width': '900px',
                          'text-align': 'left',
                          'padding': '5px 18px',
                          'height': 'auto',
                          // 允许长单词换行到下一行
                          'word-wrap': 'break-word',
                          // 允许在单词内换行
                          'word-break': 'break-all',
                          // 移除默认的 white-space 属性，确保文本能正常换行
                          'white-space': 'normal',
                          // 强制应用样式
                          'overflow': 'visible !important',
                        }"
                        :color="{
                          color: '#e0dfff',
                          borderColor: '#e0dfff',
                        }"
                      >
                        <template #avatar>
                          <div class="size-25 i-my-svg:user-avatar"></div>
                        </template>
                        {{ item.question }}
                      </n-tag>
                    </n-space>
                  </div>
                  <div
                    v-if="contentLoadingStates[index]"
                    class="i-svg-spinners:bars-scale"
                    :style="{
                      'width': `24px`,
                      'height': `24px`,
                      'color': `#b1adf3`,
                      'border-left-color': `#b1adf3`,
                      'margin-top': `80px`,
                      'animation': `spin 1s linear infinite`,
                      'margin-left': `12%`,
                      'float': `left`,
                    }"
                  ></div>
                </div>
                <div v-if="item.role === 'assistant'">
                  <MarkdownPreview
                    :reader="item.reader"
                    :model="defaultLLMTypeName"
                    :isInit="isInit"
                    :isView="isView"
                    :qaType="`${item.qa_type}`"
                    :chart-id="`${index}devID${generateRandomSuffix()}`"
                    :parentScollBottomMethod="scrollToBottom"
                    @failed="() => onFailedReader(index)"
                    @completed="() => onCompletedReader(index)"
                    @chartready="() => onChartReady(index + 1)"
                    @recycleQa="() => onRecycleQa(index)"
                    @praiseFeadBack="() => onPraiseFeadBack(index)"
                    @belittleFeedback="
                      () => onBelittleFeedback(index)
                    "
                    @beginRead="() => onBeginRead(index)"
                  />
                </div>
              </div>
            </template>

            <div
              v-if="!isInit && !stylizingLoading"
              class="w-70% ml-11% mt-[-20] bg-#f6f7fb"
            >
              <SuggestedView
                :labels="suggested_array"
                @suggested="onSuggested"
              />
            </div>
          </div>

          <div
            class="items-center shrink-0 bg-#f6f7fb"
          >
            <div
              relative
              class="flex-1 w-full p-1em"
            >
              <n-space vertical>
                <div
                  flex="~ gap-10"
                  class="h-40 ml-10%"
                >
                  <n-button
                    type="default"
                    :class="[
                      qa_type === 'COMMON_QA' && 'active-tab',
                      'rounded-100 w-120 h-36 p-15 text-13 c-#585a73',
                    ]"
                    @click="onAqtiveChange('COMMON_QA')"
                  >
                    <template #icon>
                      <n-icon size="16">
                        <svg
                          t="1742194713465"
                          class="icon"
                          viewBox="0 0 1024 1024"
                          version="1.1"
                          xmlns="http://www.w3.org/2000/svg"
                          p-id="8188"
                          width="60"
                          height="60"
                        >
                          <path
                            d="M80.867881 469.76534l0.916659 0.916659 79.711097-79.711097L160.655367 389.901467a162.210364 162.210364 0 0 1 229.164631-229.164631l236.345122 236.345122L706.028994 317.332667l-236.345123-236.803452a275.112139 275.112139 0 0 0-388.81599 389.27432z m472.690245-388.81599l-0.916658 0.916659 79.711097 79.711097 0.916659-0.916658A162.210364 162.210364 0 0 1 862.663019 389.901467l-236.345122 236.345122 79.711097 79.711098 236.803452-236.345123a275.112139 275.112139 0 0 0-389.27432-388.81599z m-84.027031 861.506236l0.916659-0.916659-79.711098-79.711097-0.916658 0.916658a162.210364 162.210364 0 0 1-229.164631-229.431989l236.345122-236.345123L317.251198 317.332667l-236.803452 236.345122a275.112139 275.112139 0 0 0 389.27432 388.815991z m99.801197-372.736272a81.811773 81.811773 0 0 0 21.197728-78.794439 80.895115 80.895115 0 0 0-57.59671-57.596711 81.620803 81.620803 0 1 0 36.398982 136.352956z m373.156407-15.659583l-0.916659-0.916659-79.711097 79.711097 0.916658 0.916659a162.248559 162.248559 0 0 1-229.431989 229.431989L396.885907 626.704918 317.251198 706.568792l236.345122 236.803452A275.073945 275.073945 0 0 0 942.374117 554.136119z"
                            fill="#297CE9"
                            p-id="8189"
                          />
                        </svg>
                      </n-icon>
                    </template>
                    深度思考
                  </n-button>
                  <n-button
                    type="default"
                    :class="[
                      qa_type === 'DATABASE_QA' && 'active-tab',
                      'rounded-100 w-120 h-36 p-15 text-13 c-#585a73',
                    ]"
                    @click="onAqtiveChange('DATABASE_QA')"
                  >
                    <template #icon>
                      <n-icon size="20">
                        <svg
                          t="1732505379377"
                          class="icon"
                          viewBox="0 0 1024 1024"
                          version="1.1"
                          xmlns="http://www.w3.org/2000/svg"
                          p-id="22977"
                          width="64"
                          height="64"
                        >
                          <path
                            d="M890.3 755.5C870.7 834.2 704.8 895.6 503 895.6s-367.8-61.4-387.3-140.1h-2V272.8c0-86 174.3-155.7 389.3-155.7s389.3 69.7 389.3 155.7v482.7h-2zM503 148.2c-197.8 0-358.2 55.8-358.2 124.6S305.2 397.4 503 397.4s358.1-55.8 358.1-124.6S700.8 148.2 503 148.2z m358.1 185.6c-59.4 55.6-197.3 94.7-358.1 94.7s-298.8-39-358.2-94.7v94.7c0 68.8 160.4 124.6 358.2 124.6s358.1-55.8 358.1-124.6v-94.7z m0 155.8c-59.4 55.6-197.3 94.7-358.1 94.7s-298.8-39-358.2-94.7v94.7c0 68.8 160.4 124.6 358.2 124.6S861.1 653 861.1 584.2v-94.6z m0 155.7c-59.4 55.6-197.3 94.6-358.1 94.6s-298.8-39-358.2-94.6v94.6c0 68.8 160.4 124.6 358.2 124.6s358.1-55.8 358.1-124.6v-94.6z m-77.8 79.1l31.1-15.6v46.7l-31.1 15.6v-46.7zM175.9 285.9v-18.2c56 48.9 181.3 82.9 327 82.9s271.1-34 327-82.9v18.2c-63.8 47.9-186.3 80.3-327 80.3s-263.2-32.4-327-80.3z"
                            fill=""
                            p-id="22978"
                          />
                        </svg>
                      </n-icon>
                    </template>
                    数据问答
                  </n-button>
                  <n-button
                    type="default"
                    :class="[
                      qa_type === 'FILEDATA_QA' && 'active-tab',
                      'rounded-100 w-120 h-36 p-15 text-13 c-#585a73',
                    ]"
                    @click="onAqtiveChange('FILEDATA_QA')"
                  >
                    <template #icon>
                      <n-icon size="20">
                        <svg
                          t="1732505460059"
                          class="icon"
                          viewBox="0 0 1024 1024"
                          version="1.1"
                          xmlns="http://www.w3.org/2000/svg"
                          p-id="25828"
                          width="64"
                          height="64"
                        >
                          <path
                            d="M858.4 943.9H137.2c-12.7 0-23-10.3-23-23V129c0-12.7 10.3-23 23-23s23 10.3 23 23v768.9h698.2c12.7 0 23 10.3 23 23s-10.3 23-23 23z"
                            fill=""
                            p-id="25829"
                          />
                          <path
                            d="M137 66l37 63h-74zM921 921l-63 37v-74zM287 381h66c17.1 0 31 13.9 31 31v354c0 17.1-13.9 31-31 31h-66c-17.1 0-31-13.9-31-31V412c0-17.1 13.9-31 31-31zM491 193h66c17.1 0 31 13.9 31 31v542c0 17.1-13.9 31-31 31h-66c-17.1 0-31-13.9-31-31V224c0-17.1 13.9-31 31-31zM695 469h66c17.1 0 31 13.9 31 31v266c0 17.1-13.9 31-31 31h-66c-17.1 0-31-13.9-31-31V500c0-17.1 13.9-31 31-31z"
                            fill=""
                            p-id="25830"
                          />
                        </svg>
                      </n-icon>
                    </template>
                    表格问答
                  </n-button>
                  <n-button
                    type="default"
                    :class="[
                      qa_type === 'REPORT_QA' && 'active-tab',
                      'rounded-100 w-120 h-36 p-15 text-13 c-#585a73',
                    ]"
                    @click="onAqtiveChange('REPORT_QA')"
                  >
                    <template #icon>
                      <n-icon size="18">
                        <svg
                          t="1732528323504"
                          class="icon"
                          viewBox="0 0 1024 1024"
                          version="1.1"
                          xmlns="http://www.w3.org/2000/svg"
                          p-id="41739"
                          width="64"
                          height="64"
                        >
                          <path
                            d="M96 896c-8 0-15.5-3.1-21.2-8.8C69.1 881.6 66 874 66 866V445c0-5.5 4.5-10 10-10s10 4.5 10 10v421c0 2.7 1 5.2 2.9 7.1 1.9 1.9 4.4 2.9 7.1 2.9h612c5.5 0 10 4.5 10 10s-4.5 10-10 10H96z m748 0v-20c2.7 0 5.2-1 7.1-2.9 1.9-1.9 2.9-4.4 2.9-7.1v-80c0-5.5 4.5-10 10-10s10 4.5 10 10v80c0 8-3.1 15.5-8.8 21.2-5.6 5.7-13.2 8.8-21.2 8.8z m20-450c-5.5 0-10-4.5-10-10V126c0-5.5-4.5-10-10-10H96c-5.5 0-10 4.5-10 10v193c0 5.5-4.5 10-10 10s-10-4.5-10-10V126c0-16.5 13.4-30 30-30h748c16.5 0 30 13.4 30 30v310c0 5.5-4.5 10-10 10z"
                            fill="#222222"
                            p-id="41740"
                          />
                          <path
                            d="M781 886m-16 0a16 16 0 1 0 32 0 16 16 0 1 0-32 0Z"
                            fill="#222222"
                            p-id="41741"
                          />
                          <path
                            d="M76 383m-16 0a16 16 0 1 0 32 0 16 16 0 1 0-32 0Z"
                            fill="#222222"
                            p-id="41742"
                          />
                          <path
                            d="M84 226h775v20H84zM750 826c-57.2 0-110.9-22.3-151.3-62.7C558.3 722.9 536 669.2 536 612s22.3-110.9 62.7-151.3C639.1 420.3 692.8 398 750 398s110.9 22.3 151.3 62.7C941.7 501.1 964 554.8 964 612s-22.3 110.9-62.7 151.3C860.9 803.7 807.2 826 750 826z m0-408c-107 0-194 87-194 194s87 194 194 194 194-87 194-194-87-194-194-194z"
                            fill="#222222"
                            p-id="41743"
                          />
                          <path
                            d="M901.7 753.2c-1 0-2.1-0.2-3.1-0.5-4.1-1.3-6.9-5.2-6.9-9.5V478.8c0-4.3 2.8-8.2 6.9-9.5 4.1-1.3 8.6 0.1 11.2 3.6 24.9 34 51.4 75.6 51.4 139.1 0 62-22.3 97.3-51.4 137.1-1.9 2.7-4.9 4.1-8.1 4.1z m10.1-241.9v200c17.9-28 29.5-56.4 29.5-99.3-0.1-40.2-11-70.5-29.5-100.7z"
                            fill="#222222"
                            p-id="41744"
                          />
                          <path
                            d="M859 788l93 130"
                            fill="#358AFE"
                            p-id="41745"
                          />
                          <path
                            d="M952 928c-3.1 0-6.2-1.5-8.1-4.2l-93-130c-3.2-4.5-2.2-10.7 2.3-14 4.5-3.2 10.7-2.2 14 2.3l93 130c3.2 4.5 2.2 10.7-2.3 14-1.8 1.3-3.9 1.9-5.9 1.9zM482.4 468.4H171.6c-8.8 0-16-7.2-16-16v-89.8c0-8.8 7.2-16 16-16h310.8c8.8 0 16 7.2 16 16v89.8c0 8.8-7.2 16-16 16z m-306.8-20h302.8v-81.8H175.6v81.8z m306.8-81.8zM384 580H165c-5.5 0-10-4.5-10-10s4.5-10 10-10h219c5.5 0 10 4.5 10 10s-4.5 10-10 10zM455 690H165c-5.5 0-10-4.5-10-10s4.5-10 10-10h290c5.5 0 10 4.5 10 10s-4.5 10-10 10zM525 800H165c-5.5 0-10-4.5-10-10s4.5-10 10-10h360c5.5 0 10 4.5 10 10s-4.5 10-10 10zM183 146c15.5 0 28 12.5 28 28s-12.5 28-28 28-28-12.5-28-28 12.5-28 28-28z m94 0c15.5 0 28 12.5 28 28s-12.5 28-28 28-28-12.5-28-28 12.5-28 28-28z m94 0c15.5 0 28 12.5 28 28s-12.5 28-28 28-28-12.5-28-28 12.5-28 28-28z"
                            fill="#222222"
                            p-id="41746"
                          />
                        </svg>
                      </n-icon>
                    </template>
                    报告问答
                  </n-button>
                </div>
                <n-input
                  ref="refInputTextString"
                  v-model:value="inputTextString"
                  type="textarea"
                  autofocus
                  h-60
                  class="textarea-resize-none text-15"
                  :style="{
                    '--n-border-radius': '15px',
                    '--n-padding-left': '60px',
                    '--n-padding-right': '20px',
                    '--n-padding-vertical': '18px',
                    'width': '80%',
                    'marginLeft': '10%',
                    'align': 'center',
                  }"
                  :placeholder="placeholder"
                  :autosize="{
                    minRows: 1,
                    maxRows: 5,
                  }"
                />
                <div
                  absolute
                  class="translate-y--50% ml-11% top-62%"
                >
                  <n-dropdown
                    :options="options"
                    @select="handleSelect"
                  >
                    <n-icon size="30">
                      <svg
                        t="1729566080604"
                        class="icon"
                        viewBox="0 0 1024 1024"
                        version="1.1"
                        xmlns="http://www.w3.org/2000/svg"
                        p-id="38910"
                        width="64"
                        height="64"
                      >
                        <path
                          d="M856.448 606.72v191.744a31.552 31.552 0 0 1-31.488 31.488H194.624a31.552 31.552 0 0 1-31.488-31.488V606.72a31.488 31.488 0 1 1 62.976 0v160.256h567.36V606.72a31.488 31.488 0 1 1 62.976 0zM359.872 381.248c-8.192 0-10.56-5.184-5.376-11.392L500.48 193.152a11.776 11.776 0 0 1 18.752 0l145.856 176.704c5.184 6.272 2.752 11.392-5.376 11.392H359.872z"
                          fill="#838384"
                          p-id="38911"
                        />
                        <path
                          d="M540.288 637.248a30.464 30.464 0 1 1-61.056 0V342.656a30.464 30.464 0 1 1 61.056 0v294.592z"
                          fill="#838384"
                          p-id="38912"
                        />
                      </svg>
                    </n-icon>
                  </n-dropdown>
                  <!-- 隐藏的文件上传按钮 -->
                  <n-upload
                    ref="uploadRef"
                    type="button"
                    :show-file-list="false"
                    action="sanic/file/upload_file"
                    accept=".xlsx,.xls,.csv"
                    style="display: none"
                    @finish="finish_upload"
                  >
                    选择文件
                  </n-upload>
                </div>
                <n-float-button
                  position="absolute"
                  top="60%"
                  right="11.5%"
                  :type="stylizingLoading ? 'primary' : 'default'"
                  color
                  :class="[
                    stylizingLoading && 'opacity-90',
                    'text-20',
                  ]"
                  style="transform: translateY(-50%)"
                  @click.stop="handleCreateStylized()"
                >
                  <div
                    v-if="stylizingLoading"
                    class="i-svg-spinners:pulse-2 c-#fff"
                  ></div>
                  <div
                    v-else
                    class="flex items-center justify-center c-#303133/60 i-mingcute:send-fill"
                  ></div>
                </n-float-button>
              </n-space>
            </div>
          </div>
        </div>
      </n-layout-content>
    </n-layout>
  </div>
</template>

<style lang="scss" scoped>
.create-chat-box {
  width: 168px;
  overflow: hidden;
  transition: all 0.3s;
  margin-right: 10px;

  &.hide {
    width: 0;
    margin-right: 0;
  }
}

.create-chat {
  width: 100%;
  height: 36px;
  text-align: center;
  font-family: Arial;
  font-weight: bold;
  font-size: 14px;
  border-radius: 20px;
}

.search-chat {
  width: 36px;
  height: 36px;
  text-align: center;
  font-family: Arial;
  font-weight: bold;
  font-size: 14px;
  border-radius: 50%;
  cursor: pointer;

  &.focus {
    width: 100%;
    border-radius: 20px;
  }
}

.scrollable-container {
  overflow-y: auto; // 添加纵向滚动条
  height: 100%;
  padding-bottom: 20px; // 底部内边距，防止内容被遮挡
  background-color: #f6f7fb;

  // background: linear-gradient(to bottom, #f0effe, #f6f7fb);
}

/* 滚动条整体部分 */

::-webkit-scrollbar {
  width: 4px; /* 竖向滚动条宽度 */
  height: 4px; /* 横向滚动条高度 */
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

:deep(.custom-table .n-data-table-thead) {
  display: none;
}

:deep(.custom-table td) {
  color: #26244c;
  font-size: 14px;

  // border: 0px;

  padding: 10px 6px;
  margin: 0 0 12px;
}

.default-page {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh; /* 使容器高度占满整个视口 */
  background-color: #f6f7fb; /* 可选：设置背景颜色 */
}

.active-tab {
  // background: linear-gradient(to left, #f3f2ff, #e1e7fe);

  background: linear-gradient(to left, #f0effe, #d4eefc);
  border-color: #635eed;
  color: #635eed;
}

/* 新建对话框的淡入淡出动画样式 */

.fade-enter-active {
  transition: opacity 1s; /* 出现时较慢 */
}

.fade-leave-active {
  transition: opacity 0s; /* 隐藏时较快 */
}

.fade-enter, .fade-leave-to /* .fade-leave-active in <2.1.8 */ {
  opacity: 0;
}

@keyframes spin {

  0% {
    transform: rotate(0deg);
  }

  100% {
    transform: rotate(360deg);
  }
}

.custom-layout {
  border-top-left-radius: 10px;
  background-color: #fff;
}

.header,
.footer {
  background-color: #fff;
}

.content {
  background-color: #fff;
  padding: 8px;
}

.footer {
  border-bottom-left-radius: 10px;
}

.icon-button {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 38px; /* 可根据需要调整 */
  height: 38px; /* 与宽度相同，形成圆形 */
  border-radius: 100%; /* 圆形 */
  border: 1px solid #e8eaf3;
  background-color: #fff; /* 按钮背景颜色 */
  cursor: pointer;
  transition: background-color 0.3s; /* 平滑过渡效果 */
  position: relative; /* 相对定位 */
}

.icon-button.selected {
  border: 1px solid #a48ef4;
}

.icon-button:hover {
  border: 1px solid #a48ef4; /* 鼠标悬停时的颜色 */
}


/** 自定义对话历史表格滚动条样式 */
.scrollable-table-container {
  overflow-y: hidden; /* 默认隐藏滚动条 */
  height: 100%; /* 根据实际情况调整高度 */
}

.scrollable-table-container:hover {
  overflow-y: auto; /* 鼠标悬停时显示滚动条 */
}

/* 隐藏滚动条轨道 */
.scrollable-table-container::-webkit-scrollbar {
  width: 4px; /* 滚动条宽度 */
}

.scrollable-table-container::-webkit-scrollbar-track {
  background: transparent; /* 滚动条轨道背景透明 */
}

.scrollable-table-container::-webkit-scrollbar-thumb {
  background-color: #bfbfbf; /* 滚动条颜色 */
  border-radius: 3px; /* 滚动条圆角 */
}
</style>
