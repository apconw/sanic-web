<script lang="tsx" setup>
import type { InputInst, UploadFileInfo } from 'naive-ui'
import { UAParser } from 'ua-parser-js'
import * as GlobalAPI from '@/api'
import { isMockDevelopment } from '@/config'
import DefaultPage from './DefaultPage.vue'
import FileListItem from './FileListItem.vue'
import FileUploadManager from './FileUploadManager.vue'
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
  showDefaultPage.value = true
}

// 新建对话
function newChat() {
  backgroundColorVariable.value = '#ffffff'

  if (showDefaultPage.value) {
    window.$ModalMessage.success(`已经是最新对话`)
    return
  }
  showDefaultPage.value = true
  isInit.value = true
  conversationItems.value = []
  stylizingLoading.value = false
  suggested_array.value = []

  // 清除表格选中状态
  currentIndex.value = null

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
        refInputTextString.value.select()
      }
    })
  }
}

const onCompletedReader = (index: number) => {
  if (conversationItems.value[index]) {
    stylizingLoading.value = false
    setTimeout(() => {
      if (refInputTextString.value) {
        refInputTextString.value.select()
      }
    })
  }

  // 查询是推荐列表
  // if (isView.value === false
  //   && qa_type.value !== 'COMMON_QA'
  //   && qa_type.value !== 'DATABASE_QA') {
  //   //  query_dify_suggested()
  // }
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
  const item = conversationItems.value[index - 1]
  onAqtiveChange(item.qa_type, item.chat_id)


  // 清空推荐列表
  suggested_array.value = []
  // 发送问题重新生成
  handleCreateStylized(item.question, item.file_key)
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
  qa_type: string
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
    role: 'user' | 'assistant'
    reader: ReadableStreamDefaultReader | null
    file_key: {
      source_file_key: string
      parse_file_key: string
      file_size: string
    }[]
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


// 改为对象存储不同问答类型的uuid
const uuids = ref<Record<string, string>>({})

// 校验文件上传状态和业务处理逻辑
const checkAllFilesUploaded = () => {
  const pendingFiles = fileUploadRef.value?.pendingUploadFileInfoList || []

  // 新增：数据问答不支持文件上传
  if (qa_type.value === 'DATABASE_QA' && pendingFiles.length > 0) {
    window.$ModalMessage.warning('数据问答不支持文件上传，请切换到智能问答和表格问答')
    return false
  }
  if (qa_type.value === 'REPORT_QA' && pendingFiles.length > 0) {
    window.$ModalMessage.warning('深度搜索暂不支持文件上传')
    return false
  }

  // 新增：表格问答只支持单个excel文件
  if (qa_type.value === 'FILEDATA_QA') {
    // if (pendingFiles.length > 1) {
    //   window.$ModalMessage.warning('表格问答只支持上传单个文件')
    //   return false
    // }

    if (pendingFiles.length === 1) {
      const file = pendingFiles[0]
      const fileName = file.name?.toLowerCase() || ''
      const isExcelFile = fileName.endsWith('.xlsx') || fileName.endsWith('.xls') || fileName.endsWith('.csv')

      if (!isExcelFile) {
        window.$ModalMessage.warning('表格问答只支持Excel文件格式(.xlsx, .xls ,.csv)')
        return false
      }
    }
  }

  for (const file of pendingFiles) {
    if (file.status !== 'finished') {
      window.$ModalMessage.warning('存在未完成上传或解析失败的文件，请检查后重试')
      return false
    }
  }
  return true
}


// 提交对话
const handleCreateStylized = async (send_text = '', file_key = []) => {
  // if (qa_type.value === 'REPORT_QA') {
  //   window.$ModalMessage.warning('深度搜索功能暂不支持，功能正在开发中..')
  //   inputTextString.value = ''
  //   return
  // }

  // 设置背景颜色
  backgroundColorVariable.value = '#f6f7fb'

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
    await GlobalAPI.stop_chat(businessStore.$state.task_id, qa_type.value)
    onCompletedReader(conversationItems.value.length - 1)
    // 隐藏加载提示动画
    contentLoadingStates.value = contentLoadingStates.value.map(() => false)
    return
  }

  // 如果输入为空，则直接返回
  if (send_text === '') {
    if (refInputTextString.value && !inputTextString.value.trim()) {
      inputTextString.value = ''
      refInputTextString.value?.select()
      return
    }
  }

  let upload_file_list
  // 判断是否有未上传的文件
  if (fileUploadRef.value?.pendingUploadFileInfoList && fileUploadRef.value.pendingUploadFileInfoList.length > 0) {
    // console.log(fileUploadRef.value.pendingUploadFileInfoList)
    // console.log(businessStore.file_list)
    // 有一个文件解析失败不允许提交
    if (!checkAllFilesUploaded()) {
      return
    }
    upload_file_list = businessStore.file_list
  }

  // 点击重新跑时 如果有文件key 则使用文件key
  if (file_key.length > 0) {
    upload_file_list = file_key
  }

  // 表格问答 则使用 上传的文件key实现 上传一次多轮对话的效果
  if (qa_type.value === 'FILEDATA_QA' && businessStore.file_list.length > 0) {
    upload_file_list = businessStore.file_list
  }

  if (showDefaultPage.value) {
    // 新建对话 时输入新问题 清空历史数据
    conversationItems.value = []
    showDefaultPage.value = false

    // 清空文件上传列表
    pendingUploadFileInfoList.value = []
    businessStore.clear_file_list()
  }

  // 自定义id
  const uuid_str = uuidv4()
  // 加入对话历史用于左边表格渲染
  const newItem = {
    uuid: uuid_str, // 或者根据你的需求计算新的索引
    key: inputTextString.value ? inputTextString.value : send_text,
    chat_id: uuids.value[qa_type.value],
    qa_type: qa_type.value,
  }

  // 如果有相同的chat_id 则不添加 使用 unshift 方法将新元素添加到数组的最前面
  const hasSameChatId = tableData.value.some((item) => item.chat_id === uuids.value[qa_type.value])
  if (!hasSameChatId) {
    tableData.value.unshift(newItem)
  }

  // 调用大模型后台服务接口
  stylizingLoading.value = true
  const textContent = inputTextString.value
    ? inputTextString.value
    : send_text
  inputTextString.value = ''

  if (!uuids.value[qa_type.value]) {
    uuids.value[qa_type.value] = uuidv4()
  }

  // 存储该轮用户对话消息
  if (textContent) {
    conversationItems.value.push({
      uuid: uuid_str,
      chat_id: uuids.value[qa_type.value],
      qa_type: qa_type.value,
      question: textContent,
      file_key: upload_file_list,
      role: 'user',
      reader: null,
    })
    // 更新 currentRenderIndex 以包含新添加的项
    currentRenderIndex.value = conversationItems.value.length - 1
    contentLoadingStates.value[currentRenderIndex.value] = true

    // 清空文件上传列表
    pendingUploadFileInfoList.value = []
    businessStore.clear_file_list()
  }

  // 调用大模型
  const { error, reader, needLogin }
    = await businessStore.createAssistantWriterStylized(
      uuid_str,
      uuids.value[qa_type.value],
      currentChatId.value,
      {
        text: textContent,
        writer_oid: currentChatId.value,
        file_list: upload_file_list,
      },
    )

  if (needLogin) {
    message.error('登录已失效，请重新登录')

    // 清空文件上传列表
    pendingUploadFileInfoList.value = []
    businessStore.clear_file_list()

    // 跳转至登录页面
    setTimeout(() => {
      router.push('/login')
    }, 500)
  }

  if (error) {
    stylizingLoading.value = false
    onCompletedReader(conversationItems.value.length - 1)

    // 清空文件上传列表
    pendingUploadFileInfoList.value = []
    businessStore.clear_file_list()
    return
  }

  if (reader) {
    // 存储该轮AI回复的消息
    outputTextReader.value = reader
    conversationItems.value.push({
      uuid: uuid_str,
      chat_id: uuids.value[qa_type.value],
      qa_type: qa_type.value,
      question: textContent,
      file_key: [],
      role: 'assistant',
      reader,
    })

    // 更新 currentRenderIndex 以包含新添加的项
    currentRenderIndex.value = conversationItems.value.length - 1

    // 清空文件上传列表
    pendingUploadFileInfoList.value = []
    businessStore.clear_file_list()
  }

  // 滚动到底部
  scrollToBottom()
}

// 滚动到底部
const scrollToBottom = () => {
  if (isView.value === false) {
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
    refInputTextString.value?.select()
  })
  refReaderMarkdownPreview.value?.abortReader()
  refReaderMarkdownPreview.value?.resetStatus()
}
handleResetState()


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
      currentIndex.value === row.uuid && 'selected-row',
    ].join(' '),
    onClick: async () => {
      backgroundColorVariable.value = '#f6f7fb'

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

      onAqtiveChange(row.qa_type, row.chat_id)
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
const onAqtiveChange = (val, chat_id) => {
  qa_type.value = val
  businessStore.update_qa_type(val)


  // 新增：切换类型时生成新uuid
  if (chat_id) {
    uuids.value[val] = chat_id
  } else {
    uuids.value[val] = uuidv4()
  }

  // 清空文件上传历史url
  // if (val === 'FILEDATA_QA') {
  //   businessStore.update_file_url('')
  // }
}

// 获取建议问题
const suggested_array = ref([])
// const query_dify_suggested = async () => {
//   if (!isInit.value) {
//     const res = await GlobalAPI.dify_suggested(uuids.value[qa_type.value])
//     const json = await res.json()
//     if (json?.data?.data !== undefined) {
//       suggested_array.value = json.data.data
//     }
//   }

//   // 滚动到底部
//   scrollToBottom()
// }
// 建议问题点击事件
const onSuggested = (index: number) => {
  // 如果是报告问答的建议问题点击后切换到通用对话
  if (qa_type.value === 'REPORT_QA') {
    onAqtiveChange('COMMON_QA', '')
  }
  handleCreateStylized(suggested_array.value[index])
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
  if (!showDefaultPage.value) {
    newChat()
  }
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

// 背景颜色 默认页面和内容页面动态调整
const backgroundColorVariable = ref('#ffffff')


// 添加一键滚动到底部功能的相关代码
const showScrollToBottom = ref(false)
const scrollThreshold = 1000 // 滚动超过100px时显示按钮

// 用户点击图标滚动到底部
const clickScrollToBottom = () => {
  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
      showScrollToBottom.value = false // 滚动到底部后隐藏按钮
    }
  })
}

// ======新增：检查是否需要显示滚动到底部按钮==========//
const checkScrollPosition = () => {
  if (messagesContainer.value) {
    const { scrollTop, scrollHeight, clientHeight } = messagesContainer.value
    const isAtBottom = scrollTop + clientHeight >= scrollHeight - 10 // 10px的容差
    showScrollToBottom.value = !isAtBottom && scrollTop > scrollThreshold
  }
}
// 新增：监听滚动事件
const handleScroll = () => {
  checkScrollPosition()
}

// 在 onMounted 或 onBeforeMount 中添加事件监听
onMounted(() => {
  if (messagesContainer.value) {
    messagesContainer.value.addEventListener('scroll', handleScroll)
  }
})

// 在组件卸载前移除事件监听
onBeforeUnmount(() => {
  if (messagesContainer.value) {
    messagesContainer.value.removeEventListener('scroll', handleScroll)
  }
})

// ============================== 文件上传 ============================//
interface FileUploadRef {
  pendingUploadFileInfoList: UploadFileInfo[] | null | undefined
  options?: any[]
  reset?: () => void
}
const fileUploadRef = ref<FileUploadRef | null>(null)

// 用于绑定文件上传信息列表
const pendingUploadFileInfoList = ref([])
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
        show-trigger="arrow-circle"
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
                color="#3e2cff"
                strong
                class="create-chat"
                :disabled="stylizingLoading"
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
                'font-size': `20px`,
                'fontcolor': `red`,
                '--n-td-color': `#ffffff`,
                'font-family': `-apple-system, BlinkMacSystemFont,'Segoe UI', Roboto, 'Helvetica Neue', Arial,sans-serif`,
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
            style="flex-shrink: 0"
          >
            <n-divider
              style="width: calc(100% - 60px); margin-left: 25px; margin-right: 35px;

--n-color: #e8eaf2;"
            />
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
            <NavigationNavBar :background-color="backgroundColorVariable" />
          </div>

          <!-- 这里循环渲染即可实现多轮对话 -->
          <div
            ref="messagesContainer"
            flex="1 ~ col"
            min-h-0
            pb-20
            class="scrollable-container"
            :style="{ backgroundColor: backgroundColorVariable }"
            @scroll="handleScroll"
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
                <div v-if="item.role === 'user'" class="flex flex-col items-end space-y-2 w-full">
                  <!-- 用户消息 -->
                  <div
                    :style="{
                      'margin-left': `10%`,
                      'margin-right': `10%`,
                      'padding': `15px`,
                      'border-radius': `5px`,
                      'text-align': `center`,
                      'max-width': '80%', // 控制宽度避免撑满
                    }"
                  >
                    <n-space>
                      <n-tag
                        size="large"
                        :bordered="false"
                        :round="true"
                        :style="{
                          'fontSize': '16px',
                          'fontFamily': `-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji'`,
                          'fontWeight': '400',
                          'color': '#26244c',
                          'max-width': '600px',
                          'text-align': 'left',
                          'padding': '5px 18px',
                          'height': 'auto',
                          'line-height': 1.5,
                          'word-wrap': 'break-word',
                          'word-break': 'break-all',
                          'white-space': 'pre-wrap',
                          'overflow': 'visible',
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

                  <!-- 用户上传的文件列表 -->
                  <div
                    v-if="item.file_key && item.file_key.length > 0"
                    class="upload-wrapper-list flex flex-wrap gap-10 items-center pb-5"
                    style="margin-left: 10%; margin-right: 10.5%; width: 80%; justify-content: flex-end;"
                  >
                    <FileListItem
                      v-for="(file, fileIndex) in item.file_key"
                      :key="fileIndex"
                      :file="file"
                    />
                  </div>

                  <!-- 加载动画：紧跟在消息下方，但对齐到左边 -->
                  <div
                    v-if="contentLoadingStates[index]"
                    class="i-svg-spinners:bars-scale"
                    :style="{
                      'width': `24px`,
                      'height': `24px`,
                      'color': `#b1adf3`,
                      'border-left-color': `#b1adf3`,
                      'animation': `spin 1s linear infinite`,
                      'margin-top': '10px',
                      'align-self': 'flex-start', // 让此元素在交叉轴（水平轴）上靠左对齐
                      'margin-left': '12%', // 与上面的消息保持一致的缩进
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
            v-show="showScrollToBottom"
            class="scroll-to-bottom-btn"
            @click="clickScrollToBottom"
          >
            <div class="i-mingcute:arrow-down-fill"></div>
          </div>

          <div
            :class="['items-center', 'shrink-0', `bg-${backgroundColorVariable}`]"
          >
            <div
              relative
              class="flex-1 w-full p-1em"
            >
              <n-space
                vertical
                class="mx-10%"
              >
                <div
                  flex="~ gap-10"
                  class="h-40"
                >
                  <n-button
                    type="default"
                    :class="[
                      qa_type === 'COMMON_QA' && 'active-tab',
                      'rounded-100 w-120 h-36 p-15 text-13 c-#585a73',
                    ]"
                    @click="onAqtiveChange('COMMON_QA', '')"
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
                    智能问答
                  </n-button>
                  <n-button
                    type="default"
                    :class="[
                      qa_type === 'DATABASE_QA' && 'active-tab',
                      'rounded-100 w-120 h-36 p-15 text-13 c-#585a73',
                    ]"
                    @click="onAqtiveChange('DATABASE_QA', '')"
                  >
                    <template #icon>
                      <n-icon size="19">
                        <svg t="1754035667476" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="50983" width="60" height="60"><path d="M512 102.6c110.7 0 215 12.3 293.9 34.7 35.8 10.2 65 22.1 84.5 34.7 18.6 12 21.3 19.7 21.6 20.6-0.2 0.9-3 8.6-21.6 20.6-19.5 12.5-48.7 24.5-84.5 34.7-78.9 22.3-183.2 34.7-293.9 34.7s-215-12.3-293.9-34.7c-35.8-10.2-65-22.1-84.5-34.7-18.6-12-21.3-19.7-21.6-20.6 0.2-0.9 3-8.6 21.6-20.6 19.5-12.5 48.7-24.5 84.5-34.7 78.9-22.4 183.2-34.7 293.9-34.7m0-40c-243 0-440 58.2-440 130s197 130 440 130 440-58.2 440-130-197-130-440-130zM112 190.4H72v641h40v-641z m840-0.3h-40v641h40v-641zM912 831v0.5c-0.2 0.9-3 8.6-21.6 20.6-19.5 12.5-48.7 24.5-84.5 34.7-78.9 22.3-183.2 34.6-293.9 34.6s-215-12.3-293.9-34.7c-35.8-10.2-65-22.1-84.5-34.7-18.6-12-21.3-19.7-21.6-20.6v-0.3l-40 0.3v0.1c0 71.8 197 130 440 130s440-58.2 440-130v-0.4l-40-0.1z m0-210.5v0.5c-0.2 0.9-3 8.6-21.6 20.6-19.5 12.5-48.7 24.5-84.5 34.7C727 698.6 622.7 711 512 711s-215-12.3-293.9-34.7c-35.8-10.2-65-22.1-84.5-34.7-18.6-12-21.3-19.7-21.6-20.6v-0.3l-40 0.3v0.1c0 71.8 197 130 440 130s440-58.2 440-130v-0.4l-40-0.2z m0-221.5v0.5c-0.2 0.9-3 8.6-21.6 20.6-19.5 12.5-48.7 24.5-84.5 34.7-78.9 22.3-183.2 34.7-293.9 34.7s-215-12.3-293.9-34.7c-35.8-10.2-65-22.1-84.5-34.7-18.6-12-21.3-19.7-21.6-20.6v-0.3l-40 0.3v0.1c0 71.8 197 130 440 130s440-58.2 440-130v-0.4l-40-0.2z" fill="" p-id="50984" /></svg>
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
                    @click="onAqtiveChange('FILEDATA_QA', '')"
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
                    @click="onAqtiveChange('REPORT_QA', '')"
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
                    深度搜索
                  </n-button>
                </div>
                <div
                  :class="[
                    'relative b b-solid b-primary bg-white',
                    'rounded-10px p-12',
                  ]"
                >
                  <FileUploadManager
                    ref="fileUploadRef"
                    v-model="pendingUploadFileInfoList"
                  />

                  <n-input
                    ref="refInputTextString"
                    v-model:value="inputTextString"
                    type="textarea"
                    class="textarea-resize-none w-full text-15 [&_.n-input\_\_border]:hidden [&_.n-input\_\_state-border]:hidden [&_.n-input-wrapper]:p-0!"
                    :style="{
                      '--n-border-radius': '15px',
                      'align': 'center',
                      'font-family': `-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji'`,
                      'font-size': '16px',
                      'line-height': '1.5',
                    }"
                    :placeholder="placeholder"
                    :autosize="{
                      minRows: 1,
                      maxRows: 10,
                    }"
                  >
                    <template
                      #prefix
                    >
                      <n-dropdown
                        :options="fileUploadRef?.options || []"
                      >
                        <div flex="~ items-center justify-center" class="rounded-50% p-7 hover:bg-primary/5 transition-all-300 bg-primary/1" b="~ solid primary/20">
                          <div class="text-20  i-uil:upload cursor-pointer"></div>
                        </div>
                        <!-- <n-icon size="30">
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
                        </n-icon> -->
                      </n-dropdown>
                    </template>
                  </n-input>

                  <n-float-button
                    position="absolute"
                    :type="stylizingLoading ? 'primary' : 'default'"
                    color
                    bottom="10px"
                    right="8px"
                    :class="[
                      stylizingLoading && 'opacity-90',
                      'text-20',
                    ]"
                    @click.stop="handleCreateStylized()"
                  >
                    <div
                      v-if="stylizingLoading"
                      class="i-svg-spinners:pulse-2 c-#fff text-20"
                    ></div>
                    <div
                      v-else
                      class="flex items-center justify-center i-mingcute:send-fill text-20 cursor-pointer transition-colors duration-300 hover:c-primary/80"
                    ></div>
                  </n-float-button>
                </div>
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

:deep(.custom-table .n-data-table-table) {
  border-collapse: collapse;
}

:deep(.custom-table .n-data-table-th),
:deep(.custom-table .n-data-table-td) {
  border: none;
}

:deep(.custom-table td) {
  color: #113;
  padding: 12px 30px;
  background-color:  #fff;
  transition: background-color 0.3s cubic-bezier(0.4, 0, 0.2, 1);

  /* 优化后的系统字体栈：优先使用系统原生字体 */

  font-family:
    /* macOS */ -apple-system,
    /* Windows */ BlinkMacSystemFont,
    /* 通用系统UI */ 'Segoe UI',
    /* 开源跨平台 */ Roboto,
    /* Linux */ Oxygen, Ubuntu, Cantarell,
    /* fallback */ 'Open Sans', 'Helvetica Neue', Arial,
    /* 终极兜底 */ sans-serif,
    /* 现代浏览器推荐 */ system-ui,
    /* 苹果新字体支持 */ "SF Pro Text";

  /* 可选：基础字体大小与行高，提升可读性 */

  font-size: 14px;

  /* 优化字体渲染 */

  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-rendering: optimizelegibility;
}

:deep(.custom-table .selected-row td) {
  color: #615ced !important;
  font-weight: bold;
  padding: 12px 30px !important;
  background: linear-gradient(to bottom, #fff, #f9f9ff);
  transform: scale(1.001);
  transition: all 0.3s ease;
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
  border-right:1px solid #f6f7fb;
  background-color: #fff;

  // padding: 8px;
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
  background-color: #fff;
  transition: background-color 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.scrollable-table-container:hover {
  overflow-y: auto; /* 鼠标悬停时显示滚动条 */
}

/* 隐藏滚动条轨道 */

.scrollable-table-container::-webkit-scrollbar {
  width: 5px; /* 滚动条宽度 */
}

.scrollable-table-container::-webkit-scrollbar-track {
  background: transparent; /* 滚动条轨道背景透明 */
}

.scrollable-table-container::-webkit-scrollbar-thumb {
  background-color: #e8eaf3; /* 滚动条颜色 */
  border-radius: 4px; /* 滚动条圆角 */
}

/* 一键到底部按钮样式，底部居中显示 */

.scroll-to-bottom-btn {
  position: absolute;
  bottom: 145px; /* 距离底部的距离 */
  left: 50%;
  transform: translateX(-50%); /* 水平居中 */
  width: 30px;
  height: 30px;
  border-radius: 50%;
  background-color: #fff;
  box-shadow: 0 4px 15px rgb(0 0 0 / 20%);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  z-index: 100;
  transition: all 0.3s ease;
  border: 1px solid #e8eaf3;
  backdrop-filter: blur(5px);
}

.scroll-to-bottom-btn:hover {
  background-color: #f6f7fb;
  transform: translateX(-50%) scale(1.1); /* 悬停时放大 */
  box-shadow: 0 6px 20px rgb(0 0 0 / 25%);
}

.scroll-to-bottom-btn::before {
  content: "";
  position: absolute;
  width: 200%;
  height: 200%;
  top: -50%;
  left: -50%;
  border-radius: 50%;
  animation: pulse 2s infinite;
}

@keyframes pulse {

  0% {
    transform: scale(0.5);
    opacity: 0;
  }

  50% {
    transform: scale(1);
    opacity: 0.2;
  }

  100% {
    transform: scale(1.5);
    opacity: 0;
  }
}

.upload-wrapper-list {
  --at-apply: flex flex-wrap gap-10 items-center;
  --at-apply: pb-12;
}
</style>
