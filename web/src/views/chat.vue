<script lang="tsx" setup>
import { isMockDevelopment } from '@/config'
import { type InputInst } from 'naive-ui'
import { useRouter } from 'vue-router'
import { UAParser } from 'ua-parser-js'
import TableModal from './TableModal.vue'
import DefaultPage from './DefaultPage.vue'
import SuggestedView from './SuggestedPage.vue'

const route = useRoute()
const router = useRouter()
const message = useMessage()
import * as GlobalAPI from '@/api'

// 显示默认页面
const showDefaultPage = ref(true)

//全局存储
const businessStore = useBusinessStore()

//是否是刚登录到系统 批量渲染对话记录
const isInit = ref(false)

// 使用 onMounted 生命周期钩子加载历史对话
onMounted(() => {
    fetchConversationHistory(
        isInit,
        conversationItems,
        tableData,
        currentRenderIndex
    )
})

//管理对话
const isModalOpen = ref(false)
function openModal() {
    isModalOpen.value = true
}
//模态框关闭
function handleModalClose(value) {
    isModalOpen.value = value
    //重新加载对话记录
    fetchConversationHistory(
        isInit,
        conversationItems,
        tableData,
        currentRenderIndex
    )
}

//新建对话
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
}

/**
 * 默认大模型
 */
const defaultLLMTypeName = 'qwen2'
const currentChatId = computed(() => {
    return route.params.chatId
})

//主页面加载提示
const loading = ref(true)
setTimeout(() => {
    loading.value = false
})

//对话等待提示词图标
const stylizingLoading = ref(false)

//输入字符串
const inputTextString = ref('')
const refInputTextString = ref<InputInst | null>()

//输出字符串 Reader 流（风格化的）
const outputTextReader = ref<ReadableStreamDefaultReader | null>()

//markdown对象
const refReaderMarkdownPreview = ref<any>()

//主内容区域
const messagesContainer = ref<HTMLElement | null>(null)

//读取失败
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

//读取完成
const onCompletedReader = (index: number) => {
    if (conversationItems.value[index]) {
        stylizingLoading.value = false
        setTimeout(() => {
            if (refInputTextString.value) {
                refInputTextString.value.focus()
            }
        })
    }

    query_dify_suggested()
    // scrollToBottom()
}

//图表子组件渲染完毕
const currentRenderIndex = ref(0)
const onChartReady = (index) => {
    if (index < conversationItems.value.length) {
        currentRenderIndex.value = index
        stylizingLoading.value = false
    }
}

const onRecycleQa = async (index: number) => {
    //设置当前选中的问答类型
    const item = conversationItems.value[index]
    onAqtiveChange(item.qa_type)

    if (item.qa_type == 'FILEDATA_QA') {
        businessStore.update_file_url(item.file_key)
    }

    // 清空推荐列表
    suggested_array.value = []
    //发送问题重新生成
    handleCreateStylized(item.question)
}

//赞 结果反馈
const onPraiseFeadBack = async (index: number) => {
    const item = conversationItems.value[index]
    const res = await GlobalAPI.fead_back(item.chat_id, 'like')
    if (res.ok) {
        window.$ModalMessage.destroyAll()
        window.$ModalMessage.success('感谢反馈', {
            duration: 1500
        })
    }
}

//踩 结果反馈
const onBelittleFeedback = async (index: number) => {
    const item = conversationItems.value[index]
    const res = await GlobalAPI.fead_back(item.chat_id, 'dislike')
    if (res.ok) {
        window.$ModalMessage.destroyAll()
        window.$ModalMessage.success('感谢反馈', {
            duration: 1500
        })
    }
}

// 侧边栏对话历史
interface TableItem {
    index: number
    key: string
}
const tableData = ref<TableItem[]>([])
const tableRef = ref(null)

//保存对话历史记录
const conversationItems = ref<
    Array<{
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
    return conversationItems.value.slice(0, currentRenderIndex.value + 1)
})

// chat_id定义
const uuid = ref('')
//提交对话
const handleCreateStylized = async (send_text = '') => {
    //设置初始化数据标识为false
    isInit.value = false

    //清空推荐列表
    suggested_array.value = []

    // 若正在加载，则点击后恢复初始状态
    if (stylizingLoading.value) {
        onCompletedReader(conversationItems.value.length - 1)
        return
    }

    //如果输入为空，则直接返回
    if (send_text == '') {
        if (refInputTextString.value && !inputTextString.value.trim()) {
            inputTextString.value = ''
            refInputTextString.value.focus()
            return
        }
    }

    //如果没有上传文件 表格问答直接提示并返回
    if (
        qa_type.value == 'FILEDATA_QA' &&
        `${businessStore.$state.file_url}` === ''
    ) {
        window.$ModalMessage.success('请先上传文件')
        return
    }

    if (showDefaultPage.value) {
        // 新建对话 时输入新问题 清空历史数据
        conversationItems.value = []
        showDefaultPage.value = false
    }

    //加入对话历史用于左边表格渲染
    const newItem = {
        index: tableData.value.length, // 或者根据你的需求计算新的索引
        key: inputTextString.value ? inputTextString.value : send_text
    }
    // 使用 unshift 方法将新元素添加到数组的最前面
    tableData.value.unshift(newItem)

    //调用大模型后台服务接口
    stylizingLoading.value = true
    const textContent = inputTextString.value
        ? inputTextString.value
        : send_text
    inputTextString.value = ''

    uuid.value = uuidv4()
    const { error, reader, needLogin } =
        await businessStore.createAssistantWriterStylized(
            uuid.value,
            currentChatId.value,
            {
                text: textContent,
                writer_oid: currentChatId.value
            }
        )

    if (needLogin) {
        message.error('登录已失效，请重新登录')

        //跳转至登录页面
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
        // 存储该轮对话消息
        conversationItems.value.push({
            chat_id: uuid.value,
            qa_type: qa_type.value,
            question: textContent,
            file_key: `${businessStore.$state.file_url}`,
            role: 'assistant',
            reader: reader
        })
        // 更新 currentRenderIndex 以包含新添加的项
        currentRenderIndex.value = conversationItems.value.length - 1
    }

    // 滚动到底部
    scrollToBottom()
}

// 滚动到底部
const scrollToBottom = () => {
    nextTick(() => {
        if (messagesContainer.value) {
            messagesContainer.value.scrollTop =
                messagesContainer.value.scrollHeight
        }
    })
}

const keys = useMagicKeys()
const enterCommand = keys['Enter']
const enterCtrl = keys['Enter']

const activeElement = useActiveElement()
const notUsingInput = computed(
    () => activeElement.value?.tagName !== 'TEXTAREA'
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
        if (!isMacos || notUsingInput.value) return

        if (stylizingLoading.value) return

        if (!enterCommand.value) {
            handleCreateStylized()
        }
    },
    {
        deep: true
    }
)

watch(
    () => enterCtrl.value,
    () => {
        if (isMacos || notUsingInput.value) return

        if (stylizingLoading.value) return

        if (!enterCtrl.value) {
            handleCreateStylized()
        }
    },
    {
        deep: true
    }
)

//重置状态
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

//文件上传
let file_name = ref('')
const finish_upload = (res) => {
    file_name.value = res.file.name
    if (res.event.target.responseText) {
        let json_data = JSON.parse(res.event.target.responseText)
        let file_url = json_data['data']['object_key']
        if (json_data['code'] == 200) {
            onAqtiveChange('FILEDATA_QA')
            businessStore.update_file_url(file_url)
            window.$ModalMessage.success(`文件上传成功`)
        } else {
            window.$ModalMessage.error(`文件上传失败`)
            return
        }
        var query_text = file_name.value + ' 总结归纳文档的关键信息'
        handleCreateStylized(query_text)
    }
}

// 下面方法用于左侧对话列表点击 右侧内容滚动
// 用于存储每个 MarkdownPreview 容器的引用
const markdownPreviews = ref<Array<HTMLElement | null>>([]) // 初始化为空数组

// 表格行点击事件
const rowProps = (row: any) => {
    return {
        onClick: () => {
            scrollToItem(row.index)
        }
    }
}

// 设置 markdownPreviews 数组中的元素
const setMarkdownPreview = (index: number, el: any) => {
    if (el && el instanceof HTMLElement) {
        // 确保 markdownPreviews 数组的长度与 visibleConversationItems 的长度一致
        if (index >= markdownPreviews.value.length) {
            markdownPreviews.value.push(null)
        }
        markdownPreviews.value[index] = el
    } else if (el && el.value && el.value instanceof HTMLElement) {
        // 处理代理对象的情况
        if (index >= markdownPreviews.value.length) {
            markdownPreviews.value.push(null)
        }
        markdownPreviews.value[index] = el.value
    }
}

// 滚动到指定位置的方法
const scrollToItem = (index: number) => {
    //判断默认页面是否显示或对话历史是否初始化
    if (
        (!showDefaultPage.value && !isInit.value) ||
        conversationItems.value.length === 0
    ) {
        fetchConversationHistory(
            isInit,
            conversationItems,
            tableData,
            currentRenderIndex
        )
    }
    //关闭默认页面
    showDefaultPage.value = false
    if (markdownPreviews.value[index]) {
        markdownPreviews.value[index].scrollIntoView({ behavior: 'smooth' })
    }
}

//默认选中的对话类型
const qa_type = ref('COMMON_QA')
const onAqtiveChange = (val) => {
    qa_type.value = val
    businessStore.update_qa_type(val)

    //清空文件上传历史url
    if (val == 'FILEDATA_QA') {
        businessStore.update_file_url('')
    }
}

//获取建议问题
const suggested_array = ref([])
const query_dify_suggested = async () => {
    if (!isInit.value) {
        const res = await GlobalAPI.dify_suggested(uuid.value)
        const json = await res.json()
        suggested_array.value = json.data.data
    }

    // 滚动到底部
    scrollToBottom()
}
// 建议问题点击事件
const onSuggested = (index: number) => {
    // 如果是报告问答的建议问题点击后切换到通用对话
    if (qa_type.value == 'REPORT_QA') {
        onAqtiveChange('COMMON_QA')
    }
    handleCreateStylized(suggested_array.value[index])
}
</script>
<template>
    <LayoutCenterPanel :loading="loading">
        <template #sidebar-header>
            <n-button
                type="primary"
                icon-placement="left"
                color="#5e58e7"
                @click="newChat"
                strong
                style="
                    width: 180px;
                    height: 38px;
                    margin: 15px;
                    align-self: center;
                    text-align: center;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
                        Roboto, 'Helvetica Neue', Arial, sans-serif;
                    font-weight: bold;
                    font-size: 14px;
                "
            >
                <template #icon>
                    <n-icon style="margin-right: 5px">
                        <div class="i-hugeicons:add-01"></div>
                    </n-icon>
                </template>
                新建对话
            </n-button>
        </template>

        <template #sidebar>
            <n-data-table
                class="custom-table"
                style="
                    --n-td-color-hover: #d5dcff;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
                        Roboto, 'Helvetica Neue', Arial, sans-serif;
                "
                size="small"
                :bordered="false"
                :bottom-bordered="false"
                :columns="[
                    { key: 'key', align: 'left', ellipsis: { tooltip: false } }
                ]"
                :data="tableData"
                ref="tableRef"
                :row-props="rowProps"
            >
                <template #empty>
                    <div></div>
                </template>
            </n-data-table>
        </template>

        <template #sidebar-action>
            <n-divider style="width: 180px" />
            <n-button
                quaternary
                icon-placement="left"
                type="primary"
                strong
                @click="openModal"
                style="
                    width: 150px;
                    height: 38px;
                    align-self: center;
                    text-align: center;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
                        Roboto, 'Helvetica Neue', Arial, sans-serif;
                    font-size: 14px;
                "
            >
                <template #icon>
                    <n-icon>
                        <div class="i-hugeicons:voice-id"></div>
                    </n-icon>
                </template>
                管理对话
            </n-button>

            <TableModal :show="isModalOpen" @update:show="handleModalClose" />
        </template>
        <!-- 内容区域 -->
        <div flex="~ col" h-full style="background-color: #f6f7fb">
            <div flex="~ justify-between items-center">
                <NavigationNavBar />
            </div>

            <!--这里循环渲染即可实现多轮对话-->
            <div
                flex="1 ~ col"
                min-h-0
                pb-20
                class="scrollable-container"
                ref="messagesContainer"
            >
                <!--默认对话页面-->
                <transition name="fade">
                    <div v-if="showDefaultPage">
                        <DefaultPage />
                    </div>
                </transition>

                <div
                    v-if="!showDefaultPage"
                    v-for="(item, index) in visibleConversationItems"
                    :key="index"
                    class="mb-4"
                    :ref="(el) => setMarkdownPreview(index, el)"
                >
                    <MarkdownPreview
                        :reader="item.reader"
                        :model="defaultLLMTypeName"
                        :isInit="isInit"
                        :qaType="`${item.qa_type}`"
                        :chart-id="`${index}devID${generateRandomSuffix()}`"
                        :parentScollBottomMethod="scrollToBottom"
                        @failed="() => onFailedReader(index)"
                        @completed="() => onCompletedReader(index)"
                        @chartready="() => onChartReady(index + 1)"
                        @recycleQa="() => onRecycleQa(index)"
                        @praiseFeadBack="() => onPraiseFeadBack(index)"
                        @belittleFeedback="() => onBelittleFeedback(index)"
                    />
                </div>
                <div
                    v-if="!isInit && !stylizingLoading"
                    style="
                        align-items: center;
                        width: 70%;
                        margin-left: 11%;
                        margin-top: -24px;
                    "
                >
                    <SuggestedView
                        :labels="suggested_array"
                        @suggested="onSuggested"
                    />
                </div>
            </div>

            <div
                style="display: flex; align-items: center"
                flex-basis="10%"
                p="60"
                py="5"
            >
                <div style="margin-top: 40px">
                    <n-upload
                        type="button"
                        :show-file-list="false"
                        action="sanic/file/upload_file"
                        accept=".xlsx,.xls,.csv"
                        class="mr-2"
                        v-on:finish="finish_upload"
                    >
                        <n-icon size="35"
                            ><svg
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
                                ></path>
                                <path
                                    d="M540.288 637.248a30.464 30.464 0 1 1-61.056 0V342.656a30.464 30.464 0 1 1 61.056 0v294.592z"
                                    fill="#838384"
                                    p-id="38912"
                                ></path>
                            </svg>
                        </n-icon>
                    </n-upload>
                </div>
                <div
                    style="
                        position: relative;
                        flex: 1;
                        width: 100%;
                        padding: 1em;
                    "
                >
                    <n-space vertical>
                        <div
                            style="
                                display: flex;
                                gap: 10px;
                                margin-left: 5px;
                                margin-bottom: 5px;
                            "
                        >
                            <n-button
                                type="default"
                                :class="{
                                    'active-tab': qa_type === 'COMMON_QA'
                                }"
                                @click="onAqtiveChange('COMMON_QA')"
                                style="
                                    border-radius: 100px;
                                    width: 100px;
                                    padding: 15px;
                                    height: 20px;
                                    font-size: 12px;
                                    color: #585a73;
                                "
                            >
                                <template #icon>
                                    <n-icon>
                                        <svg
                                            t="1732513350517"
                                            class="icon"
                                            viewBox="0 0 1024 1024"
                                            version="1.1"
                                            xmlns="http://www.w3.org/2000/svg"
                                            p-id="29434"
                                            width="64"
                                            height="64"
                                        >
                                            <path
                                                d="M428.6 2C193.5 2 2.2 193.3 2.2 428.4s191.3 426.4 426.4 426.4S855 663.5 855 428.4 663.8 2 428.6 2z m0 823.4c-218.9 0-397-178.1-397-397s178.1-397 397-397 397 178.1 397 397-178 397-397 397z"
                                                fill="#333234"
                                                p-id="29435"
                                            ></path>
                                            <path
                                                d="M447 751.9c-170.3 0-308.8-145.1-308.8-323.5 0-8.1-6.6-14.7-14.7-14.7s-14.7 6.6-14.7 14.7c0 194.6 151.7 352.9 338.2 352.9 8.1 0 14.7-6.6 14.7-14.7s-6.6-14.7-14.7-14.7z"
                                                fill="#333234"
                                                p-id="29436"
                                            ></path>
                                            <path
                                                d="M527.9 733.5m-16.5 0a16.5 16.5 0 1 0 33 0 16.5 16.5 0 1 0-33 0Z"
                                                fill="#333234"
                                                p-id="29437"
                                            ></path>
                                            <path
                                                d="M1019.7 998.6L810.2 796.5c-5.9-5.6-15.2-5.5-20.8 0.4-5.7 5.8-5.5 15.1 0.3 20.8l209.5 202.2c2.9 2.8 6.5 4.1 10.2 4.1 3.8 0 7.7-1.5 10.6-4.5 5.7-5.9 5.5-15.2-0.3-20.9z"
                                                fill="#333234"
                                                p-id="29438"
                                            ></path>
                                        </svg>
                                    </n-icon>
                                </template>
                                通用问答
                            </n-button>
                            <n-button
                                type="default"
                                :class="{
                                    'active-tab': qa_type === 'DATABASE_QA'
                                }"
                                @click="onAqtiveChange('DATABASE_QA')"
                                style="
                                    border-radius: 100px;
                                    width: 100px;
                                    padding: 15px;
                                    height: 20px;
                                    font-size: 12px;
                                    color: #585a73;
                                "
                            >
                                <template #icon>
                                    <n-icon>
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
                                            ></path>
                                        </svg>
                                    </n-icon>
                                </template>
                                数据问答
                            </n-button>
                            <n-button
                                type="default"
                                :class="{
                                    'active-tab': qa_type === 'FILEDATA_QA'
                                }"
                                @click="onAqtiveChange('FILEDATA_QA')"
                                style="
                                    border-radius: 100px;
                                    width: 100px;
                                    padding: 15px;
                                    height: 20px;
                                    font-size: 12px;
                                    color: #585a73;
                                "
                            >
                                <template #icon>
                                    <n-icon>
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
                                            ></path>
                                            <path
                                                d="M137 66l37 63h-74zM921 921l-63 37v-74zM287 381h66c17.1 0 31 13.9 31 31v354c0 17.1-13.9 31-31 31h-66c-17.1 0-31-13.9-31-31V412c0-17.1 13.9-31 31-31zM491 193h66c17.1 0 31 13.9 31 31v542c0 17.1-13.9 31-31 31h-66c-17.1 0-31-13.9-31-31V224c0-17.1 13.9-31 31-31zM695 469h66c17.1 0 31 13.9 31 31v266c0 17.1-13.9 31-31 31h-66c-17.1 0-31-13.9-31-31V500c0-17.1 13.9-31 31-31z"
                                                fill=""
                                                p-id="25830"
                                            ></path>
                                        </svg>
                                    </n-icon>
                                </template>
                                表格问答
                            </n-button>
                            <n-button
                                type="default"
                                :class="{
                                    'active-tab': qa_type === 'REPORT_QA'
                                }"
                                @click="onAqtiveChange('REPORT_QA')"
                                style="
                                    border-radius: 100px;
                                    width: 100px;
                                    padding: 15px;
                                    height: 20px;
                                    font-size: 12px;
                                    color: #585a73;
                                "
                            >
                                <template #icon>
                                    <n-icon>
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
                                            ></path>
                                            <path
                                                d="M781 886m-16 0a16 16 0 1 0 32 0 16 16 0 1 0-32 0Z"
                                                fill="#222222"
                                                p-id="41741"
                                            ></path>
                                            <path
                                                d="M76 383m-16 0a16 16 0 1 0 32 0 16 16 0 1 0-32 0Z"
                                                fill="#222222"
                                                p-id="41742"
                                            ></path>
                                            <path
                                                d="M84 226h775v20H84zM750 826c-57.2 0-110.9-22.3-151.3-62.7C558.3 722.9 536 669.2 536 612s22.3-110.9 62.7-151.3C639.1 420.3 692.8 398 750 398s110.9 22.3 151.3 62.7C941.7 501.1 964 554.8 964 612s-22.3 110.9-62.7 151.3C860.9 803.7 807.2 826 750 826z m0-408c-107 0-194 87-194 194s87 194 194 194 194-87 194-194-87-194-194-194z"
                                                fill="#222222"
                                                p-id="41743"
                                            ></path>
                                            <path
                                                d="M901.7 753.2c-1 0-2.1-0.2-3.1-0.5-4.1-1.3-6.9-5.2-6.9-9.5V478.8c0-4.3 2.8-8.2 6.9-9.5 4.1-1.3 8.6 0.1 11.2 3.6 24.9 34 51.4 75.6 51.4 139.1 0 62-22.3 97.3-51.4 137.1-1.9 2.7-4.9 4.1-8.1 4.1z m10.1-241.9v200c17.9-28 29.5-56.4 29.5-99.3-0.1-40.2-11-70.5-29.5-100.7z"
                                                fill="#222222"
                                                p-id="41744"
                                            ></path>
                                            <path
                                                d="M859 788l93 130"
                                                fill="#358AFE"
                                                p-id="41745"
                                            ></path>
                                            <path
                                                d="M952 928c-3.1 0-6.2-1.5-8.1-4.2l-93-130c-3.2-4.5-2.2-10.7 2.3-14 4.5-3.2 10.7-2.2 14 2.3l93 130c3.2 4.5 2.2 10.7-2.3 14-1.8 1.3-3.9 1.9-5.9 1.9zM482.4 468.4H171.6c-8.8 0-16-7.2-16-16v-89.8c0-8.8 7.2-16 16-16h310.8c8.8 0 16 7.2 16 16v89.8c0 8.8-7.2 16-16 16z m-306.8-20h302.8v-81.8H175.6v81.8z m306.8-81.8zM384 580H165c-5.5 0-10-4.5-10-10s4.5-10 10-10h219c5.5 0 10 4.5 10 10s-4.5 10-10 10zM455 690H165c-5.5 0-10-4.5-10-10s4.5-10 10-10h290c5.5 0 10 4.5 10 10s-4.5 10-10 10zM525 800H165c-5.5 0-10-4.5-10-10s4.5-10 10-10h360c5.5 0 10 4.5 10 10s-4.5 10-10 10zM183 146c15.5 0 28 12.5 28 28s-12.5 28-28 28-28-12.5-28-28 12.5-28 28-28z m94 0c15.5 0 28 12.5 28 28s-12.5 28-28 28-28-12.5-28-28 12.5-28 28-28z m94 0c15.5 0 28 12.5 28 28s-12.5 28-28 28-28-12.5-28-28 12.5-28 28-28z"
                                                fill="#222222"
                                                p-id="41746"
                                            ></path>
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
                            h-full
                            class="textarea-resize-none text-15"
                            :style="{
                                '--n-border-radius': '20px',
                                '--n-padding-left': '20px',
                                '--n-padding-right': '20px',
                                '--n-padding-vertical': '15px'
                            }"
                            :placeholder="placeholder"
                            :autosize="{
                                minRows: 1,
                                maxRows: 5
                            }"
                        />
                        <n-float-button
                            position="absolute"
                            :right="22"
                            top="62%"
                            :type="stylizingLoading ? 'primary' : 'default'"
                            color
                            :class="[
                                stylizingLoading && 'opacity-90',
                                'text-20'
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
    </LayoutCenterPanel>
</template>

<style lang="scss" scoped>
.scrollable-container {
    overflow-y: auto; // 添加纵向滚动条
    max-height: calc(
        100vh - 120px
    ); // 设置最大高度，确保输入框和导航栏有足够的空间
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
</style>
