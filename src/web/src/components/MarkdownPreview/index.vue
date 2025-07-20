<script lang="ts" setup>
import type { TransformStreamModelTypes } from './transform'
import QatypeIcon from '../IconFont/QatypeIcon.vue'
import MarkdownEcharts from './MarkdownEcharts.vue'
import MarkdownTable from './MarkdownTable.vue'
import MarkdownInstance from './plugins/markdown'
import {

  transformStreamValue,
} from './transform'

const props = withDefaults(defineProps<Props>(), {
  isInit: false, // 用于控制 页面渲染速度 初始化时快一点 问答时慢一点
  chartId: '', // 用于区分多个图表实例
  reader: null,
  qaType: '', // 问答类型
  model: 'standard',
  parentScollBottomMethod: () => {},
})

// 自定义事件用于 子父组件传递事件信息
const emit = defineEmits([
  'failed',
  'completed',
  'beginRead',
  'update:reader',
  'chartready',
  'recycleQa',
  'praiseFeadBack',
  'belittleFeedback',
])

const { copy, copyDuration } = useClipText()

interface Props {
  isInit: boolean
  isView: boolean
  chartId: string
  qaType: string
  reader: ReadableStreamDefaultReader<Uint8Array> | null
  model?: TransformStreamModelTypes
  parentScollBottomMethod: () => void // 父组件滚动方法
}

// 解构 props
const { parentScollBottomMethod } = toRefs(props)

// 定义响应式变量
const displayText = ref('')
const textBuffer = ref('')

const readerLoading = ref(false)

const isAbort = ref(false)

const isCompleted = ref(false)

const refWrapperContent = ref<HTMLElement>()

let typingAnimationFrame: number | null = null

// 全局存储
const businessStore = useBusinessStore()

/**
 * reader 读取是否结束
 */
const readIsOver = ref(false)

const renderedMarkdown = computed(() => {
  return MarkdownInstance.render(displayText.value)
})

const renderedContent = computed(() => {
  // 在 renderedMarkdown 末尾插入光标标记
  return `${renderedMarkdown.value}`
})

const abortReader = () => {
  if (props.reader) {
    props.reader.cancel()
  }

  isAbort.value = true
  readIsOver.value = false
  emit('update:reader', null)
  initializeEnd()
  isCompleted.value = true
}

const resetStatus = () => {
  isAbort.value = false
  isCompleted.value = false
  readIsOver.value = false

  emit('update:reader', null)

  initializeEnd()
  displayText.value = ''
  textBuffer.value = ''
  readerLoading.value = false
  if (typingAnimationFrame) {
    cancelAnimationFrame(typingAnimationFrame)
    typingAnimationFrame = null
  }
}

/**
 * 检查是否有实际内容
 */
function hasActualContent(html) {
  const text = html.replace(/<[^>]*>/g, '')
  return /\S/.test(text)
}

const showCopy = computed(() => {
  if (!isCompleted.value) {
    return false
  }

  if (hasActualContent(displayText.value)) {
    return true
  }

  return false
})

const initialized = ref(false)

const initializeStart = () => {
  initialized.value = true
}

const initializeEnd = () => {
  initialized.value = false
}

// 定义图表类型
const currentChartType = ref('')

// 读取数据流
const readTextStream = async () => {
  if (!props.reader) {
    return
  }

  const textDecoder = new TextDecoder('utf-8')
  readerLoading.value = true
  while (true) {
    if (isAbort.value) {
      break
    }
    try {
      if (!props.reader) {
        readIsOver.value = true
        break
      }
      const { value, done } = await props.reader.read()
      if (!props.reader) {
        readIsOver.value = true
        break
      }
      if (done) {
        readIsOver.value = true
        break
      }

      const transformer = transformStreamValue[props.model]
      if (!transformer) {
        break
      }

      const stream = transformer.call(
        transformStreamValue,
        value,
        textDecoder,
      )
      if (stream.done) {
        readIsOver.value = true

        break
      }

      // 每条消息换行显示
      textBuffer.value += stream.content

      if (typingAnimationFrame === null) {
        showText()
      }
    } catch (error) {
      console.log('渲染失败信息', error)
      readIsOver.value = true
      emit('failed', error)
      resetStatus()
      break
    } finally {
      initializeEnd()
    }
  }
}

const scrollToBottom = async () => {
  await nextTick()
  if (!refWrapperContent.value) {
    return
  }

  refWrapperContent.value.scrollTop = refWrapperContent.value.scrollHeight
}
const scrollToBottomByThreshold = async () => {
  if (!refWrapperContent.value) {
    return
  }

  const threshold = 100
  const distanceToBottom
        = refWrapperContent.value.scrollHeight
          - refWrapperContent.value.scrollTop
          - refWrapperContent.value.clientHeight
  if (distanceToBottom <= threshold) {
    scrollToBottom()
  }
}

// 滚动到底部
const scrollToBottomIfAtBottom = async () => {
  scrollToBottomByThreshold()
}

/**
 * 读取 buffer 内容，逐字追加到 displayText
 */
const runReadBuffer = (readCallback = () => {}, endCallback = () => {}) => {
  if (textBuffer.value.length > 0) {
    const lengthToExtract = props.isInit || props.isView ? 1000 : 5
    const nextChunk = textBuffer.value.substring(0, lengthToExtract)
    displayText.value += nextChunk
    textBuffer.value = textBuffer.value.substring(lengthToExtract)
    readCallback()
  } else {
    endCallback()
  }

  // 动态渲染时实时调用父组件滚动条至最底端
  parentScollBottomMethod.value()
}

const showText = () => {
  if (isAbort.value && typingAnimationFrame) {
    cancelAnimationFrame(typingAnimationFrame)
    typingAnimationFrame = null
    readerLoading.value = false
    return
  }

  // 若 reader 还没结束，则保持打字行为
  if (!readIsOver.value) {
    runReadBuffer()
    typingAnimationFrame = requestAnimationFrame(showText)
  } else {
    // 读取剩余的 buffer
    runReadBuffer(
      () => {
        typingAnimationFrame = requestAnimationFrame(showText)
      },
      () => {
        // 这里只有需要显示图表数据时才显示图表
        const dataType = businessStore.writerList.dataType
        if (dataType && dataType === 't04') {
          currentChartType.value
                        = businessStore.writerList.data.template_code
        }

        emit('update:reader', null)
        emit('completed')
        emit('chartready')
        readerLoading.value = false
        isCompleted.value = true
        nextTick(() => {
          readIsOver.value = false
        })
        typingAnimationFrame = null
      },
    )
  }
  scrollToBottomIfAtBottom()
}

watch(
  () => props.reader,
  () => {
    if (props.reader) {
      readTextStream()
    }
  },
  {
    immediate: true,
    deep: true,
  },
)

// 监听 displayText 变化
watch(displayText, (newValue) => {
  if (newValue !== '') {
    emit('beginRead')
  }
})

onUnmounted(() => {
  resetStatus()
})

defineExpose({
  abortReader,
  resetStatus,
  initializeStart,
  initializeEnd,
})

const showLoading = computed(() => {
  if (initialized.value) {
    return true
  }

  if (!props.reader) {
    return false
  }

  if (!readerLoading.value) {
    return false
  }
  if (displayText.value) {
    return false
  }

  return false
})

// 复制文本
const handlePassClip = async () => {
  await copy(displayText.value)
  window.$ModalMessage.destroyAll()
  window.$ModalMessage.success('已复制', {
    duration: copyDuration,
  })
}
// 点赞反馈
const praiseFeedback = () => {
  emit('praiseFeadBack')
}

// 踩 反馈
const belittleFeedback = () => {
  emit('belittleFeedback')
}

// 重新提问
const handleRecycleAquestion = function () {
  emit('recycleQa')
}

// 监控表格图表是否渲染完毕
const onTableCompletedReader = function () {
  emit('chartready')
}
// 监控表格图表是否渲染完毕
const onChartCompletedReader = function () {
  emit('chartready')
}
</script>

<template>
  <n-spin
    relative
    flex="1 ~"
    min-h-0
    w-full
    h-full
    content-class="w-full h-full flex"
    :show="showLoading"
    :rotate="false"
    class="bg-#f6f7fb"
    :style="{
      '--n-opacity-spinning': '0.3',
    }"
  >
    <template #icon>
      <div class="i-svg-spinners:3-dots-rotate"></div>
    </template>
    <!-- b="~ solid #ddd" -->
    <div
      flex="1 ~"
      min-w-0
      min-h-0
      :class="[reader ? '' : 'justify-center items-center']"
    >
      <div
        text-16
        class="w-full h-full overflow-hidden"
        :class="[!displayText && 'flex items-center justify-center']"
      >
        <!-- <n-empty v-if="!displayText" size="large" class="font-bold">
                    <template #icon>
                        <n-icon>
                            <div class="i-hugeicons:ai-chat-02"></div>
                        </n-icon>
                    </template>
                </n-empty> -->

        <div
          v-if="displayText"
          ref="refWrapperContent"
          text-16
          class="w-full h-full overflow-y-auto"
          p-15px
        >
          <div
            class="markdown-wrapper"
            v-html="renderedContent"
          ></div>

          <div
            v-if="readerLoading"
            size-24
            style="margin-left: 10%"
            class="i-svg-spinners:pulse-3"
          ></div>

          <div
            v-if="
              currentChartType
                && currentChartType !== 'temp01'
                && isCompleted
            "
            whitespace-break-spaces
            text-center
            :style="{
              'align-items': `center`,
              'width': `80%`,
              'margin-left': `10%`,
              'margin-right': `10%`,
            }"
          >
            <MarkdownEcharts
              :chart-id="props.chartId"
              @chartRendered="() => onChartCompletedReader()"
            />
          </div>

          <div
            v-if="
              currentChartType
                && currentChartType === 'temp01'
                && isCompleted
            "
            whitespace-break-spaces
            text-center
            :style="{
              'align-items': `center`,
              'width': `80%`,
              'margin-left': `10%`,
              'margin-right': `10%`,
            }"
          >
            <MarkdownTable
              @tableRendered="() => onTableCompletedReader()"
            />
          </div>
          <div
            v-if="isCompleted"
            :style="{
              'background-color': '#ffffff',
              'width': '80%',
              'margin-left': '10%',
              'margin-right': '10%',
              'padding': '18px 15px',
              'display': 'flex',
              'border-bottom-right-radius': '15px',
              'border-bottom-left-radius': '15px',
              'justify-content': 'space-between',
              'margin-top': currentChartType === '' ? '-5px' : '0',
            }"
          >
            <div style="display: flex">
              <QatypeIcon :qa_type="qaType" />
            </div>

            <div style="display: flex">
              <n-button
                icon-placement="left"
                type="default"
                ghost
                size="tiny"
                :bordered="false"
                style="margin-right: 15px"
                @click="praiseFeedback"
              >
                <template #icon>
                  <n-icon size="20">
                    <svg
                      t="1734514601988"
                      class="icon"
                      viewBox="0 0 1024 1024"
                      version="1.1"
                      xmlns="http://www.w3.org/2000/svg"
                      p-id="13495"
                      width="200"
                      height="200"
                    >
                      <path
                        d="M757.76 852.906667c36.906667-0.021333 72.832-30.208 79.296-66.56l51.093333-287.04c10.069333-56.469333-27.093333-100.522667-84.373333-100.096l-10.261333 0.085333a19972.266667 19972.266667 0 0 1-52.842667 0.362667 3552.853333 3552.853333 0 0 1-56.746667 0l-30.997333-0.426667 11.498667-28.8c10.24-25.642667 21.76-95.744 21.504-128.021333-0.618667-73.045333-31.36-114.858667-69.290667-114.410667-46.613333 0.554667-69.461333 23.466667-69.333333 91.136 0.213333 112.661333-102.144 226.112-225.130667 225.109333a1214.08 1214.08 0 0 0-20.629333 0l-3.52 0.042667c-0.192 0 0.64 409.109333 0.64 409.109333 0-0.085333 459.093333-0.490667 459.093333-0.490666z m-17.301333-495.914667a15332.288 15332.288 0 0 0 52.693333-0.362667l10.282667-0.085333c84.010667-0.618667 141.44 67.52 126.72 150.250667L879.061333 793.813333c-10.090667 56.661333-63.68 101.696-121.258666 101.76l-458.922667 0.384A42.666667 42.666667 0 0 1 256 853.546667l-0.853333-409.173334a42.624 42.624 0 0 1 42.346666-42.730666l3.669334-0.042667c5.909333-0.064 13.12-0.064 21.333333 0 98.176 0.789333 182.293333-92.437333 182.144-182.378667C504.469333 128.021333 546.24 86.186667 616.106667 85.333333c65.173333-0.768 111.68 62.506667 112.448 156.714667 0.256 28.48-6.848 78.826667-15.701334 115.050667 8.021333 0 17.28-0.042667 27.584-0.106667zM170.666667 448v405.333333h23.466666a21.333333 21.333333 0 0 1 0 42.666667H154.837333A26.709333 26.709333 0 0 1 128 869.333333v-437.333333c0-14.784 12.074667-26.666667 26.773333-26.666667h38.912a21.333333 21.333333 0 0 1 0 42.666667H170.666667z"
                        fill="#3f3f3f"
                        p-id="13496"
                      />
                    </svg>
                  </n-icon>
                </template>
              </n-button>
              <n-button
                icon-placement="left"
                type="default"
                ghost
                size="tiny"
                :bordered="false"
                style="margin-right: 15px"
                @click="belittleFeedback"
              >
                <template #icon>
                  <n-icon size="20">
                    <svg
                      t="1734514913827"
                      class="icon"
                      viewBox="0 0 1024 1024"
                      version="1.1"
                      xmlns="http://www.w3.org/2000/svg"
                      p-id="22122"
                      width="200"
                      height="200"
                    >
                      <path
                        d="M936.6784 451.56352c21.48352 80.11776-26.24 162.75456-106.40896 184.23808-11.78624 3.1488-24.13056 4.62336-38.81984 4.62336l-100.63872 0.41472c12.50304 56.94976 14.55104 142.49472 14.55104 193.44896 0 59.30496-48.16896 107.4944-107.40224 107.4944-59.21792 0-107.4944-48.18944-107.4944-107.4944v-21.44256c0-118.49728-96.34816-214.89664-214.89664-214.89664a21.46816 21.46816 0 0 1-21.44256-21.4528 21.41696 21.41696 0 0 1 21.44256-21.51936c142.2592 0 257.87392 115.61472 257.87392 257.8688v21.44256c0 35.59424 28.92288 64.52224 64.51712 64.52224 35.54304 0 64.41984-28.92288 64.41984-64.52224 0-96.77824-7.46496-174.21312-20.0192-207.17056a21.44256 21.44256 0 0 1 2.31936-19.86048 21.51936 21.51936 0 0 1 17.66912-9.30304l128.96768-0.512c10.97216 0 19.82464-1.00352 27.83744-3.1488 57.24672-15.36512 91.33056-74.36288 76.01664-131.56864-17.1264-63.9744-104.93952-285.7472-113.2032-306.56-11.62752-19.13856-32.1536-30.56128-55.1168-30.61248-1.29536 0-2.59072-0.1536-3.80416-0.36352H404.51072c-11.86304 0-21.44256-9.65632-21.44256-21.53472a21.4528 21.4528 0 0 1 21.44256-21.44256h322.39104c1.46944 0 2.88768 0.15872 4.28032 0.4096 36.7616 1.50528 70.47168 21.65248 88.77568 53.30944 0.49664 0.91136 0.91648 1.83808 1.34656 2.80064 3.94752 9.91744 96.77824 243.67104 115.37408 312.832zM275.56864 82.21696c11.8784 0 21.53984 9.64608 21.53984 21.44256v472.84736c0 11.8016-9.66144 21.4528-21.53984 21.4528H168.17152c-47.42144 0-85.95456-38.5792-85.95456-85.95456V168.17664c0-47.44192 38.53312-85.95968 85.95456-85.95968h107.39712z m-21.43744 472.77056V125.19424H168.17152c-23.71072 0-42.97728 19.2512-42.97728 42.9824v343.82848c0 23.71072 19.26656 42.9824 42.97728 42.9824h85.95968z"
                        fill="#3f3f3f"
                        p-id="22123"
                      />
                    </svg>
                  </n-icon>
                </template>
              </n-button>

              <n-button
                ghost
                size="tiny"
                icon-placement="left"
                type="default"
                :bordered="false"
                style="margin-right: 15px"
                @click="handlePassClip()"
              >
                <template #icon>
                  <n-icon size="20">
                    <svg
                      t="1734515176870"
                      class="icon"
                      viewBox="0 0 1024 1024"
                      version="1.1"
                      xmlns="http://www.w3.org/2000/svg"
                      p-id="26346"
                      width="200"
                      height="200"
                    >
                      <path
                        d="M955.85804 265.068028l0 595.439364L195.018625 860.507392 195.018625 728.187761 62.698994 728.187761 62.698994 132.748397l760.839415 0 0 132.319631L955.85804 265.068028zM195.018625 695.108365 195.018625 265.068028l595.439364 0 0-99.240235L95.779414 165.827793l0 529.279548L195.018625 695.107341zM922.778644 298.148447 228.099045 298.148447 228.099045 827.427996l694.679599 0L922.778644 298.148447z"
                        fill="#3f3f3f"
                        p-id="26347"
                      />
                    </svg>
                  </n-icon>
                </template>
              </n-button>

              <n-button
                ghost
                :bordered="false"
                icon-placement="left"
                type="default"
                size="tiny"
                @click="handleRecycleAquestion()"
              >
                <template #icon>
                  <n-icon size="22">
                    <svg
                      t="1734598608672"
                      class="icon"
                      viewBox="0 0 1024 1024"
                      version="1.1"
                      xmlns="http://www.w3.org/2000/svg"
                      p-id="60134"
                      width="256"
                      height="256"
                    >
                      <path
                        d="M934.625972 772.390495l-66.949808-130.025379C814.153156 797.841144 670.155555 903.623375 505.826905 903.623375c-174.766372 0-327.506079-120.400161-371.418194-292.809859l27.833929-7.085372c40.686654 159.656233 181.963285 271.148513 343.584266 271.148513 153.86125 0 288.48946-100.409874 336.555176-247.354598l-137.110751 51.179636-10.044774-26.907836 175.818331-65.659419 89.115644 173.096337L934.625972 772.390495zM89.766978 234.477312l-25.927509 12.339026 81.259722 170.634262 176.954201-48.850591-7.631818-27.694759-139.03252 38.356586c53.03182-138.688689 182.650947-230.154867 330.437851-230.154867 156.344814 0 292.572452 102.429881 339.010087 254.889201l27.497261-8.361435c-50.155307-164.636664-197.436698-275.259134-366.507348-275.259134-157.678182 0-296.178583 96.150874-354.877473 242.543012L89.766978 234.477312z"
                        p-id="60135"
                        fill="#3f3f3f"
                      />
                    </svg>
                  </n-icon>
                </template>
              </n-button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </n-spin>
</template>

<style lang="scss">
.markdown-wrapper {
  margin-left: 10%;
  margin-right: 10%;
  background-color: #fff;

  // background: linear-gradient(to right, #f0effe, #d4eefc);

  padding: 1px 18px;
  border-top-right-radius: 15px;
  border-top-left-radius: 15px;
  color: #26244c;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
    'Helvetica Neue', Arial, sans-serif;

  h1 {
    font-size: 2em;
  }

  h2 {
    font-size: 1.5em;
    padding-bottom: 0.3em;
    border-bottom: 1px solid #f6f7fb;
  }

  h3 {
    font-size: 1.25em;
  }

  h4 {
    font-size: 1em;
  }

  h5 {
    font-size: 0.875em;
  }

  h6 {
    font-size: 0.85em;
  }

  h1,
  h2,
  h3,
  h4,
  h5,
  h6 {
    margin: 0 auto;
    line-height: 1.25;
    margin-top: 20px; /* 添加顶部外边距，这里设置为20像素，你可以根据需要调整这个值 */
    margin-bottom: 15px;
  }

  & ul,
  ol {
    padding-left: 1.5em;
    line-height: 0.8;
  }

  & ul,
  li,
  ol {
    list-style-position: outside;
    white-space: normal;
  }

  li {
    line-height: 2;
  }

  ol ol {
    padding-left: 20px;
  }

  ul ul {
    padding-left: 20px;
  }

  hr {
    margin: 16px 0;
  }

  a {
    color: $color-default;
    font-weight: bolder;
    text-decoration: underline;
    padding: 0 3px;
    display: block; // 让 a 标签作为块级元素，实现换行
  }

  p {
    line-height: 2;

    // padding-left: 15px;

    & > code {
      --at-apply: 'bg-#e5e5e5';
      --at-apply: whitespace-pre mx-4px px-6px py-3px rounded-5px;
    }

    img {
      display: inline-block;
    }
  }

  li > p {
    line-height: 2;
  }

  blockquote {
    padding: 10px;
    margin: 20px 0;
    border-left: 5px solid #ccc;
    background-color: #f9f9f9;
    color: #555;

    & > p {
      margin: 0;
    }
  }

  table {
    border-collapse: collapse; /* 合并相邻单元格的边框 */
    width: 100%;
  }

  th,
  td {
    border: 1px solid #f6f7fb; /* 将边框颜色设为红色 */
    padding: 8px;
    text-align: left;
  }

  th {
    background-color: #f2f2f2; /* 可选：给表头设置背景色 */
  }

  code {
    background-color: #f5f5f5;
    color: #333;
  }

  // 为代码块添加样式

  .markdown-code-wrapper {
    max-width: 870px; /* 继承父容器宽度 */
    overflow-x: auto;
    box-sizing: border-box;
    white-space: pre-wrap;
    word-break: break-word; /* 增加断词规则 */
  }

  /* 覆盖pre标签默认样式 */

  pre {
    width: 100%;
    max-width: 100%;
    white-space: pre-wrap;
    word-wrap: break-word;
  }

  /* 添加图片样式，约束宽度 */

  img {
    // height: auto;
    // display: block; // 将图片转为块级元素
    // margin-left: auto; // 左外边距自动
    // margin-right: auto; // 右外边距自动
    // max-width: 100%; // 最大宽度为容器宽度
    // // height: auto; // 高度自适应

    width: 95%; // 设置宽度为视口宽度
    height: auto; // 设置高度为视口高度
    object-fit: cover; // 保持图片比例，覆盖整个容器，可能会裁剪部分图片
    display: block;
    margin: 0; // 移除外边距
  }

  .active-tab {
    // background: linear-gradient(to left, #f3f2ff, #e1e7fe);

    background: linear-gradient(to left, #f0effe, #d4eefc);
    border-color: #635eed;
    color: #635eed;
  }
  // 为代码块添加样式

  .markdown-code-wrapper {
    max-width: 100%; // 指定最大宽度，可按需调整
    margin-left: auto;
    margin-right: auto;
    overflow-x: auto; // 当代码超出最大宽度时显示水平滚动条
  }
  // think {
  //     color: #635eed;
  // }
}
</style>
