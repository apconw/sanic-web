<script lang="tsx" setup>
import { Transformer } from 'markmap-lib'
import { Toolbar } from 'markmap-toolbar'
import { Markmap } from 'markmap-view'
import * as GlobalAPI from '@/api'
import PdfViewer from './PdfViewer.vue'

const transformer = new Transformer()
const initValue = ref(
  `# 大模型测试助手\n1. 上传需求文档\n2. 模型抽取需求\n3. 生成测试用例\n4. 导出测试用例`,
)
const mm = ref()
const svgRef = ref()

const update = () => {
  const { root } = transformer.transform(initValue.value)
  mm.value.setData(root)
  mm.value.fit()
}
const container = ref()
onMounted(() => {
  query_test_assistant_records()

  // 创建 Markmap 实例并传入 opts 参数
  mm.value = Markmap.create(svgRef.value, {
    autoFit: true, // 布尔值，如果为true，则自动调整视图以适应容器大小
    // color: (node) => '#8276f2', // 函数，根据节点返回颜色字符串
    duration: 500, // 数字，动画持续时间，单位毫秒
    embedGlobalCSS: true, // 布尔值，是否嵌入全局CSS样式
    fitRatio: 1, // 数字，适配比例，用于调整自动缩放的程度
    // initialExpandLevel: 1, // 数字，初始展开层级，决定首次加载时展开的节点深度
    lineWidth: (node) => 1, // 函数，根据节点返回线条宽度
    maxInitialScale: 2, // 数字，最大初始缩放比例
    maxWidth: 800, // 数字，思维导图的最大宽度
    nodeMinHeight: 20, // 数字，节点最小高度
    paddingX: 20, // 数字，水平内边距
    pan: true, // 布尔值，允许平移（拖拽）视图
    scrollForPan: true, // 布尔值，当视图到达边界时是否通过滚动来继续平移
    spacingHorizontal: 30, // 数字，水平间距
    spacingVertical: 20, // 数字，垂直间距
    // style: (id) => `#custom-style`, // 函数，基于ID返回自定义样式
    toggleRecursively: true, // 布尔值，是否递归地切换子节点的可见性
    zoom: true, // 布尔值，允许缩放视图
  })

  const { el } = Toolbar.create(mm.value)
  el.style.position = 'absolute'
  el.style.bottom = '1.5rem'
  el.style.right = '1rem'
  el.style.alignItems = 'center' // 垂直居中对齐子元素
  // el.style.border = '1px solid #ccc'
  el.style.display = 'flex' // 使用 flexbox 布局
  el.style.flexDirection = 'row' // 水平排列子元素
  el.style.alignItems = 'center' // 垂直居中对齐子元素
  el.style.width = '120px' // 确保容器宽度为100%
  el.style.justifyContent = 'space-between' // 子元素之间留有空间

  container.value.append(el)

  update()
  // mm.value.handleClick = (e, d) => {
  //     console.log(e, d)
  // }
})

const query_test_assistant_records = async () => {
  const res = await GlobalAPI.query_test_assistant_records(1, 999999)
  const json = await res.json()
  if (json?.data !== undefined) {
    tableData.value = json.data.records
  }
}

onBeforeUnmount(() => {
  if (mm.value && typeof mm.value.destroy === 'function') {
    mm.value.destroy() // 确保Markmap实例被正确销毁
  }
})

const collapsed = ref(false)
const toggleCollapsed = () => {
  collapsed.value = !collapsed.value
  query_test_assistant_records()
}

const loading = ref(false)
const finish_upload = (res) => {
  if (res.event.target.responseText) {
    const json_data = JSON.parse(res.event.target.responseText)
    const kile_key = json_data.data.object_key
    if (json_data.code === 200) {
      window.$ModalMessage.success(`文件上传成功`)
      collapsed.value = true
      loading.value = true
      word_to_md(kile_key)
    } else {
      window.$ModalMessage.error(`文件上传失败`)
    }
  }
}

const word_to_md = async (file_key) => {
  const res = await GlobalAPI.word_to_md(file_key)
  const json = await res.json()
  if (json?.data !== undefined) {
    loading.value = false
    initValue.value = json.data
    update()
  }
}

// 下拉菜单选项选择事件处理程序
const uploadDocRef = ref(null)
function handleDocClick() {
  // 使用 nextTick 确保 DOM 更新完成后执行
  nextTick(() => {
    if (uploadDocRef.value) {
      // 尝试直接调用 n-upload 的点击方法
      // 如果 n-upload 没有提供这样的方法，可以查找内部的 input 并调用 click 方法
      const fileInput
                = uploadDocRef.value.$el.querySelector('input[type="file"]')
      if (fileInput) {
        fileInput.click()
      }
    }
  })
}

const pdfUrl = ref()
// 侧边栏对话历史
interface TableItem {
  id: number
  file_key: string
}
const tableData = ref<TableItem[]>([])
const tableRef = ref(null)
// 表格行点击事件
const rowProps = (row: any) => {
  return {
    onClick: () => {
      initValue.value = row.markdown
      pdfUrl.value = row.file_url
      update()
    },
  }
}

interface RefDocsItem {
  filename: string
  url: string
  page_no: number
  content_pos: {
    page_no: number
    left_top: {
      x: number
      y: number
    }
    right_bottom: {
      x: number
      y: number
    }
  }[]
}
// 准备好要传递给子组件的数据
const pdfDocument = ref<RefDocsItem>({
  filename: 'example.pdf',
  url: 'http://localhost:19000/filedata/%E6%B2%B3%E5%8D%97%E9%9C%80%E6%B1%82.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=sIR5eeDkiwoo779yNJbw%2F20250120%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250120T071545Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=9e960555d54e0d35750297fc83b328b5c15ec81dcf8d8a45f987cbf47cd05dbe', // 替换为实际的PDF文件URL
  page_no: 1, // 指定要打开的页码
  content_pos: [
    {
      page_no: 1,
      left_top: { x: 0, y: 0 },
      right_bottom: { x: 612, y: 792 }, // A4纸大小的默认尺寸
    },
  ],
})
</script>

<script lang="ts"></script>

<template>
  <div size-full>
    <n-space vertical size="large">
      <n-layout
        has-sider
        style="margin-top: 10px; margin-right: 5px; border-radius: 10px"
      >
        <n-layout-sider
          collapse-mode="width"
          :collapsed-width="0"
          :width="260"
          :collapsed="collapsed"
          show-trigger="arrow-circle"
          content-style="padding: 24px;"
          bordered
          style="height: 98vh"
          @update:collapsed="toggleCollapsed"
        >
          <n-layout-header
            class="header flex items-center justify-start shrink-0 sticky top-0 z-1"
          >
            <n-button
              type="primary"
              icon-placement="left"
              color="#5e58e7"
              strong
              class="w-140 h-36 mr-10 mb-10 text-center font-[Arial] font-600 text-14 rounded-20"
              @click="handleDocClick"
            >
              <template #icon>
                <n-icon style="margin-right: 5px">
                  <svg
                    t="1737097386092"
                    class="icon"
                    viewBox="0 0 1024 1024"
                    version="1.1"
                    xmlns="http://www.w3.org/2000/svg"
                    p-id="12487"
                    width="200"
                    height="200"
                  >
                    <path
                      d="M906.672 128H309.328A37.328 37.328 0 0 0 272 165.328V320l352 80 320-80v-154.672A37.328 37.328 0 0 0 906.672 128z"
                      fill="#41A5EE"
                      p-id="12488"
                    />
                    <path
                      d="M944 320H272v192l352 96 320-96V320z"
                      fill="#2B7CD3"
                      p-id="12489"
                    />
                    <path
                      d="M272 512v192l352 112 320-112V512H272z"
                      fill="#185ABD"
                      p-id="12490"
                    />
                    <path
                      d="M309.328 896h597.344A37.328 37.328 0 0 0 944 858.672V704H272v154.672A37.328 37.328 0 0 0 309.328 896z"
                      fill="#103F91"
                      p-id="12491"
                    />
                    <path
                      d="M528 325.28v421.44a27.744 27.744 0 0 1-0.64 6.4A37.024 37.024 0 0 1 490.72 784H272V288h218.72A37.216 37.216 0 0 1 528 325.28z"
                      p-id="12492"
                    />
                    <path
                      d="M544 325.28v389.44A53.792 53.792 0 0 1 490.72 768H272V272h218.72A53.472 53.472 0 0 1 544 325.28z"
                      p-id="12493"
                    />
                    <path
                      d="M528 325.28v389.44A37.216 37.216 0 0 1 490.72 752H272V288h218.72A37.216 37.216 0 0 1 528 325.28z"
                      p-id="12494"
                    />
                    <path
                      d="M512 325.28v389.44A37.216 37.216 0 0 1 474.72 752H272V288h202.72A37.216 37.216 0 0 1 512 325.28z"
                      p-id="12495"
                    />
                    <path
                      d="M64 288m37.328 0l373.344 0q37.328 0 37.328 37.328l0 373.344q0 37.328-37.328 37.328l-373.344 0q-37.328 0-37.328-37.328l0-373.344q0-37.328 37.328-37.328Z"
                      fill="#185ABD"
                      p-id="12496"
                    />
                    <path
                      d="M217.184 574.272q1.104 8.64 1.44 15.056h0.848q0.496-6.08 1.936-14.72t2.8-14.56l39.264-169.376h50.768l40.608 166.848a242.08 242.08 0 0 1 5.072 31.472h0.688a240.288 240.288 0 0 1 4.224-30.448l32.48-167.872h46.208l-56.864 242.656h-53.984L293.92 472.576q-1.68-6.944-3.808-18.112-2.112-11.168-2.624-16.24h-0.672q-0.672 5.92-2.624 17.6t-3.12 17.248l-36.384 160.256h-54.832L132.48 390.672h47.04l35.376 169.728q1.184 5.248 2.288 13.872z"
                      fill="#FFFFFF"
                      p-id="12497"
                    />
                  </svg>
                </n-icon>
              </template>
              上传需求文档
            </n-button>
            <div class="icon-button">
              <n-icon size="17" class="icon">
                <div class="i-hugeicons:search-01"></div>
              </n-icon>
            </div>
          </n-layout-header>
          <n-layout-content class="content">
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
                  key: 'file_key',
                  align: 'left',
                  ellipsis: { tooltip: false },
                },
              ]"
              :data="tableData"
              :row-props="rowProps"
            />
          </n-layout-content>
        </n-layout-sider>
        <n-layout-content>
          <n-spin
            w-full
            h-full
            content-class="w-full h-full flex"
            :show="loading"
            :rotate="false"
            class="bg-#ffffff"
            :style="{ '--n-opacity-spinning': '0' }"
          >
            <div
              ref="container"
              size-full
              flex="~ justify-center items-center"
              class="bg-#f6f7fb"
            >
              <!-- 左边面板：显示 PDF -->
              <div
                clsas="w-50% h-full bg-#f6f7fb"
              >
                <!-- 使用子组件并传递 dcsInfo 数据 -->
                <PdfViewer :dcsInfo="pdfDocument" />
              </div>

              <!-- 右边面板：保持不变 -->
              <div
                flex="~ justify-center items-center"
                class="w-50% h-full bg-#f6f7fb"
              >
                <svg
                  ref="svgRef"
                  style="height: 100%; width: 100%"
                />
              </div>
            </div>
          </n-spin>
        </n-layout-content>
      </n-layout>
    </n-space>
    <n-upload
      ref="uploadDocRef"
      type="button"
      :show-file-list="false"
      action="sanic/file/upload_file"
      accept=".doc, .docx"
      style="display: none"
      @finish="finish_upload"
    >
      选择文件
    </n-upload>
  </div>
</template>

<style lang="scss" scoped>
.icon-button {
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 10px;
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

:deep(.mm-toolbar-brand) {
  display: none !important;
}

:deep(.mm-toolbar-item:hover) {
  background-color: #f5f5f5;
}

:deep(.mm-toolbar-item:active) {
  background-color: #e0e0e0;
}

:deep(.custom-table .n-data-table-thead) {
  display: none;
}

:deep(.custom-table td) {
  color: #26244c;
  font-size: 14px;
  padding: 10px 6px;
  margin: 0 0 12px;
}
</style>
<!--
<template>
    <div class="docWrap">
        <div ref="wordFile"></div>
    </div>
</template>

<script setup>
import { renderAsync } from 'docx-preview'

const wordFile = ref(null)
const docUrl = ref(
    'http://localhost:19000/filedata/input.docx?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=sIR5eeDkiwoo779yNJbw%2F20250119%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250119T121316Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=d5c58187581cf74e7a628c54c8c8b12071eb49f689bd3b74933f0d6883102843'
) // 替换为你的Word文档URL

onMounted(() => {
    const xhr = new XMLHttpRequest()
    xhr.open('get', docUrl.value)
    xhr.responseType = 'blob'
    xhr.onload = function () {
        const blob = new window.Blob([xhr.response], {
            type: 'application/docx'
        })
        renderAsync(blob, wordFile.value).then(() => {
            // 添加事件监听
            wordFile.value.addEventListener('mouseup', handleSelection)
            wordFile.value.addEventListener('keyup', handleSelection)
        })
    }
    xhr.send()
})

const handleSelection = () => {
    const selection = window.getSelection()
    const selectedText = selection.toString().trim()
    if (selectedText) {
        console.log('选中的内容:', selectedText)
        // 可以在这里触发其他事件或进行其他操作
    }
}
</script>

<style scoped>
.docWrap {
    width: 100%;
    height: 600px;
    overflow: auto;
}
</style> -->
