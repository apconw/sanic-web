<script setup lang="ts">
import type { DataTableColumns } from 'naive-ui'
import { useBusinessStore } from '@/store/business'

// 每页显示条目数

// 自定义事件用于 子父组件传递事件信息
const emit = defineEmits(['tableRendered'])
const businessStore = useBusinessStore()
const tableData = ref(businessStore.writerList.data.data || [])
const currentPage = ref(1) // 当前页
const pageSize = ref(5)// 分页设置
const pagination = computed(() => ({
  page: currentPage.value,
  pageSize: pageSize.value,
  total: tableData.value.length, // 总条目数
  onPageChange: (page: number) => {
    currentPage.value = page
  },
  onPageSizeChange: (size: number) => {
    pageSize.value = size
  },
}))

// 计算 scrollX 的值
const scrollX = computed(() => {
  if (tableData.value.length > 0) {
    const keys = Object.keys(tableData.value[0])
    const totalWidth = keys.length * 120 // 每列宽度120px
    return totalWidth
  }
  return 0
})

const minRowHeight = 48
const heightForRow = () => 48

// 动态生成表格列定义
const columns = computed<DataTableColumns>(() => {
  if (tableData.value.length > 0) {
    const keys = Object.keys(tableData.value[0])
    return keys.map((key, index) => ({
      title: key,
      key,
      width: 120, // 每列宽度调整为 120px
      // 固定前三个列
      fixed: index < 3 ? 'left' : undefined,
      render(row: any) {
        return row[key]
      },
    }))
  }
  return []
})

// 根据当前页和每页大小计算分页数据
const pagedTableData = computed(() => {
  const start = (currentPage.value - 1) * pageSize.value
  const end = start + pageSize.value
  return tableData.value.slice(start, end)
})

onMounted(() => {
  emit('tableRendered') // 触发父组件事件通知已渲染完毕
  businessStore.clearWriterList()
})
</script>

<template>
  <div style="background-color: #fff">
    <n-card
      title="表格"
      embedded
      bordered
      :content-style="{ 'background-color': '#ffffff' }"
      :header-style="{
        'color': '#26244c',
        'height': '10px',
        'background-color': '#f0effe',
        'text-align': 'left',
        'font-size': '14px',
        'font-family': 'PMingLiU',
      }"
      :footer-style="{
        'color': '#666',
        'background-color': '#ffffff',
        'text-align': 'left',
        'font-size': '14px',
        'font-family': 'PMingLiU',
      }"
    >
      <div
        flex="~ space-between"
        mb-10
      ></div>
      <n-data-table
        :style="{
          // 调整高度
          'height': `auto`, 
          'width': `98%`,
          'margin': `0 5px`,
          'background-color': `#fff`,
          // 确保横向滚动条在需要时显示
          'overflow-x': 'auto' 
        }"
        :columns="columns"
        :data="pagedTableData"
        :pagination="pagination"
        :scroll-x="scrollX" 
        :min-row-height="minRowHeight"
        :height-for-row="heightForRow"
        :header-height="48"
      />
      <template #footer>
        数据来源: 大模型生成的数据, 以上信息仅供参考
      </template>
    </n-card>
  </div>
</template>
