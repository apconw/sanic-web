<script setup>
import * as GlobalAPI from '@/api'

const props = defineProps({
  show: Boolean,
})

const emit = defineEmits(['update:show', 'delete'])

const localShow = ref(props.show)
const tableData = ref([])
const columns = ref([
  {
    type: 'selection',
  },
  {
    title: '用户问题',
    key: 'question',
    ellipsis: true,
  },
  {
    title: '创建时间',
    key: 'create_time',
  },
])
const loading = ref(false)
const checkedRowKeys = ref([])

// 分页配置
const pagination = ref({
  page: 1,
  pageSize: 8,
  total: 0, // 总记录数
  pageCount: 0, // 总页数
  onChange: (page) => handlePageChange(page),
  onUpdatePageSize: (pageSize) => handlePageSizeChange(pageSize),
})

async function fetchData() {
  loading.value = true
  try {
    const res = await GlobalAPI.query_user_qa_record(
      pagination.value.page,
      pagination.value.pageSize,
    )
    if (res.ok) {
      const data = await res.json()
      if (data && data.data) {
        tableData.value = data.data.records || []
        pagination.value.total = data.data.total_count || 0
        pagination.value.pageCount = data.data.total_pages || 0
      } else {
        console.error('Unexpected data format:', data)
        tableData.value = []
        pagination.value.total = 0
        pagination.value.pageCount = 0
      }
    } else {
      console.error('API request failed with status:', res.status)
      tableData.value = []
      pagination.value.total = 0
      pagination.value.pageCount = 0
    }
  } catch (error) {
    console.error('Error fetching data:', error)
    tableData.value = []
    pagination.value.total = 0
    pagination.value.pageCount = 0
  } finally {
    loading.value = false
  }
}
function close() {
  localShow.value = false
  emit('update:show', false)
  pagination.value.page = 1
}

const rowKey = (row) => row.id

function handleCheck(rowKeys) {
  checkedRowKeys.value = rowKeys
}

async function deleteSelectedData() {
  if (checkedRowKeys.value.length === 0) {
    return
  }
  const res = await GlobalAPI.delete_user_record(checkedRowKeys.value)
  if (res.ok) {
    fetchData()
  }
}

function handlePageChange(page) {
  pagination.value.page = page
  fetchData()
}

function handlePageSizeChange(newPageSize) {
  pagination.value.pageSize = newPageSize
  pagination.value.page = 1 // 重置到第一页
  fetchData()
}

const modalTitle = computed(
  () => `管理对话记录 · 共${pagination.value.total}条`,
)

watch(
  () => props.show,
  (newVal) => {
    if (newVal !== localShow.value) {
      localShow.value = newVal
      if (newVal) {
        fetchData()
      }
    }
  },
)

const tableRef = useTemplateRef('tableRef')
</script>

<template>
  <n-modal
    v-model:show="localShow"
    :mask-closable="false"
    :on-after-leave="close"
    preset="card"
    :title="modalTitle"
    class="w-900 h-600 flex flex-col"
  >
    <div
      class="modal-content"
      style="flex: 1; display: flex; flex-direction: column"
    >
      <n-spin :show="loading" style="flex: 1; overflow: auto">
        <n-data-table
          ref="tableRef"
          :data="tableData"
          :columns="columns"
          :row-key="rowKey"
          :checked-row-keys="checkedRowKeys"
          style="height: 100%; width: 100%"
          @update:checked-row-keys="handleCheck"
        >
          <template #bodyCell="{ column, row }">
            <td :key="column.key">{{ row[column.key] }}</td>
          </template>
        </n-data-table>
      </n-spin>
      <div
        class="footer"
        style="
display: flex;
justify-content: space-between;
align-items: center;
padding: 10px;
background-color: var(--n-modal-footer-bg);
border-top: 1px solid var(--n-modal-border-color);
                "
      >
        <n-pagination
          v-model:page="pagination.page"
          v-model:page-size="pagination.pageSize"
          :page-count="pagination.pageCount"
          :page-size="pagination.pageSize"
          @update:page="handlePageChange"
          @update:page-size="handlePageSizeChange"
        />
        <div>
          <n-button style="margin-right: 10px" @click="close">
            取消
          </n-button>
          <n-button
            type="error"
            :disabled="checkedRowKeys.length === 0"
            @click="deleteSelectedData"
          >
            删除所选
          </n-button>
        </div>
      </div>
    </div>
  </n-modal>
</template>

<style scoped>
.modal-content {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.footer {
  display: flex;
  gap: 10px;
  margin-top: 10px;
  padding: 10px;
  background-color: var(--n-modal-footer-bg);
  border-top: 1px solid var(--n-modal-border-color);
  justify-content: flex-end;
}

/* 确保分页组件在新的一行 */

.n-pagination {
  margin-top: 10px;
}
</style>
