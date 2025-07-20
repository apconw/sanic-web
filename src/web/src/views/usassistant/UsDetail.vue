<script setup lang="tsx">
const router = useRouter()

// 优化函数命名
const callBack = () => {
  router.push('/testAssitant')
}

// 将表格列定义提取为常量
const columns = [
  { title: 'ID', key: 'id' },
  { title: '描述', key: 'description' },
]

const tableData = ref([
  { id: 'US01', description: '创建新帐户' },
  { id: 'US02', description: '重命名帐户' },
  { id: 'US03', description: '更新个人信息' },
  { id: 'US04', description: '设置支付方式' },
])

// 右侧表格列定义
const rightTableColumns = [
  { title: '序号', key: 'index' },
  { title: '用例', key: 'case' },
  { title: '前置条件', key: 'precondition' },
  { title: '步骤', key: 'steps' },
  { title: '期望', key: 'expectation' },
  { title: '操作', key: 'operation' },
]

// 右侧表格数据示例
const rightTableData = ref([
  {
    index: 1,
    case: '登录测试',
    precondition: '用户已注册',
    steps: '输入用户名和密码，点击登录',
    expectation: '成功登录',
    operation: '检查登录状态',
  },
  {
    index: 2,
    case: '注册测试',
    precondition: '无',
    steps: '输入用户名、密码等信息，点击注册',
    expectation: '注册成功',
    operation: '检查用户列表',
  },
])
</script>

<template>
  <n-layout class="main-layout" style="height: 100vh">
    <!-- 头部区域 -->
    <n-layout-header class="header">
      <div class="size-26 i-hugeicons:ai-chat-02"></div>
      <div class="header-content">测试用例生成工具</div>
    </n-layout-header>
    <!-- 内容区域 -->
    <n-layout-content
      style="flex: 1; display: flex; flex-direction: column"
    >
      <n-layout class="sub-layout" style="flex: 1">
        <!-- 子头部区域 -->
        <n-layout-header class="sub-header">
          <n-button
            quaternary
            icon-placement="left"
            type="primary"
            strong
            class="w-35 h-38 mt-8 ml-10 self-center"
            @click="callBack"
          >
            <template #icon>
              <n-icon size="24">
                <svg
                  t="1740725444100"
                  class="icon"
                  viewBox="0 0 1024 1024"
                  version="1.1"
                  xmlns="http://www.w3.org/2000/svg"
                  p-id="16679"
                  width="200"
                  height="200"
                >
                  <path
                    d="M671.3 480.1H301.7L473.8 308c12.4-12.4 12.4-32.8 0-45.3-12.4-12.4-32.8-12.4-45.3 0L201.9 489.5c-5 5-8 11.2-9 17.7v0.1c-0.1 0.4-0.1 0.9-0.2 1.3v0.3c0 0.4-0.1 0.8-0.1 1.2v4c0 0.4 0.1 0.8 0.1 1.2v0.3c0 0.4 0.1 0.9 0.2 1.3v0.1c1 6.5 4 12.7 9 17.7l226.7 226.7c12.4 12.4 32.8 12.4 45.3 0 12.4-12.4 12.4-32.8 0-45.3l-172.2-172h369.6c17.6 0 32-14.4 32-32s-14.4-32-32-32z"
                    p-id="16680"
                    fill="#2c2c2c"
                  />
                </svg>
              </n-icon>
            </template>
          </n-button>
        </n-layout-header>
        <!-- 子内容区域 -->
        <n-layout-content style="flex: 1; display: flex">
          <n-split
            direction="horizontal"
            size="300px"
            min="300px"
            max="500px"
            style="flex: 1"
          >
            <!-- 左侧区域 -->
            <template #1>
              <n-space
                vertical
                class="h-full flex flex-col"
              >
                <n-space inline align="center" :size="-4">
                  <n-tag
                    checkable
                    class="font-600 text-15"
                  >
                    用户需求
                  </n-tag>
                  <n-button
                    quaternary
                    style="width: 2px; height: 28px"
                  >
                    <template #icon>
                      <n-icon size="14">
                        <svg
                          t="1742291076520"
                          class="icon"
                          viewBox="0 0 1024 1024"
                          version="1.1"
                          xmlns="http://www.w3.org/2000/svg"
                          p-id="5763"
                          width="200"
                          height="200"
                        >
                          <path
                            d="M832 640H640v192c0 35.2-28.8 64-64 64h-128c-35.2 0-64-28.8-64-64V640H192c-35.2 0-64-28.8-64-64v-128C128 412.8 156.8 384 192 384H384V192c0-35.2 28.8-64 64-64h128c35.2 0 64 28.8 64 64V384h192c35.2 0 64 28.8 64 64v128c0 35.2-28.8 64-64 64z"
                            fill="#1296db"
                            p-id="5764"
                          />
                        </svg>
                      </n-icon>
                    </template>
                  </n-button>
                </n-space>
                <!-- 数据表格 -->
                <n-data-table
                  ref="tableRef"
                  class="custom-table text-14 flex-1"
                  :style="{
                    '--n-td-color-hover': `#d5dcff`,
                    '--n-body-overflow-y': `auto`,
                    'font-family': `-apple-system,
                      BlinkMacSystemFont, 'Segoe UI',
                      Roboto, 'Helvetica Neue', Arial,
                      sans-serif`,
                  }"
                  size="small"
                  :bordered="false"
                  :bottom-bordered="false"
                  :single-line="false"
                  :columns="columns"
                  :data="tableData"
                />
              </n-space>
            </template>
            <!-- 右侧区域 -->
            <template
              #2
            >
              <n-data-table
                class="text-14 flex-1"
                :style="{
                  '--n-td-color-hover': `#d5dcff`,
                  '--n-body-overflow-y': `auto`,
                  'font-family': `-apple-system,
                    BlinkMacSystemFont, 'Segoe UI',
                    Roboto, 'Helvetica Neue', Arial,
                    sans-serif`,
                }"
                size="small"
                :bordered="false"
                :bottom-bordered="false"
                :single-line="false"
                :columns="rightTableColumns"
                :data="rightTableData"
              />
            </template>
          </n-split>
        </n-layout-content>
      </n-layout>
    </n-layout-content>
  </n-layout>
</template>

<style scoped>
.header {
  display: flex;
  height: 52px;
  justify-content: center;
  align-items: center;
  padding: 10px 20px;
  background-color: #f6f7fb;
}

.header-content {
  font-size: 18px;
  font-weight: bold;
  color: #26244c;
  text-align: center;
}

.main-layout,
.sub-layout {
  height: 100%;
  border-radius: 10px;
  margin-bottom: 10px;
  background-color: #fff;
}

.sub-header {
  margin: 5px 0;
  height: 52px;
  background: #f6f7fb;
}

:deep(.custom-table .n-data-table-thead) {
  display: none;
}

/* 优化滚动条样式 */

::-webkit-scrollbar {
  width: 8px;
}

::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 4px;
}

::-webkit-scrollbar-thumb {
  background: #888;
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: #555;
}
</style>
