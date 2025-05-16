<template>
  <LayoutSlotCenterPanel v-bind="$attrs">
    <template #left> </template>
    <div
      h-full
      style="background: linear-gradient(to bottom, #8874f1, #588af9)"
    >
      <slot name="default"></slot>
    </div>
  </LayoutSlotCenterPanel>
</template>

<style lang="scss" scoped>
/** 确保侧边栏内容区域可滚动 */

.scrollable-sidebar {
  overflow-y: auto; /* 添加纵向滚动条 */
  max-height: calc(
    100vh - 120px
  ); /* 设置最大高度，确保输入框和导航栏有足够的空间 */

  padding-bottom: 20px; /* 底部内边距，防止内容被遮挡 */
  background-color: #fff;
}

/* 滚动条整体部分 */

.scrollable-sidebar::-webkit-scrollbar {
  width: 4px; /* 竖向滚动条宽度 */
  height: 4px; /* 横向滚动条高度 */
}

/* 滚动条的轨道 */

.scrollable-sidebar::-webkit-scrollbar-track {
  background: #fff; /* 轨道背景色 */
}

/* 滚动条的滑块 */

.scrollable-sidebar::-webkit-scrollbar-thumb {
  background: #cac9f9; /* 滑块颜色 */
  border-radius: 10px; /* 滑块圆角 */
}

/* 滚动条的滑块在悬停状态下的样式 */

.scrollable-sidebar::-webkit-scrollbar-thumb:hover {
  background: #cac9f9; /* 悬停时滑块颜色 */
}
</style>
