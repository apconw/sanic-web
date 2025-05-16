<script lang="ts" setup>
import { NButton } from 'naive-ui'

// 定义 props 接口以获取类型检查
interface Props {
  labels?: string[]
}

// 使用 withDefaults 提供默认值
const props = withDefaults(defineProps<Props>(), {
  labels: () => [],
})

// 自定义事件用于 子父组件传递事件信息
const emit = defineEmits(['suggested'])

// 定义默认按钮文案
const defaultLabels = []

// 计算属性用于决定实际使用的按钮文案
const buttonLabels = computed(() =>
  props.labels.length > 0 ? props.labels : defaultLabels,
)

// 点击事件处理函数
const handleClick = (index: number) => {
  emit('suggested', index)
}
</script>

<template>
  <div class="button-container">
    <!-- 使用 v-for 指令循环渲染 n-button -->
    <n-button
      v-for="(text, index) in buttonLabels"
      :key="index"
      class="block-button"
      @click="handleClick(index)"
    >
      {{ text }}
    </n-button>
  </div>
</template>

<style scoped>
/* 添加一些样式以区分按钮 */

.button-container {
  /* 如果需要更复杂的布局，可以在这里添加更多样式 */
}

.block-button {
  display: block; /* 使每个按钮独占一行 */
  margin: 10px 4px; /* 调整上下间距 */
  background-color: #fff;
  border-radius: 10px;
  height: 38px;
  color: #666;
  font-size: 13px;
  font-family: PMingLiU;
}
</style>
