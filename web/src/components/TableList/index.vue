<script lang="ts" setup>
import type { DataTableColumns } from 'naive-ui'

interface Props {
  columns: DataTableColumns<any>
  loading: boolean
  data: Array<any>
}

const props = withDefaults(defineProps<Props>(), {
  loading: false,
})
</script>

<template>
  <n-spin
    :show="loading"
    :rotate="false"
    class="bg-#fff:80 h-full"
    :style="{
      '--n-opacity-spinning': '0.5',
    }"
  >
    <template #icon>
      <div class="i-svg-spinners:tadpole"></div>
    </template>
    <n-data-table
      v-if="data.length"
      :columns="columns"
      :data="data"
      :bordered="false"
      :single-line="false"
      :scroll-x="2200"
      class="bg-#fff"
      v-bind="$attrs"
    />
    <section v-else flex="~ col items-center justify-center" select-none>
      <img
        class="w-260"
        select-none
        src="@/assets/svg/empty-status.svg"
      >
      <span class="text-16 c-#909399">
        <slot name="emptyText"></slot>
      </span>
    </section>
  </n-spin>
</template>

<style lang="scss" scoped></style>
