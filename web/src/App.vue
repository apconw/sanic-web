<script lang="ts" setup>
import { dateZhCN, zhCN } from 'naive-ui'
import NaiveProvider from './NaiveProvider.vue'

defineOptions({
  name: 'App',
})

const { defaultTheme, themeOverrides } = useTheme()

useCopyCode()
</script>

<template>
  <NConfigProvider
    class="h-full"
    :locale="zhCN"
    :date-locale="dateZhCN"
    :theme="defaultTheme"
    :theme-overrides="themeOverrides"
  >
    <NaiveProvider>
      <RouterView />
    </NaiveProvider>
  </NConfigProvider>
</template>

<style lang="scss">
@use '@/styles/index.scss';
</style>
