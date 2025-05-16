<script lang="ts" setup>
// 引入所需的组件
import { NLayout, NLayoutContent, NLayoutFooter, NLayoutHeader } from 'naive-ui'
</script>

<template>
  <n-layout class="custom-layout">
    <n-layout-header class="header">上</n-layout-header>
    <n-layout-content class="content">中</n-layout-content>
    <n-layout-footer class="footer">下</n-layout-footer>
  </n-layout>
</template>

<style scoped>
.custom-layout {
  width: 240px;
  height: 100%;
  border-top-left-radius: 20px;
  border-bottom-left-radius: 20px;
  background-color: gray;
  margin-top: 10px;
}

.header,
.footer {
  background-color: darkgray;
  padding: 8px;
}

.content {
  background-color: lightgray;
  padding: 8px;
}
</style>
