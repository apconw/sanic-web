<script lang="ts" setup>
defineOptions({
  name: 'LayoutDefault',
})
</script>

<template>
  <router-view v-slot="{ Component }">
    <Component :is="Component" />
  </router-view>
</template>
