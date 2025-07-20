<script lang="ts">
export default defineComponent({
  name: 'NotFound',
})
</script>

<template>
  <div class="w-full h-full flex items-center justify-center">
    <n-result
      status="404"
      title="404 你什么也找不到"
      description="生活总归带点荒谬"
    >
      <template #footer>
        <n-button @click="$router.replace('/')">看看别的</n-button>
      </template>
    </n-result>
  </div>
</template>

<style>

</style>
