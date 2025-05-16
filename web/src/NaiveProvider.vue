<script setup lang="ts">
function registerNaiveTools() {
  window.$ModalMessage = useMessage()
  window.$ModalNotification = useNotification()
  window.$ModalDialog = useDialog()
  window.$ModalLoadingBar = useLoadingBar()
}

const NaiveProviderWrapper = defineComponent({
  name: 'NaiveProviderWrapper',
  setup() {
    registerNaiveTools()
  },
  render() {
    return h('div')
  },
})
</script>

<template>
  <NLoadingBarProvider>
    <NDialogProvider>
      <NNotificationProvider>
        <NMessageProvider>
          <slot></slot>
          <NaiveProviderWrapper />
        </NMessageProvider>
      </NNotificationProvider>
    </NDialogProvider>
  </NLoadingBarProvider>
</template>
