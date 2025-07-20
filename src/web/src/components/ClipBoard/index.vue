<script lang="tsx" setup>
const props = withDefaults(
  defineProps<Props>(),
  {
    text: '',
    autoColor: true,
    noCopy: false,
  },
)

const { copy, copied, copyDuration } = useClipText()

interface Props {
  text: string
  autoColor?: boolean
  noCopy?: boolean
}
const copyText = async () => {
  if (copied.value) {
    return
  }

  await copy(props.text)
  window.$ModalMessage.destroyAll()
  window.$ModalMessage.success('已复制', {
    duration: copyDuration,
  })
}

defineExpose({
  copyText,
})
</script>

<template>
  <template
    v-if="$slots.default"
  >
    <slot
      name="default"
      v-bind="{
        copyText,
        copied,
      }"
    ></slot>
  </template>
  <div
    v-else
    class="size-20"
    :class="[
      copied
        ? `cursor-initial i-ic:baseline-check ${autoColor && 'c-primary'}`
        : 'cursor-pointer i-ci:copy',
    ]"
    v-bind="{
      onClick: !noCopy ? copyText : () => {},
    }"
  ></div>
</template>

<style lang="scss" scoped>

</style>
