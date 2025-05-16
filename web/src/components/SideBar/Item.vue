<script lang="ts" setup>
interface Props {
  active?: boolean
}
withDefaults(defineProps<Props>(), {
  active: false,
})

const emit = defineEmits(['click', 'edit', 'remove'])
</script>

<template>
  <div
    flex="~ justify-between"
    py="10px"
    px="14px"
    rounded-8px
    cursor-pointer
    class="group font-bold transition-colors-260 b b-solid"
    :class="[
      active
        ? 'c-primary b-primary'
        : 'hover:b-info/50 c-#303133 b-transparent',
    ]"
    @click="emit('click')"
  >
    <div flex="1" text-nowrap text-ellipsis overflow-x-hidden pr-6px>
      <slot></slot>
    </div>
    <div
      class="opacity-0 transition-colors-200 text-16"
      px="4"
      :class="[
        active
          ? 'opacity-100 c-primary/60 hover:c-primary'
          : 'group-hover:opacity-100 c-#303133/60 hover:c-primary',
      ]"
      flex="~ justify-center items-center"
      @click.stop="emit('edit')"
    >
      <div class="i-mingcute:pencil-2-line"></div>
    </div>
    <n-popconfirm @positive-click="emit('remove')">
      <template #trigger>
        <div
          class="opacity-0 transition-colors-200 text-16"
          px="4"
          :class="[
            active
              ? 'opacity-100 c-primary/60 hover:c-primary'
              : 'group-hover:opacity-100 c-#303133/60 hover:c-primary',
          ]"
          flex="~ justify-center items-center"
          @click.stop
        >
          <div class="i-mingcute:delete-2-line"></div>
        </div>
      </template>
      确认删除？
    </n-popconfirm>
  </div>
</template>

<style lang="scss" scoped></style>
