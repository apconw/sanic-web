<script lang="ts" setup>
import type { IconifyIcon, IconifyRenderMode } from '@iconify/vue'
import { Icon } from '@iconify/vue'

interface IconProps {
  icon: IconifyIcon | string
  mode?: IconifyRenderMode
  color?: string
  flip?: string
  width?: string
  height?: string
}

const props = defineProps<IconProps>()
</script>

<template>
  <Icon
    :style="{
      'font-size': '1.2em',
    }"
    v-bind="{
      ...props,
      ...$attrs,
    }"
  />
</template>
