<script lang="tsx" setup>
import { systemTitle } from '@/base'

interface Props {
  transparent?: boolean
  hasBorder?: boolean
}
withDefaults(defineProps<Props>(), {
  transparent: true,
  hasBorder: true,
})

const handleToRepo = () => {
  const link = ref('https://github.com/apconw/sanic-web')
  window.open(link.value, '_blank')
}
</script>

<template>
  <header
    class="navigation-nav-header-container b-b"
    :class="[
      transparent ? 'bg-bgcolor' : 'bg-transparent',
      hasBorder ? 'b-b-#000/8 b-b-solid' : 'b-b-transparent',
    ]"
  >
    <div class="header-left"></div>
    <div class="flex-1">
      <div flex="~ col items-center justify-center" px-36px>
        <div
          flex="~ items-center justify-center"
          :style="{
            'font-family': `-apple-system, BlinkMacSystemFont,
              'Segoe UI', Roboto, 'Helvetica Neue', Arial,
              sans-serif`,
            'font-size': `17px`,
            'font-weight': `bold`,
            'color': `#6a30e6`,
          }"
          select-none
          cursor-pointer
          @click="handleToRepo()"
        >
          <div class="size-26 i-hugeicons:ai-chat-02"></div>
          <div class="flex-1 text-center">
            {{ systemTitle }}
          </div>
        </div>
        <slot name="bottom"></slot>
      </div>
    </div>

    <div class="header-right"></div>
  </header>
</template>

<style lang="scss" scoped>
.navigation-nav-header-container {
  --at-apply: w-full flex items-center justify-center py-10;

  height: 60px;
  border-bottom: 0;

  .header-left,
  .header-right {
    --at-apply: flex items-center h-full text-16;
  }

  .header-left {
    --at-apply: h-50px;
  }

  .header-right {
    --at-apply: flex items-center h-full text-16;
  }

  // 添加具体的背景颜色

  &.bg-bgcolor {
    // background-color: #f0effe;
    // background: linear-gradient(to right, #f0effe, #ddeefe);

    background-color: #f6f7fb;

    // 替换为你的颜色
  }

  &.bg-transparent {
    background-color: transparent;
  }
}
</style>
