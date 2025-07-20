<script lang="ts" setup>
interface Props {
  loading?: boolean
}
withDefaults(defineProps<Props>(), {
  loading: false,
})

const router = useRouter()
const userStore = useUserStore()
const appStore = useAppStore()

appStore.areaLoading = true
// 主页面加载提示
setTimeout(() => {
  appStore.areaLoading = false
}, 400)
</script>

<template>
  <LayoutSlotFrame :class="['bg-no-repeat bg-cover bg-center']">
    <template #center>
      <div
        flex="~"
        size-full
        overflow-hidden
        class="panel-shadow"
      >
        <n-spin
          w-full
          h-full
          content-class="w-full h-full flex"
          :show="appStore.areaLoading"
          :rotate="false"
          class="bg-#ffffff"
          :style="{
            '--n-opacity-spinning': '0',
          }"
        >
          <section
            flex="~ col"
            min-w-0
            w-70
            h-full
            overflow-hidden
            relative
          >
            <NavigationSideBar />
          </section>
          <section
            flex="1 ~"
            min-w-0
            h-full
            overflow-hidden
            py-12
            pr-12
            style="background: linear-gradient(to bottom, #8874f1, #588af9)"
          >
            <div
              size-full
              rounded-10
              overflow-hidden
            >
              <LayoutDefault />
            </div>
          </section>
        </n-spin>
      </div>
    </template>
    <template #bottom>
      <NavigationNavFooter />
    </template>
  </LayoutSlotFrame>
</template>

<style lang="scss" scoped>
.panel-shadow {
  --shadow: 50px 50px 100px 10px rgb(0 0 0 / 10%);
  --at-apply: 'shadow-[--shadow]';
}
</style>
