<script lang="ts" setup>
const router = useRouter()


const testRoutesNavs = computed(() => {
  const routes = router.getRoutes()
  const _routes = routes.filter((routeItem) => {
    return (routeItem.name !== 'UserLogin')
      && !routeItem.redirect
      && routeItem?.meta?.title
  })
  return _routes.sort((prev, next) => {
    return prev.meta.title! > next.meta.title!
      ? 1
      : -1
  })
})

/**
 * 随机生成测试 params hash
 */
const extractParamsFromRoute = (routePath) => {
  const paramRegex = /:([^/?]+)/g
  const params = {}

  routePath.replace(paramRegex, (__, paramName) => {
    params[paramName] = uuidv4()
    return ''
  })

  return params
}

const handleNavigateTo = (routeItem: RouteRecordRaw) => {
  router.push({
    name: routeItem.name,
    params: extractParamsFromRoute(routeItem.path),
  })
}
</script>

<template>
  <aside
    class="navigation-nav-sidebar-container"
  >
    <div>该侧边栏路由仅用作开发初期调试</div>
    <n-button
      v-for="(routeItem) in testRoutesNavs"
      :key="routeItem.path"
      type="primary"
      :dashed="routeItem.name !== $route.name"
      @click="handleNavigateTo(routeItem)"
    >
      {{ routeItem.meta.title }}
    </n-button>
  </aside>
</template>

<style lang="scss" scoped>
.navigation-nav-sidebar-container {
  --at-apply: flex flex-col;
  --at-apply: w-full h-full overflow-y-auto b-1 b-solid;
  --at-apply: b-1 b-solid;
  --at-apply: p-20;

  & > * {
    --at-apply: flex justify-start;
    --at-apply: mb-20px;

    &:last-child {
      --at-apply: mb-0;
    }
  }
}
</style>
