import { createWebHashHistory } from 'vue-router'
import { isMockDevelopment } from '@/config'
import { createRouterGuards } from '@/router/permission'
import routes from './routes'

const history = isMockDevelopment ? createWebHashHistory() : createWebHistory()

const router = createRouter({
  history,
  routes,
})

// 全局前置守卫
router.beforeEach((to, from, next) => {
  const userStore = useUserStore()
  if (to.meta.requiresAuth && !userStore.isLoggedIn) {
    // 如果目标路由需要认证且用户未登录，则重定向到登录页面
    next('/login')
  } else {
    next()
  }
})

export async function setupRouter(app: App) {
  createRouterGuards(router)
  app.use(router)

  await router.isReady()
}

export default router
