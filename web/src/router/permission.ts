import type { Router } from 'vue-router'
import NProgress from 'nprogress'

NProgress.configure({
  showSpinner: false,
})

export function createRouterGuards(router: Router) {
  router.beforeEach(async (to, from, next) => {
    NProgress.start()
    next()
  })

  router.afterEach(() => {
    NProgress.done()
  })
}
