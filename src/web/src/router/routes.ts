import childRoutes from '@/router/child-routes'

const routes: Array<RouteRecordRaw> = [
  {
    path: '/',
    name: 'Root',
    redirect: {
      name: 'ChatRoot',
    },
    component: () => import('@/components/Layout/SlotCenterPanel.vue'),
    meta: { requiresAuth: true }, // 标记需要认证
    children: childRoutes,
  },
  {
    path: '/login',
    name: 'Login',
    component: () => import('@/views/Login.vue'),
  },
  {
    path: '/:pathMatch(.*)',
    name: '404',
    component: () => import('@/components/404.vue'),
  },
]

export default routes
