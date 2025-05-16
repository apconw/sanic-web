import childRoutes from '@/router/child-routes'

const routes: Array<RouteRecordRaw> = [
  {
    path: '/',
    name: 'Root',
    redirect: {
      name: 'ChatRoot',
    },
    meta: { requiresAuth: true }, // 标记需要认证
  },
  ...childRoutes,
  {
    path: '/:pathMatch(.*)',
    name: '404',
    component: () => import('@/components/404.vue'),
  },
  {
    path: '/login',
    name: 'Login',
    component: () => import('@/views/Login.vue'),
  },
  {
    path: '/testAssitant',
    name: 'TestAssitant',
    component: () => import('@/views/DemandManager.vue'),
    meta: { requiresAuth: true }, // 标记需要认证
  },
  {
    path: '/uaDetail/:id',
    name: 'UaDetail',
    component: () => import('@/views/usassistant/UsDetail.vue'),
    meta: { requiresAuth: true },
  },
  // {
  //     path: '/testAssitant',
  //     name: 'TestAssitant',
  //     component: () => import('@/views/TestAssistant.vue'),
  //     meta: { requiresAuth: true } // 标记需要认证
  // },
  {
    path: '/mcpChat',
    name: 'McpChat',
    component: () => import('@/views/mcp/MCPClient.vue'),
    meta: { requiresAuth: true }, // 标记需要认证
  },
]

export default routes
