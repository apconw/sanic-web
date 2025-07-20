const LayoutDefault = () => import('@/components/Layout/default.vue')

const childrenRoutes: Array<RouteRecordRaw> = [
  {
    path: 'chat',
    meta: { requiresAuth: true },
    name: 'ChatRoot',
    redirect: {
      name: 'ChatIndex',
    },
    children: [
      {
        path: '',
        name: 'ChatIndex',
        component: () => import('@/views/chat.vue'),
      },
    ],
  },
  {
    path: 'testAssitant',
    name: 'TestAssitant',
    component: () => import('@/views/DemandManager.vue'),
    meta: { requiresAuth: true }, // 标记需要认证
  },
  {
    path: 'uaDetail/:id',
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
    path: 'mcpChat',
    name: 'McpChat',
    component: () => import('@/views/mcp/MCPClient.vue'),
    meta: { requiresAuth: true }, // 标记需要认证
  },
]

export default childrenRoutes
