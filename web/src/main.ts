import App from '@/App.vue'

import InstallGlobalComponents from '@/components'
import { setupRouter } from '@/router'

import { setupStore } from '@/store'

import 'virtual:uno.css'

const app = createApp(App)

function setupPlugins() {
  app.use(InstallGlobalComponents)
}

async function setupApp() {
  setupStore(app)
  await setupRouter(app)
  app.mount('#app')
}

setupPlugins()
setupApp()

// 初始化用户状态
const userStore = useUserStore()
userStore.init()

export default app
