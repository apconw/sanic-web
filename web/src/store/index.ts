import { createPinia } from 'pinia'
import { pluginPinia } from '@/store/plugins'

const store = createPinia()

export function setupStore(app: App<Element>) {
  app.use(store)
}

store.use(pluginPinia)
export { store }
