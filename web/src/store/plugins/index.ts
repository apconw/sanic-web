/**
 * Plugins for Pinia
 */

import router from '@/router'
import { getFilterResponse } from '@/store/utils/mixin'

export const pluginPinia = ({ store }) => {
  store.filterResponse = getFilterResponse
  store.router = router
}
