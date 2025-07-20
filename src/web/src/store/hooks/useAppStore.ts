import { defineStore } from 'pinia'
import { store } from '@/store'

export const useAppStore = defineStore('app-store', () => {
  const areaLoading = ref(false)

  return {
    areaLoading,
  }
})

export function useAppStoreWithOut() {
  return useAppStore(store)
}
