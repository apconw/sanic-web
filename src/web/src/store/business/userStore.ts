// src/store/userStore.ts
import { defineStore } from 'pinia'

export const useUserStore = defineStore('user', {
  state: () => ({
    user: null as null | { token: string },
  }),
  actions: {
    login(user: { token: string }) {
      this.user = user
      // 将用户信息存储到 sessionStorage
      sessionStorage.setItem('user', JSON.stringify(user))
    },
    logout() {
      this.user = null
      // 清除 sessionStorage 中的用户信息
      sessionStorage.removeItem('user')
    },
    init() {
      // 从 sessionStorage 中恢复用户信息
      const storedUser = sessionStorage.getItem('user')
      if (storedUser) {
        this.user = JSON.parse(storedUser)
      }
    },
    getUserToken() {
      const storedUser = sessionStorage.getItem('user')
      if (storedUser) {
        this.user = JSON.parse(storedUser)
      }
      return this.user?.token
    },
  },
  getters: {
    isLoggedIn: (state) => !!state.user,
  },
})
