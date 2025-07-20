<script lang="tsx" setup>
import { useMessage } from 'naive-ui'
import * as GlobalAPI from '@/api'

// 定义表单数据
const form = ref({
  username: 'admin',
  password: '123456',
})
const formRef = ref(null)
const message = useMessage()
const router = useRouter()
const userStore = useUserStore()

onMounted(() => {
  // 获取用户信息，如果已经登录，则直接跳转首页
  if (userStore.isLoggedIn) {
    router.push('/')
  }
})

// 登录处理函数
const handleLogin = () => {
  if (form.value.username && form.value.password) {
    GlobalAPI.login(form.value.username, form.value.password).then(
      async (res) => {
        if (res.body) {
          const responseData = await res.json() // 解析为JSON对象
          if (responseData.code === 200) {
            const user = {
              token: responseData.data.token,
            }
            // 存储用户信息到 store
            userStore.login(user)

            setTimeout(() => {
              router.push('/')
            }, 500) // 2000毫秒等于2秒
          } else {
            message.error('登录失败，请检查用户名或密码')
          }
        }
      },
    )
  } else {
    message.error('请填写完整信息')
  }
}
</script>

<template>
  <div class="login-container">
    <transition name="fade" mode="out-in">
      <n-card
        v-if="!userStore.isLoggedIn"
        class="w-400 max-w-90vw"
        :style="{
          background: `linear-gradient(to bottom, #e2e1fb, white)`,
        }"
        title="登录"
      >
        <n-form ref="formRef" @submit.prevent="handleLogin">
          <n-form-item label="用户名" path="username">
            <n-input
              v-model:value="form.username"
              placeholder="请输入用户名"
              default-value="admin"
            />
          </n-form-item>
          <n-form-item label="密码" path="password">
            <n-input
              v-model:value="form.password"
              type="password"
              placeholder="请输入密码"
            />
          </n-form-item>
          <n-form-item>
            <n-button
              type="primary"
              :onclick="handleLogin"
              :block="true"
            >
              登录
            </n-button>
          </n-form-item>
        </n-form>
      </n-card>
    </transition>
  </div>
</template>

<style scoped>
/* 保持原有的样式 */

.login-container {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh; /* 使容器高度为视口高度 */
  background: linear-gradient(to bottom, #5e58e7, white);
}

body,
html {
  margin: 0;
  padding: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
    'Helvetica Neue', Arial, sans-serif;
}

/* 添加新的动画样式 */

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.5s ease;
}

.fade-enter, .fade-leave-to /* .fade-leave-active in <2.1.8 */ {
  opacity: 0;
}
</style>
