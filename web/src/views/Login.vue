<script lang="tsx" setup>
import { useMessage } from 'naive-ui'
import { onMounted, onUnmounted, ref } from 'vue'
import * as GlobalAPI from '@/api'

/* ---------- 登录业务 ---------- */
const form = ref({ username: 'admin', password: '123456' })
const formRef = ref()
const message = useMessage()
const router = useRouter()
const userStore = useUserStore()

/* ---------- 反重力粒子背景 ---------- */
class Particle {
  constructor(public x: number,
    public y: number,
    public r: number,
    public vx: number,
    public vy: number,
    public hue: number) {}

  update(w: number, h: number) {
    this.x += this.vx
    this.y += this.vy
    // 轻微左右摆动
    this.vx += (Math.random() - 0.5) * 0.05
    this.vx *= 0.99
    // 飞出顶部后从底部重生
    if (this.y + this.r < 0) {
      this.y = h + this.r
      this.x = Math.random() * w
    }
  }

  draw(ctx: CanvasRenderingContext2D) {
    ctx.save()
    ctx.beginPath()
    ctx.arc(this.x, this.y, this.r, 0, Math.PI * 2)
    ctx.fillStyle = `hsla(${200 + (this.hue % 40)}, 80%, 70%, 0.4)`
    ctx.fill()
    ctx.restore()
  }
}

let canvas: HTMLCanvasElement | null = null
let ctx: CanvasRenderingContext2D | null = null
let particles: Particle[] = []
let raf = 0

const initParticles = () => {
  if (!canvas) {
    return
  }
  const { innerWidth: w, innerHeight: h } = window
  canvas.width = w
  canvas.height = h
  particles = []
  for (let i = 0; i < 120; i++) {
    particles.push(new Particle(
      Math.random() * w,
      Math.random() * h,
      Math.random() * 3 + 1.5, // 粒子半径
      (Math.random() - 0.5) * 0.2, // 水平速度
      -(Math.random() * 0.8 + 0.5), // 垂直速度
      Math.random() * 360, // 色相
    ))
  }
}

const animate = () => {
  if (!ctx || !canvas) {
    return
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  particles.forEach((p) => {
    p.update(canvas!.width, canvas!.height)
    p.draw(ctx)
  })
  raf = requestAnimationFrame(animate)
}

const startBg = () => {
  canvas = document.getElementById('bg') as HTMLCanvasElement
  if (canvas) {
    ctx = canvas.getContext('2d')!
    initParticles()
    animate()
  }
}
const stopBg = () => cancelAnimationFrame(raf)

/* ---------- 生命周期 ---------- */
onMounted(() => {
  if (userStore.isLoggedIn) {
    router.push('/')
  } else {
    startBg()
  }
  window.addEventListener('resize', initParticles)
})
onUnmounted(() => {
  stopBg()
  window.removeEventListener('resize', initParticles)
})

/* ---------- 登录 ---------- */
const handleLogin = () => {
  if (!form.value.username || !form.value.password) {
    message.error('请填写完整信息')
    return
  }
  GlobalAPI.login(form.value.username, form.value.password).then(async (res) => {
    if (!res.body) {
      return
    }
    const data = await res.json()
    if (data.code === 200) {
      userStore.login({ token: data.data.token })
      setTimeout(() => router.push('/'), 500)
    } else {
      message.error('登录失败，请检查用户名或密码')
    }
  })
}
</script>

<template>
  <div class="login-container">
    <canvas id="bg"></canvas>
    <transition name="fade" mode="out-in">
      <n-card
        v-if="!userStore.isLoggedIn"
        class="w-400 max-w-90vw login-card"
        title="登录"
      >
        <n-form ref="formRef" @submit.prevent="handleLogin">
          <n-form-item label="用户名" path="username">
            <n-input v-model:value="form.username" placeholder="请输入用户名" />
          </n-form-item>
          <n-form-item label="密码" path="password">
            <n-input v-model:value="form.password" type="password" placeholder="请输入密码" />
          </n-form-item>
          <n-form-item>
            <n-button
              type="primary"
              block
              class="login-button"
              @click="handleLogin"
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
.login-container {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
  position: relative;
  overflow: hidden;
  background: linear-gradient(135deg, #4b6cb7 0%, #182848 100%);
}

#bg {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  z-index: 0;
}

.login-card {
  position: relative;
  z-index: 1;
  width: 400px;
  max-width: 90vw;
  backdrop-filter: blur(10px);
  background: rgb(255 255 255 / 20%);
  border-radius: 16px;
  border: 1px solid rgb(255 255 255 / 25%);
  box-shadow: 0 8px 32px rgb(0 0 0 / 20%);
}

.login-button {
  --n-color-primary: #6a8cff;
  --n-color-hover: #809dff;
  --n-color-pressed: #507aff;
  --n-color-focus: #6a8cff;
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.5s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
