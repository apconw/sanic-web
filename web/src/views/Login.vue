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

/* ---------- 科技感粒子背景 ---------- */
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
    // 简单的漂浮效果
    this.vx += (Math.random() - 0.5) * 0.02
    this.vx *= 0.99
    
    // 边界检查，循环
    if (this.y + this.r < 0) {
      this.y = h + this.r
      this.x = Math.random() * w
    } else if (this.y - this.r > h) {
      this.y = -this.r
      this.x = Math.random() * w
    }
    
    if (this.x + this.r < 0) {
      this.x = w + this.r
      this.y = Math.random() * h
    } else if (this.x - this.r > w) {
      this.x = -this.r
      this.y = Math.random() * h
    }
  }

  draw(ctx: CanvasRenderingContext2D) {
    ctx.save()
    ctx.beginPath()
    ctx.arc(this.x, this.y, this.r, 0, Math.PI * 2)
    // 科技蓝/青色调
    const alpha = Math.random() * 0.3 + 0.3
    ctx.fillStyle = `hsla(${this.hue}, 80%, 60%, ${alpha})`
    // 添加发光效果
    ctx.shadowBlur = 10
    ctx.shadowColor = ctx.fillStyle
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
  // 增加粒子数量，减小体积，营造数据流感
  for (let i = 0; i < 100; i++) {
    particles.push(new Particle(
      Math.random() * w,
      Math.random() * h,
      Math.random() * 2 + 1, // 较小的粒子
      (Math.random() - 0.5) * 0.5, 
      (Math.random() - 0.5) * 0.5,
      180 + Math.random() * 60, // 青色到蓝色 (180-240)
    ))
  }
}

const animate = () => {
  if (!ctx || !canvas) {
    return
  }
  // 稍微保留上一帧，制造拖尾效果 (可选，这里为了简洁保持清晰)
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  
  // 绘制连接线 (可选，增加科技感)
  // 这里为了保持"简单"，只画粒子，但让它们稍微亮一点
  
  particles.forEach((p) => {
    p.update(canvas!.width, canvas!.height)
    p.draw(ctx!)
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
      <div v-if="!userStore.isLoggedIn" class="login-box">
        <div class="header">
          <h1 class="title">大模型数据助手</h1>
          <p class="subtitle">智能 · 高效 · 安全</p>
        </div>
        
        <n-card :bordered="false" class="login-card">
          <n-form ref="formRef" @submit.prevent="handleLogin" size="large">
            <n-form-item path="username" :show-label="false">
              <n-input 
                v-model:value="form.username" 
                placeholder="请输入用户名" 
                class="custom-input"
              >
                <template #prefix>
                  <div class="i-carbon-user text-cyan-400" />
                </template>
              </n-input>
            </n-form-item>
            <n-form-item path="password" :show-label="false">
              <n-input 
                v-model:value="form.password" 
                type="password" 
                placeholder="请输入密码" 
                class="custom-input"
                show-password-on="click"
              >
                <template #prefix>
                  <div class="i-carbon-password text-cyan-400" />
                </template>
              </n-input>
            </n-form-item>
            <n-form-item>
              <n-button
                type="primary"
                block
                class="login-button"
                @click="handleLogin"
              >
                立即登录
              </n-button>
            </n-form-item>
          </n-form>
        </n-card>
      </div>
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

  /* 深色科技背景 */

  background: radial-gradient(circle at center, #1a2a3a 0%, #000 100%);
}

#bg {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  z-index: 0;
}

.login-box {
  position: relative;
  z-index: 1;
  width: 400px;
  max-width: 90vw;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.header {
  text-align: center;
  margin-bottom: 10px;
}

.title {
  font-size: 32px;
  font-weight: 700;
  color: #fff;
  margin: 0;
  letter-spacing: 2px;
  background: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
  background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  text-shadow: 0 0 20px rgb(0 242 254 / 30%);
}

.subtitle {
  font-size: 14px;
  color: rgb(255 255 255 / 60%);
  margin: 5px 0 0;
  letter-spacing: 4px;
  text-transform: uppercase;
}

.login-card {
  backdrop-filter: blur(20px);
  background: rgb(16 24 39 / 60%); /* 深色半透明 */
  border-radius: 16px;
  border: 1px solid rgb(0 242 254 / 20%); /* 青色微光边框 */
  box-shadow: 0 0 40px rgb(0 0 0 / 50%), inset 0 0 20px rgb(0 242 254 / 5%);
  padding: 20px 10px 0;
}

/* 自定义输入框样式 */

:deep(.n-input) {
  background-color: rgb(0 0 0 / 30%) !important;
  border: 1px solid rgb(255 255 255 / 10%);
  border-radius: 8px;
  transition: all 0.3s;
}

:deep(.n-input:hover), :deep(.n-input:focus-within) {
  border-color: rgb(0 242 254 / 50%);
  box-shadow: 0 0 10px rgb(0 242 254 / 10%);
}

:deep(.n-input__input-el) {
  color: #fff !important;
}

:deep(.n-input__placeholder) {
  color: rgb(255 255 255 / 30%) !important;
}

.login-button {
  height: 44px;
  font-size: 16px;
  letter-spacing: 2px;
  border-radius: 8px;
  background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%);
  border: none;
  font-weight: bold;
  transition: all 0.3s ease;
  box-shadow: 0 4px 15px rgb(0 114 255 / 30%);
}

.login-button:hover {
  transform: translateY(-1px);
  box-shadow: 0 6px 20px rgb(0 114 255 / 40%);
  filter: brightness(1.1);
}

.login-button:active {
  transform: translateY(1px);
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
