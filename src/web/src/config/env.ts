const isDev = import.meta.env.DEV

/**
 * TODO: 若是本地开发环境、Github 部署环境，则模拟大模型相关策略，不调接口
 */
// export const isMockDevelopment = isDev
//   || process.env.VITE_ROUTER_MODE === 'hash'

// 打开此行，则会调用真实的大模型接口（需提前配置好 Key）
export const isMockDevelopment = false
