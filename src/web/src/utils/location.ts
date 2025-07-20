const locationHost = {
  hostname: 'localhost',
  baseApiIp: 'http://10.30.10.54:10001',
  baseApi: 'http://10.30.10.54:10001/api',
}

const hostList = [
  locationHost,
]

/**
 *  获取当前服务的 host 前缀
 */
export const currentHost = hostList.find((hostItem) => {
  return window.location.hostname === hostItem.hostname
}) || locationHost
