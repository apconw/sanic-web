// 千分符函数 【判断是否四舍五入】
export const comma = (num: any, suffix = '') => {
  if (!num) {
    return
  }

  const strNum = _.isString(num) ? num : String(num)
  const intNum = _.isString(num) ? Number(num) : num

  if (isNaN(intNum)) {
    return num
  }

  let source = [] as Array<any>
  if (strNum.includes('.')) {
    source = String(intNum.toFixed(2)).split('.') // 保留两位(四舍五入); 按小数点分成2部分
    source[0] = source[0].replace(/(\d)(?=(\d{3})+$)/g, '$1,')// 只将整数部分进行都好分割
    return source.join('.') + suffix // 再将小数部分合并进来
  }

  return strNum.replace(/(\d)(?=(\d{3})+$)/g, '$1,') + suffix
}

export const generateYears = (startYear) => {
  const currentYear = new Date().getFullYear()
  const endYear = currentYear + 1 // 明年
  const years: string[] = []

  let year = startYear
  for (; year <= endYear; year++) {
    years.push(year)
  }

  return years
}
