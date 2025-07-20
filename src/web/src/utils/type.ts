const originToString = Object.prototype.toString

export function isFunction(obj: any) {
  return typeof (obj) === 'function'
}

export function isObject(obj: any) {
  return obj === Object(obj)
}

export function isArray(obj: any) {
  return originToString.call(obj) === '[object Array]'
}

export function isDate(obj: any) {
  return originToString.call(obj) === '[object Date]'
}

export function isRegExp(obj: any) {
  return originToString.call(obj) === '[object RegExp]'
}

export function isBoolean(obj: any) {
  return originToString.call(obj) === '[object Boolean]'
}

export function isString(obj: any): obj is string {
  return originToString.call(obj) === '[object String]'
}

export function isUndefined(obj: any) {
  return originToString.call(obj) === '[object Undefined]'
}

export function isNull(obj: any) {
  return originToString.call(obj) === '[object Null]'
}

export function isBigInt(obj: any) {
  return originToString.call(obj) === '[object BigInt]'
}

export function isNumberical(obj: any) {
  return !isNaN(Number.parseFloat(obj)) && isFinite(obj)
}
