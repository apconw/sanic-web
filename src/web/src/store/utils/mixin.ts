
export function getFilterResponse(
  res: globalThis.IRequestData,
  successCallback?: globalThis.IStoreFilterCallBack | null,
  errorCallback?: globalThis.IStoreFilterCallBack | null,
): Promise<globalThis.IRequestData> {
  return new Promise((resolve) => {
    if (res && res.error === 0) {
      successCallback && successCallback(res)
    } else {
      errorCallback
        ? errorCallback(res)
        : window.$ModalMessage.error(res.msg!, {
            closable: true,
          })
    }
    resolve(res)
  })
}
