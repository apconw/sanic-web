import type { AxiosRequestConfig } from 'axios'

import type router from '@/router'
import type { getFilterResponse } from '@/store/utils/mixin'

declare module 'vue' {
  /**
   *
   */
  interface ComponentCustomProperties extends Window {
    // ...

  }
}

declare module 'axios' {
  /**
   * Costom Axios Field.
   */
  export interface AxiosRequestConfig {
    redirect?: string
    /**
     * 是否触发浏览器下载弹框，默认会触发（仅限 blob type）
     */
    autoDownLoadFile?: boolean
  }
}

declare module 'pinia' {
  export interface PiniaCustomProperties {
    filterResponse: typeof getFilterResponse
    router: typeof router
  }
}

declare module 'vue-router' {
  export interface RouteMeta {
    title?: string
  }
}

declare global {

  /**
   * General Object Types.
   */
  type ObjectValueSuite<T = any> = { [key in any]: T }

  /**
   * `error`: Response Status Code.
   *
   * `data`: Response Body.
   *
   * `msg`: Response Message.
   */
  export interface IRequestData {
    error: number
    data: any
    msg: string
    aborted?: boolean
  }

  interface IRequestSuite {
    get(uri: string, params?: ObjectValueSuite, config?: AxiosRequestConfig): Promise<IRequestData>
    post(uri: string, data?: any, config?: AxiosRequestConfig): Promise<IRequestData>
    put(uri: string, data?: any, config?: AxiosRequestConfig): Promise<IRequestData>
    patch(uri: string, data?: any, config?: AxiosRequestConfig): Promise<IRequestData>
    delete(uri: string, config?: AxiosRequestConfig): Promise<IRequestData>
  }

  type IModulesApiSuite = ObjectValueSuite<(...args: any) => Promise<IRequestData>>

  /**
   * Store FilterResponse Callback Type.
   */
  type IStoreFilterCallBack = (res: IRequestData) => Promise<IRequestData> | void

}
export { }
