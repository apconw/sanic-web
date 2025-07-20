import { defineStore } from 'pinia'
import * as GlobalAPI from '@/api'
import * as TransformUtils from '@/components/MarkdownPreview/transform'

export interface BusinessState {
  writerList: any
  qa_type: any
  file_url: any,
  task_id: any
}

export const useBusinessStore = defineStore('business-store', {
  state: (): BusinessState => {
    return {
      writerList: {},
      // 全局报错问答类型
      qa_type: 'COMMON_QA',
      // 全局保存文件问答地址
      file_url: '',
      // 全局保存dify 任务id
      task_id: '',
    }
  },
  actions: {
    /**
     * 更新 问答类型
     */
    update_qa_type(qa_type) {
      this.qa_type = qa_type
    },
    /**
     * 更新文件url
     */
    update_file_url(file_url) {
      this.file_url = file_url
    },
    update_writerList(writerList) {
      this.writerList = writerList
    },
    clearWriterList() {
      this.writerList = []
    },
    update_task_id(task_id) {
      this.task_id = task_id
    },
    clear_task_id() {
      this.task_id = ''
    },
    /**
     * Event Stream 调用大模型python服务接口
     */
    async createAssistantWriterStylized(
      uuid,
      chat_id,
      writerOid,
      data,
    ): Promise<{
        error: number
        reader: ReadableStreamDefaultReader<string> | null
        needLogin: boolean
      }> {
      return new Promise((resolve) => {
        const query_str = data.text
        const processResponse = (res) => {
          if (res.status === 401) {
            // 登录失效
            return {
              error: 1,
              reader: null,
              needLogin: true,
            }
          } else if (res.status === 200) {
            const reader = res.body
              .pipeThrough(new TextDecoderStream())
              .pipeThrough(TransformUtils.splitStream('\n'))
              .pipeThrough(
                new TransformStream({
                  transform: (chunk, controller) => {
                    try {
                      const jsonChunk = JSON.parse(
                        chunk.split('data:')[1],
                      )
                      if (jsonChunk.task_id) {
                        // 调用已有的更新方法来更新 task_id
                        this.update_task_id(
                          jsonChunk.task_id,
                        )
                      }
                      switch (jsonChunk.dataType) {
                        case 't11':
                          controller.enqueue(
                            JSON.stringify({
                              content: `问题: ${query_str}`,
                            }),
                          )
                          break
                        case 't02':
                          if (
                            jsonChunk.data
                            && jsonChunk.data.content
                          ) {
                            controller.enqueue(
                              JSON.stringify(
                                jsonChunk.data,
                              ),
                            )
                          }
                          break
                        case 't04':
                          this.writerList = jsonChunk
                          break
                        default:
                                                // 可以在这里处理其他类型的 dataType
                      }
                    } catch (e) {
                      console.error(
                        'Error processing chunk:',
                        e,
                      )
                    }
                  },
                  flush: (controller) => {
                    controller.terminate()
                  },
                }),
              )
              .getReader()

            return {
              error: 0,
              reader,
              needLogin: false,
            }
          } else {
            return {
              error: 1,
              reader: null,
              needLogin: false,
            }
          }
        }

        // 调用后端接口拿大模型结果
        GlobalAPI.createOllama3Stylized(query_str, this.qa_type, uuid,chat_id)
          .then((res) => resolve(processResponse(res)))
          .catch((err) => {
            console.error('Request failed:', err)
            resolve({
              error: 1,
              reader: null,
              needLogin: false,
            })
          })
      })
    },
  },
})
