type ContentResult = {
  content: string
  done?: never
}

type DoneResult = {
  done: true
  content?: never
}

type TransformResult = ContentResult | DoneResult
type TransformFunction<T = any> = (rawValue: T, ...args: any) => TransformResult

/**
 * 转义处理响应值为 data: 的 json 字符串
 * 如: 科大讯飞星火大模型的 response
 */
export const parseJsonLikeData = (content) => {
  if (content.startsWith('data: ')) {
    const dataString = content.substring(6).trim()
    if (dataString === '[DONE]') {
      return {
        done: true,
      }
    }
    try {
      return JSON.parse(dataString)
    } catch (error) {
      console.error('JSON parsing error:', error)
    }
  }
  return null
}

/**
 * 大模型映射列表
 */
export const LLMTypes = [
  {
    label: '模拟数据模型',
    modelName: 'standard',
  },
  {
    label: 'Spark 星火大模型',
    modelName: 'spark',
  },
  {
    label: 'Qwen 2大模型',
    modelName: 'qwen2',
  },
  {
    label: 'SiliconFlow 硅基流动大模型',
    modelName: 'siliconflow',
  },
] as const

export type TransformStreamModelTypes = (typeof LLMTypes)[number]['modelName']

/**
 * 用于处理不同类型流的值转换器
 */
export const transformStreamValue: Record<
  TransformStreamModelTypes,
  TransformFunction
> = {
  standard(readValue: Uint8Array, textDecoder: TextDecoder) {
    let content = ''
    if (readValue instanceof Uint8Array) {
      content = textDecoder.decode(readValue, {
        stream: true,
      })
    } else {
      content = readValue
    }
    return {
      content,
    }
  },
  spark(readValue) {
    const stream = parseJsonLikeData(readValue)
    if (stream.done) {
      return {
        done: true,
      }
    }
    return {
      content: stream.choices[0].delta.content || '',
    }
  },
  siliconflow(readValue) {
    // 与 spark 类似，直接复用
    return this.spark(readValue)
  },
  qwen2(readValue) {
    const stream = JSON.parse(readValue)
    return {
      content: stream.content,
    }
  },
}

const processParts = (
  buffer,
  controller: TransformStreamDefaultController,
  splitOn,
) => {
  const parts = buffer.split(splitOn)
  parts.slice(0, -1).forEach((part) => {
    if (part.trim() !== '') {
      controller.enqueue(part)
    }
  })
  return parts[parts.length - 1]
}

export const splitStream = (splitOn): TransformStream<string, string> => {
  let buffer = ''
  return new TransformStream({
    transform(chunk, controller) {
      buffer += chunk

      if (buffer.trim().startsWith('data:')) {
        buffer = processParts(buffer, controller, splitOn)
      } else {
        // 尝试是否能够直接解析为 JSON
        try {
          JSON.parse(buffer)
          buffer = processParts(buffer, controller, splitOn)
        } catch (error) {
          // 如果解析失败，按原文本处理
          controller.enqueue(chunk)
          buffer = ''
        }
      }
    },
    flush(controller) {
      if (buffer.trim() !== '') {
        controller.enqueue(buffer)
      }
    },
  })
}
