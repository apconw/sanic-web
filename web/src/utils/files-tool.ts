import * as TransformUtils from '@/components/MarkdownPreview/transform'
import { mockEventStreamText } from '@/data'

export const bytesToKB = (bytes) => {
  return bytes / 1024
}

export const bytesToMB = (bytes) => {
  return bytes / (1024 * 1024) // 1048576 = 1024 * 1024
}

export const bytesToGB = (bytes) => {
  return bytes / (1024 * 1024 * 1024) // 1073741824 = 1024 * 1024 * 1024
}

export const bytesToTB = (bytes) => {
  return bytes / (1024 * 1024 * 1024 * 1024) // 1099511627776 = 1024 * 1024 * 1024 * 1024
}

export const formatBytes = (bytes) => {
  if (bytes === 0) {
    return '0 Bytes'
  }
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `≈ ${Number.parseFloat((bytes / k ** i).toFixed(0))} ${sizes[i]}`
}

/**
 * 将文本内容转换为 .txt 文件
 */
export const convertTextToFile = (textContent, fileName) => {
  const blob = new Blob([textContent], {
    type: 'text/plain',
  })
  const file = new File([blob], fileName, {
    type: 'text/plain',
  })
  return file
}


/**
 * 将文本中的非法文件名字符替换为 '-'，并在末尾追加或替换为 .txt 后缀
 * @param {string} text - 要处理的文本
 * @returns {string} - 处理后的文件名
 */
export const sanitizeAndAppendTxtExtension = (text) => {
  // 替换非法文件名字符为 '-'，包括 '.', 逗号和其他非法字符
  const sanitizedText = text.replace(/[/\\:*?"<>|.,;]/g, '-')

  // 使用正则表达式检查并替换最后的后缀
  return `${sanitizedText.replace(/\.[^/.]+$/, '')}.txt`
}


export const createMockReader = (delay = 5): ReadableStreamDefaultReader<string> => {
  const chunkSize = 10
  const originData = mockEventStreamText
  const contentData = originData.repeat(1)
  const encoder = new TextEncoder()
  const encodedData = encoder.encode(contentData)
  let offset = 0

  const stream = new ReadableStream<Uint8Array>({
    async pull(controller) {
      if (offset < encodedData.length) {
        await new Promise((resolve) => setTimeout(resolve, delay))
        const end = Math.min(offset + chunkSize, encodedData.length) // 确保不超出边界
        controller.enqueue(encodedData.subarray(offset, end))
        offset = end
      } else {
        controller.close()
      }
    },
  })

  return stream.pipeThrough(new TextDecoderStream())
    .pipeThrough(TransformUtils.splitStream('\n'))
    .getReader()
}

export function parseMultiJson(jsonStr: string): any[] {
  const jsonArr: any[] = []
  let startIndex = 0

  while (startIndex < jsonStr.length) {
    // 寻找潜在 JSON 对象的开始位置
    const objectStart = jsonStr.indexOf('{', startIndex)
    if (objectStart === -1) {
      // 如果没有找到更多的 JSON 对象，将剩余部分作为单个项添加
      if (startIndex === 0) {
        return [jsonStr] // 整个字符串不是 JSON
      }
      break
    }

    // 寻找 JSON 对象的结束位置
    let openBrackets = 1
    let objectEnd = objectStart + 1
    while (openBrackets > 0 && objectEnd < jsonStr.length) {
      if (jsonStr[objectEnd] === '{') {
        openBrackets++
      } else if (jsonStr[objectEnd] === '}') {
        openBrackets--
      }
      objectEnd++
    }

    // 提取潜在的 JSON 对象
    const potentialJson = jsonStr.substring(objectStart, objectEnd)

    try {
      const parsedJson = JSON.parse(potentialJson)
      jsonArr.push(parsedJson)
      startIndex = objectEnd
    } catch (error) {
      // 如果解析失败，移动到下一个字符并继续搜索
      if (jsonArr.length === 0) {
        return [jsonStr] // 整个字符串不是有效的 JSON
      }
      startIndex = objectStart + 1
    }
  }

  return jsonArr.length > 0 ? jsonArr : [jsonStr]
}
