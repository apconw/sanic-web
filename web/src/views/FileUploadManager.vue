<script lang="tsx" setup>
import type { UploadFileInfo } from 'naive-ui'

// 全局存储
const businessStore = useBusinessStore()

// 在您的项目中添加类型扩展
interface ExtendedUploadFileInfo extends UploadFileInfo {
  error?: Error
}

// 定义模型，用于双向绑定上传文件列表
const pendingUploadFileInfoList = defineModel<ExtendedUploadFileInfo[]>({ default: () => [] })

// 上传附件 下拉菜单的选项
const options = [
  {
    key: 'document',
    type: 'render',
    render() {
      return (
        <n-upload
          accept=".docx,.ppt,.pptx,.pdf,.txt,.xlsx,.csv"
          default-upload={false}
          show-file-list={false}
          multiple={false}
          onChange={(res) => {
            pendingUploadFileInfoList.value.push(res.file)
            // 触发实际上传
            handleFileUpload(res.file)
          }}
        >
          <div class="px-4">
            <div
              flex="~ items-center gap-4"
              class="cursor-pointer px-12 py-4 hover:bg-primary/10 transition-all-300"
            >
              <span class="i-material-symbols:file-open-outline text-16" />
              <span>上传文档</span>
            </div>
          </div>
        </n-upload>
      )
    },
  },
  {
    key: 'image',
    type: 'render',
    render() {
      return (
        <div onClick={(e) => {
          // 阻止事件冒泡
          e.stopPropagation()
          // 显示提示信息
          window.$ModalMessage.info('暂不支持图片解析')
        }}
        >
          <div class="px-4">
            <div
              flex="~ items-center gap-4"
              class="cursor-pointer px-12 py-4 hover:bg-primary/10 transition-all-300"
            >
              <span class="i-mdi:file-image-outline text-16" />
              <span>上传图片</span>
            </div>
          </div>
        </div>
      )
    },
  },
]

// 处理文件上传的函数
const handleFileUpload = async (fileInfo: ExtendedUploadFileInfo) => {
  const formData = new FormData()
  if (fileInfo.file) {
    formData.append('file', fileInfo.file)
  }

  try {
    const response = await fetch('sanic/file/upload_file_and_parse', {
      method: 'POST',
      body: formData,
    })

    const result = await response.json()

    if (result.code === 200) {
      // 更新文件状态为成功
      const index = pendingUploadFileInfoList.value.findIndex((f) => f.id === fileInfo.id)
      if (index !== -1) {
        pendingUploadFileInfoList.value[index].status = 'finished'
        pendingUploadFileInfoList.value[index].percentage = 100
        businessStore.add_file(result.data)
      }
      window.$ModalMessage.success(`文件上传并解析成功`)
    } else {
      // 更新文件状态为失败
      const index = pendingUploadFileInfoList.value.findIndex((f) => f.id === fileInfo.id)
      if (index !== -1) {
        pendingUploadFileInfoList.value[index].status = 'error'
        pendingUploadFileInfoList.value[index].error = new Error(result.msg || '上传失败')
      }
      window.$ModalMessage.error(`文件上传失败`)
    }
  } catch (error) {
    // 更新文件状态为失败
    const index = pendingUploadFileInfoList.value.findIndex((f) => f.id === fileInfo.id)
    if (index !== -1) {
      pendingUploadFileInfoList.value[index].status = 'error'
      if (error instanceof Error) {
        pendingUploadFileInfoList.value[index].error = error
      } else {
        pendingUploadFileInfoList.value[index].error = new Error(String(error))
      }
    }
    window.$ModalMessage.error(`文件上传失败: ${error instanceof Error ? error.message : String(error)}`)
  }
}

const UploadWrapperItem = defineComponent({
  name: 'UploadWrapperItem',
  props: {
    fileInfo: {
      type: Object as PropType<UploadFileInfo>,
      default: () => null,
    },
  },
  emits: ['remove'],
  setup(props, { emit }) {
    // 解析中、解析失败、解析完成
    const statusList = ref([
      {
        status: 'parsing',
        text: '解析中...',
        icon: 'i-svg-spinners:6-dots-rotate',
      },
      {
        status: 'failed',
        text: '解析失败',
        icon: 'i-carbon:error c-red',
      },
      {
        status: 'success',
        text: '解析完成',
        icon: 'i-carbon:checkmark',
      },
    ])

    // 根据实际上传文件状态确定解析状态
    const _status = computed(() => {
      if (props.fileInfo.status === 'finished') {
        if ((props.fileInfo as ExtendedUploadFileInfo).percentage === 100 && !(props.fileInfo as ExtendedUploadFileInfo).error) {
          return 'success'
        } else if ((props.fileInfo as ExtendedUploadFileInfo).error) {
          return 'failed'
        }
        return 'parsing'
      } else if (props.fileInfo.status === 'error') {
        return 'failed'
      }
      return 'parsing'
    })

    const isImage = computed(() => {
      return props.fileInfo.type?.includes('image')
    })

    const fileName = computed(() => {
      return props.fileInfo.name || ''
    })

    const previewImageUrl = ref('')
    const onImageFile = () => {
      const file = props.fileInfo.file
      if (!file) {
        return
      }

      if (!isImage.value) {
        return
      }

      previewImageUrl.value = URL.createObjectURL(file)
    }

    watchEffect(onImageFile)

    const currentStatus = computed(() => {
      return statusList.value.find((item) => item.status === _status.value)
    })

    const fileTypeIconMap = ref({
      xlsx: 'i-vscode-icons:file-type-excel2',
      xls: 'i-vscode-icons:file-type-excel2',
      csv: 'i-vscode-icons:file-type-excel2',
      docx: 'i-vscode-icons:file-type-word',
      doc: 'i-vscode-icons:file-type-word',
      pdf: 'i-vscode-icons:file-type-pdf2',
      pptx: 'i-vscode-icons:file-type-powerpoint',
      ppt: 'i-vscode-icons:file-type-powerpoint',
    })

    const fileIcon = computed(() => {
      const fileExtension = fileName.value.split('.').pop()?.toLowerCase()
      return fileTypeIconMap.value[fileExtension as any]
    })

    const removeFile = () => {
      emit('remove')
    }

    return {
      isImage,
      previewImageUrl,
      fileName,
      statusList,
      currentStatus,
      fileTypeIconMap,
      fileIcon,
      removeFile,
    }
  },
  render() {
    return (
      <div
        class="relative w-200 px-16 py-5 b b-solid b-bgcolor rounded-8 group transition-all-300"
        flex="~ gap-5 items-center"
      >
        <div class="absolute z-1 top--9 right--9 group-hover:opacity-100 opacity-0 transition-all-300">
          <div
            class="text-20 c-info cursor-pointer i-famicons:remove-circle-outline transition-all-300 hover:c-primary"
            onClick={this.removeFile}
          ></div>
        </div>
        <div class="size-30">
          {
            this.isImage
              ? (
                  <img
                    src={this.previewImageUrl}
                    class="size-full object-contain"
                  />
                )
              : (
                  <div
                    class={[
                      this.fileIcon,
                      'size-full opacity-80',
                    ]}
                  ></div>
                )
          }
        </div>
        <div
          flex="1 ~ col gap-2"
          class="min-w-0 text-13 overflow-x-hidden"
        >
          <n-ellipsis
            tooltip
          >
            {{
              default: () => this.fileName,
              tooltip: () => this.fileName,
            }}
          </n-ellipsis>
          <div
            flex="~ gap-3 items-center"
            class="text-[#999]"
          >
            <span class={[
              'text-12',
              this.currentStatus?.icon,
            ]}
            ></span>
            <span class="text-11">{ this.currentStatus?.text }</span>
          </div>
        </div>
      </div>
    )
  },
})

// 暴露属性和方法给父组件
defineExpose({
  options,
  pendingUploadFileInfoList,
  UploadWrapperItem,
})
</script>

<template>
  <div>
    <div
      v-if="pendingUploadFileInfoList.length"
      class="upload-wrapper-list"
    >
      <UploadWrapperItem
        v-for="(pendingUploadFileInfo, index) in pendingUploadFileInfoList"
        :key="pendingUploadFileInfo.id"
        :file-info="pendingUploadFileInfo"
        @remove="() => {
          // 从businessStore中删除对应的文件
          const fileInfo = pendingUploadFileInfoList[index];
          if (fileInfo.status === 'finished' && fileInfo.percentage === 100) {
            // 获取对应文件的source_file_key并从store中删除
            const fileData = businessStore.file_list.find((f, idx) => idx === index);
            if (fileData) {
              businessStore.remove_file(fileData.source_file_key);
            }
          }
          pendingUploadFileInfoList.splice(index, 1)
        }"
      />
    </div>
  </div>
</template>

<style lang="scss" scoped>
.upload-wrapper-list {
  --at-apply: flex flex-wrap gap-10 items-center;
  --at-apply: pb-12;
}
</style>
