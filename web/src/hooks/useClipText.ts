export function useClipText() {
  const copied = ref(false)
  const copyDuration = 1500

  const handleCopied = () => {
    copied.value = true
    setTimeout(() => {
      copied.value = false
    }, copyDuration)
  }

  function copy(textToCopy) {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(textToCopy).then(() => {
        handleCopied()
      })
    } else {
      const textArea = document.createElement('textarea')
      textArea.value = textToCopy
      textArea.style.position = 'fixed'
      textArea.style.opacity = '0'
      textArea.style.left = '-999999px'
      textArea.style.top = '-999999px'
      document.body.appendChild(textArea)
      textArea.focus()
      textArea.select()
      return new Promise((resolve, reject) => {
        setTimeout(() => {
          const exec = document.execCommand('copy')
          if (exec) {
            handleCopied()
            resolve('')
          } else {
            reject(new Error())
          }
          textArea.remove()
        })
      })
    }
  }

  return {
    copy,
    copied,
    copyDuration,
  }
}
