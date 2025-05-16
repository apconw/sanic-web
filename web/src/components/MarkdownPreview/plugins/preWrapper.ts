import type MarkdownIt from 'markdown-it'
import PrismJsComponents from 'prismjs/components'

export interface Options {
  codeCopyButtonTitle: string
  hasSingleTheme: boolean
}

// 使用正则表达式匹配字符串的第一个字符，并将其转换为大写
function capitalizeFirstLetter(str) {
  return str.replace(/^\w/, (match) => match.toUpperCase())
}

const getBaseLanguageName = (nameOrAlias, components = PrismJsComponents) => {
  const _nameOrAlias = nameOrAlias.toLowerCase()

  const allLanguages = components.languages
  const allLanguageKeys = Object.keys(allLanguages)

  const lang = {
    value: capitalizeFirstLetter(nameOrAlias || 'markdown'),
  }

  for (let index = 0; index < allLanguageKeys.length; index++) {
    const languageKey = allLanguageKeys[index]
    const languageItem = allLanguages[languageKey]

    const { title, alias, aliasTitles } = languageItem

    if (languageKey === _nameOrAlias) {
      lang.value = title
      break
    }

    if (!alias) {
      continue
    }

    if (Array.isArray(alias)) {
      if (aliasTitles && aliasTitles[_nameOrAlias]) {
        lang.value = aliasTitles[_nameOrAlias]
        break
      }

      if (alias.includes(_nameOrAlias)) {
        lang.value = title
        break
      }
    } else {
      if (alias === _nameOrAlias) {
        lang.value = title
        break
      }
    }
  }

  return lang.value
}

export function preWrapperPlugin(md: MarkdownIt, options: Options) {
  const fence = md.renderer.rules.fence!
  md.renderer.rules.fence = (...args) => {
    const [tokens, idx] = args
    const token = tokens[idx]

    // remove title from info
    token.info = token.info.replace(/\[.*\]/, '')

    const active = / active( |$)/.test(token.info) ? ' active' : ''
    token.info = token.info.replace(/ active$/, '').replace(/ active /, ' ')

    const lang = extractLang(token.info)

    const content = fence(...args)
    return (
      `
      <div class="markdown-code-wrapper flex language-${lang}${getAdaptiveThemeMarker(options)}${active}">
        <div class="markdown-code-header">
          <span class="markdown-code-lang">${getBaseLanguageName(lang)}</span>
          <button class="markdown-code-copy">
            <div class="markdown-copy-icon"></div>
            <span class="markdown-copy-text default">复制代码</span>
            <span class="markdown-copy-text done">已复制</span>
          </button>
        </div>
        ${content}
      </div>
      `
    )
  }
}

export function getAdaptiveThemeMarker(options: Options) {
  return options.hasSingleTheme ? '' : ' xx-adaptive-theme'
}

export function extractTitle(info: string, html = false) {
  if (html) {
    return (
      info.replace(/<!--[\s\S]*?-->/g, '').match(/data-title="(.*?)"/)?.[1] || ''
    )
  }
  return info.match(/\[(.*)\]/)?.[1] || extractLang(info) || 'txt'
}

function extractLang(info: string) {
  return info
    .trim()
    .replace(/=(\d*)/, '')
    .replace(/:(no-)?line-numbers(\{| |$|=\d*).*/, '')
    .replace(/(-vue|\{| ).*$/, '')
    .replace(/^vue-html$/, 'template')
    .replace(/^ansi$/, '')
}
