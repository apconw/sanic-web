import MarkdownIt from 'markdown-it'
import markdownItHighlight from 'markdown-it-highlightjs'

import hljs from './highlight'
import { preWrapperPlugin } from './preWrapper'

const md = new MarkdownIt({
  html: true,
  linkify: true,
  typographer: true,
})

// Customize the image rendering rule
md.renderer.rules.image = function (tokens, idx, options, env, self) {
  const token = tokens[idx]
  token.attrPush(['referrerpolicy', 'no-referrer'])
  return self.renderToken(tokens, idx, options)
}

md.use(markdownItHighlight, {
  hljs,
}).use(preWrapperPlugin, {
  hasSingleTheme: true,
})

export default md
