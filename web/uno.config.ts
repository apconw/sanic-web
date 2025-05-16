import presetRemToPx from '@unocss/preset-rem-to-px'

import {
  defineConfig,
  presetAttributify,
  presetIcons,
  presetWind3,
  toEscapedSelector,
  transformerDirectives,
} from 'unocss'


export default defineConfig({
  presets: [
    presetWind3(),
    presetAttributify(),
    presetIcons(),
    presetRemToPx({
      baseFontSize: 4,
    }),
  ],
  transformers: [
    transformerDirectives(),
  ],
  theme: {
    colors: {
      primary: '#692ee6',
      success: '#52c41a',
      warning: '#fe7d18',
      danger: '#fa5555',
      info: '#909399',
      bgcolor: '#f2ecee',
    },
  },
  rules: [
    [
      'navbar-shadow', {
        'box-shadow': '0 1px 4px rgb(0 21 41 / 8%)',
      },
    ],
    [
      /^wrapper-dialog-(.+)$/,
      ([, name], { rawSelector, theme }) => {
        const themeColor = (theme as any).colors
        const selector = toEscapedSelector(rawSelector)
        return `
          ${selector} {
            display: flex;
            flex-direction: column;
            padding: 0;
            overflow: hidden;
          }
          ${selector} .n-dialog__title {
            padding: var(--n-padding);
          }
          ${selector} .n-dialog__content {
            display: flex;
            flex: 1;
            min-height: 0;
          }
      `
      },
    ],
  ],
})
