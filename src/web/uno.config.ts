import path from 'node:path'
import { FileSystemIconLoader } from '@iconify/utils/lib/loader/node-loaders'
import presetRemToPx from '@unocss/preset-rem-to-px'

import {
  defineConfig,
  presetAttributify,
  presetIcons,
  presetWind3,
  transformerAttributifyJsx,
  transformerDirectives,
} from 'unocss'


export default defineConfig({
  presets: [
    presetWind3(),
    presetAttributify(),
    presetIcons({
      customizations: {
        transform(svg) {
          return svg.replace(/#fff/, 'currentColor')
        },
      },
      collections: {
        'my-svg': FileSystemIconLoader(
          path.join(__dirname, 'src/assets/svg'),
        ),
      },
    }),
    presetRemToPx({
      baseFontSize: 4,
    }),
  ],
  transformers: [
    transformerDirectives(),
    transformerAttributifyJsx(),
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
  ],
})
