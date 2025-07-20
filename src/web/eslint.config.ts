import antfu from '@antfu/eslint-config'
import pluginVue from 'eslint-plugin-vue'


export default antfu({
  stylistic: {
    indent: 2,
    quotes: 'single',
    semi: false,
  },
  vue: {
    vueVersion: 3,
  },

  ignores: [
    'node_modules',
    'dist',
    'public',
    'src/assets/*',
  ],
  ...pluginVue.configs['flat/base'],
  ...pluginVue.configs['flat/essential'],
  ...pluginVue.configs['flat/strongly-recommended'],
  ...pluginVue.configs['flat/recommended'],
  rules: {
    'vue/prop-name-casing': 'warn',
    'vue/prefer-separate-static-class': 'off',
    'vue/one-component-per-file': 'off',
    'vue/html-self-closing': ['error', {
      html: {
        void: 'never',
        normal: 'never',
        component: 'always',
      },
      svg: 'always',
      math: 'always',
    }],
    'vue/multi-word-component-names': 'off',
    'vue/block-order': ['error', {
      order: ['script', 'template', 'style'],
    }],
    'vue/no-mutating-props': ['error', {
      shallowOnly: true,
    }],
    'vue/eqeqeq': ['error', 'always'],
    'node/prefer-global/process': 'off',
    'no-async-promise-executor': 'off',
    'prefer-spread': 'off',
    'no-case-declarations': 'off',
    'one-var': 'off',
    'unicorn/prefer-includes': 'off',
    'no-console': 'off',
    'unicorn/prefer-node-protocol': 'off',
    'unicorn/no-new-array': 'off',
    'unused-imports/no-unused-vars': 'warn',
    'symbol-description': 'off',
    'prefer-promise-reject-errors': 'off',
    'array-callback-return': 'off',
    'curly': ['error', 'all'],
    'eslint-comments/no-unlimited-disable': 'off',

    'vue/singleline-html-element-content-newline': 'off',
    'vue/attribute-hyphenation': 'off',
    'vue/component-name-in-template-casing': 'off',
    'vue/v-on-event-hyphenation': 'off',

    'ts/no-use-before-define': 'off',
    'ts/method-signature-style': 'off',
    'ts/consistent-type-definitions': 'off',

    'style/no-trailing-spaces': 'error',
    'style/jsx-curly-brace-presence': 'off',
    'style/jsx-one-expression-per-line': 'off',
    'style/jsx-curly-newline': 'off',
    'style/jsx-closing-tag-location': 'off',
    'style/brace-style': ['error', '1tbs'],
    'style/arrow-parens': ['error', 'always'],
    'style/no-multiple-empty-lines': ['error', {
      max: 2,
      maxEOF: 0,
    }],

    'antfu/consistent-list-newline': 'off',
    'antfu/top-level-function': 'off',
    'antfu/if-newline': 'off',

    'eslint-comments/no-aggregating-enable': 'off',
    'eslint-comments/no-duplicate-disable': 'off',
    'eslint-comments/no-unused-enable': 'off',
  },
})
