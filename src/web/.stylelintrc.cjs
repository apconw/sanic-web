module.exports = {
  extends: [
    'stylelint-config-standard',
    'stylelint-config-standard-scss',
    'stylelint-config-recommended-vue',
    'stylelint-config-recommended-vue/scss',
  ],
  plugins: ['@stylistic/stylelint-plugin'],
  ignoreFiles: ['**/*.js', '**/*.ts'],
  defaultSeverity: 'error',
  rules: {
    'unit-disallowed-list': [
      'rem',
      'pt',
    ],
    '@stylistic/indentation': [
      2,
      {
        baseIndentLevel: 0,
      },
    ],
    'no-empty-source': null,
    'block-no-empty': null,
    'declaration-block-no-duplicate-custom-properties': null,
    'font-family-no-missing-generic-family-keyword': null,

    'selector-class-pattern': '^[a-z]([a-z0-9-]+)?(__([a-z0-9]+-?)+)?(__([a-z0-9]+-?)+)?(--([a-z0-9]+-?)+){0,2}$|^Mui.*$|^([a-z][a-z0-9]*)(_[a-z0-9]+)*$',

    'scss/at-mixin-pattern': '^[a-z]([a-z0-9-]+)?(__([a-z0-9]+-?)+)?(__([a-z0-9]+-?)+)?(--([a-z0-9]+-?)+){0,2}$|^Mui.*$|^([a-z][a-z0-9]*)(_[a-z0-9]+)*$',
    'scss/double-slash-comment-whitespace-inside': 'always',
    'scss/dollar-variable-pattern': null,

    'selector-pseudo-class-no-unknown': [
      true,
      {
        ignorePseudoClasses: [
          'export',
          'deep',
        ],
      },
    ],
    'color-function-notation': ['modern', {
      ignore: ['with-var-inside'],
    }],
    'property-no-unknown': null,
    'at-rule-empty-line-before': [
      'always',
      {
        except: ['first-nested', 'blockless-after-same-name-blockless'],
      },
    ],
    'custom-property-empty-line-before': [
      'always',
      {
        except: ['after-custom-property', 'first-nested'],
      },
    ],
    'declaration-empty-line-before': [
      'always',
      {
        except: ['after-declaration', 'first-nested'],
      },
    ],
    'rule-empty-line-before': ['always-multi-line'],

    // 忽视 -webkit-xxxx 等兼容写法
    'property-no-vendor-prefix': [
      true,
      {
        ignoreProperties: ['box-shadow'],
      },
    ],
  },
}
