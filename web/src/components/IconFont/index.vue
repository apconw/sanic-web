<script lang="ts">
export default defineComponent({
  name: 'IconFont',
  props: {
    icon: {
      type: String,
      default: '',
    },
    shadow: {
      type: Boolean,
      default: false,
    },
    verticalCenter: {
      type: Boolean,
      default: false,
    },
    cursor: {
      type: Boolean,
      default: false,
    },
    disabled: {
      type: Boolean,
      default: false,
    },
  },
  emits: ['click'],
  setup(props, { emit }) {
    const getClassName = computed(() => {
      const className: string[] = []
      if (props.verticalCenter) {
        className.push('middle')
      }
      if (props.cursor) {
        className.push('cursor')
      }
      if (props.disabled) {
        className.push('disabled')
      }
      return className
    })

    const handleClick = () => {
      !props.disabled && emit('click')
    }

    const getAttrs = () => {
      const attrs: any = {}
      props.shadow
      && (attrs.filter = 'url(#drop-shadow)')
      return attrs
    }
    return {
      getClassName,

      handleClick,
      getAttrs,
    }
  },
})
</script>

<template>
  <svg
    class="icon-font"
    aria-hidden="true"
    :class="getClassName"
    @click="handleClick()"
  >
    <filter
      id="drop-shadow"
      xmlns="http://www.w3.org/2000/svg"
    >
      <feGaussianBlur
        in="SourceAlpha"
        stdDeviation="2"
      />
      <feOffset
        dx="1"
        dy="1"
        result="offsetblur"
      />
      <feComponentTransfer>
        <feFuncA
          type="linear"
          slope="0.2"
        />
      </feComponentTransfer>
      <feMerge>
        <feMergeNode />
        <feMergeNode in="SourceGraphic" />
      </feMerge>
    </filter>
    <g
      v-bind="getAttrs()"
    >
      <use :xlink:href="`#${icon}`" />
    </g>
  </svg>
</template>

<style lang="scss" scoped>
.icon-font {
  width: 1em;
  height: 1em;
  vertical-align: -0.15em;
  fill: currentcolor;
  overflow: hidden;

  &.middle {
    vertical-align: middle;
  }

  &.cursor {
    cursor: pointer;
  }

  &.disabled {
    color: #c0c4cc;
    cursor: not-allowed;
  }
}
</style>
