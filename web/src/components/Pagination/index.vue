<script lang="tsx" setup>
/**
使用:
 
<Pagination
  v-model:page="currentPage"
  :page-count="100"
  @change="handleChangePage"
/>
 
const currentPage = ref(2)
const handleChangePage = (n: number) => {
  console.log('nnnn', n)
}
 
 */

defineOptions({
  name: 'Pagination',
})
const props = defineProps({
  page: {
    type: Number,
    default: 1,
  },
  pageCount: {
    type: Number,
    default: 1,
  },
})

const emits = defineEmits([
  'update:page',
  'change',
])

const {
  page,
  pageCount,
} = toRefs(props)

/**
 * 当前页码（非数组下标）
 */
const activePage = ref(page.value)


watch(
  () => activePage.value,
  () => {
    emits('update:page', activePage.value)
  },
)

watch(
  () => page.value,
  () => {
    activePage.value = page.value
  },
  {
    deep: true,
  },
)

const handleToChangePage = () => {
  emits('change', activePage.value)
}


/**
 * 当超过此数值时自动分组（包括此值）
 */
const overflowCount = 10

const overflowStartNumber = 6
const overflowEndNumber = 5

/**
 * 是否自动分组
 */
const showGroupsAuto = computed(() => {
  if (pageCount.value < overflowCount) {
    return false
  }
  return true
})


const allPageNumbers = computed(() => {
  return Array
    .from({
      length: pageCount.value,
    })
    .map((_, idx) => idx + 1)
})

/**
 * 是否展示快捷方式跳转 - prev
 */
const showOverflowPrev = computed(() => {
  if (pageCount.value < overflowCount) {
    return false
  }

  if (activePage.value >= overflowStartNumber) {
    return true
  }

  return false
})

/**
 * 是否展示快捷方式跳转 - next
 */
const showOverflowNext = computed(() => {
  if (pageCount.value < overflowCount) {
    return false
  }

  if (
    (
      pageCount.value - activePage.value
    ) >= overflowEndNumber
  ) {
    return true
  }

  return false
})


/**
 * 获取分组展示的前后页码索引
 */
const groupPageRange = computed(() => {
  let startIndex = 0
  let endIndex = pageCount.value
  const extendCenterSize = 3
  const extendBetweenSize = 7

  if (showOverflowPrev.value && showOverflowNext.value) {
    startIndex = activePage.value - extendCenterSize
    endIndex = activePage.value + extendCenterSize - 1
  }

  if (showOverflowPrev.value && !showOverflowNext.value) {
    startIndex = pageCount.value - extendBetweenSize
    endIndex = pageCount.value
  }

  if (!showOverflowPrev.value && showOverflowNext.value) {
    startIndex = 0
    endIndex = extendBetweenSize
  }

  return {
    startIndex,
    endIndex,
  }
})

/**
 * 获取分组展示的前后页码的范围数组
 */
const getGroupPageNumbers = computed(() => {
  const { startIndex, endIndex } = groupPageRange.value
  return allPageNumbers.value.slice(startIndex, endIndex)
})


/**
 * 上一页跳转
 */
const handleToPrev = () => {
  if (activePage.value - 1 <= 0) {
    return
  }
  activePage.value--
  handleToChangePage()
}
/**
 * 下一页跳转
 */
const handleToNext = () => {
  if (activePage.value + 1 > pageCount.value) {
    return
  }
  activePage.value++
  handleToChangePage()
}
/**
 * 上一页加速跳转
 */
const handleToQuickPrev = () => {
  const { startIndex } = groupPageRange.value
  activePage.value = startIndex
  handleToChangePage()
}
/**
 * 下一页加速跳转
 */
const handleToQuickNext = () => {
  const { endIndex } = groupPageRange.value
  activePage.value = endIndex + 1
  handleToChangePage()
}


interface SinglePageContainerType {
  active?: boolean
  disabled?: boolean
  onClick?: () => any
}


/**
 * 分页指示器外层盒子
 */
const RenderSinglePageContainer = ({
  active = false,
  disabled = false,
  onClick = () => { },
  ghost = false,
}: SinglePageContainerType & SetupContext['attrs'], VNode: SetupContext) => {
  const slots = VNode.slots.default
    ? VNode.slots.default()
    : null

  const defaultClassName = [
    'flex items-center justify-center min-w-30px h-30px c-primary cursor-pointer select-none',
    'rounded-3px c-#fff',
    'b',
  ]

  const activeClassName = active ? 'c-primary b-primary b-solid' : ''
  const disabledClassName = disabled ? 'cursor-not-allowed bg-#fafafc c-#c2c2c2 b-#e0e0e6' : ''

  return (
    <div
      onClick={onClick}
      class={
        [
          ...defaultClassName,
          activeClassName,
          disabledClassName,
        ]
      }
    >{ slots }</div>
  )
}


/**
 * 页码容器
 */
const RenderSinglePageNumber = ({ num = 1, ...attrs }: { num: number, key?: number } & SinglePageContainerType) => {
  const onClick = () => {
    activePage.value = num
    handleToChangePage()
  }
  return (
    <RenderSinglePageContainer
      onClick={onClick}
      {...attrs}
    >
      { num }
    </RenderSinglePageContainer>
  )
}


const RenderPartOfNavNumbers = () => {
  return (
    <>
      {
        showOverflowPrev.value ? <RenderSinglePageNumber num={1} /> : null
      }
      {
        showOverflowPrev.value
          ? (
              <RenderSinglePageContainer
                onClick={() => handleToQuickPrev()}
                class={'group'}
              >
                <div class="i-bx:dots-horizontal-rounded group-hover:i-ic:twotone-keyboard-double-arrow-left"></div>
              </RenderSinglePageContainer>
            )
          : null
      }

      {
        getGroupPageNumbers.value.map((num) => (
          <RenderSinglePageNumber
            active={activePage.value === num}
            num={num}
          />
        ))
      }

      {
        showOverflowNext.value
          ? (
              <RenderSinglePageContainer
                onClick={() => handleToQuickNext()}
                class={'group'}
              >
                <div class="i-bx:dots-horizontal-rounded group-hover:i-ic:twotone-keyboard-double-arrow-right"></div>
              </RenderSinglePageContainer>
            )
          : null
      }
      {
        showOverflowNext.value ? <RenderSinglePageNumber num={pageCount.value} /> : null
      }
    </>
  )
}
</script>


<template>
  <div class="pagination-container">
    <RenderSinglePageContainer
      :disabled="activePage === 1"
      ghost
      @click="handleToPrev()"
    >
      <div class="i-ic:twotone-keyboard-arrow-left"></div>
    </RenderSinglePageContainer>
    <template v-if="showGroupsAuto">
      <RenderPartOfNavNumbers />
    </template>
    <template v-else>
      <RenderSinglePageNumber
        v-for="(num) in allPageNumbers"
        :key="num"
        :active="activePage === num"
        :num="num"
      />
    </template>
    <RenderSinglePageContainer
      :disabled="activePage === pageCount"
      @click="handleToNext()"
    >
      <div class="i-ic:twotone-keyboard-arrow-right"></div>
    </RenderSinglePageContainer>
  </div>
</template>

<style lang="scss">
.pagination-container {
  --at-apply: py-10 flex justify-center items-center;

  & > * {
    --at-apply: ml-10px "first:ml-0px" text-14px;
  }
}
</style>
