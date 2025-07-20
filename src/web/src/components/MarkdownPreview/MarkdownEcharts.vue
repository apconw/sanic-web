<script lang="ts" setup>
import * as echarts from 'echarts'

const props = defineProps({
  chartId: {
    type: String,
    required: true,
  },
})

// 自定义事件用于 子父组件传递事件信息
const emit = defineEmits(['chartRendered'])

// 全局存储
const businessStore = useBusinessStore()

// 定义图表类型
const currentChartType = ref('')
const currentChartTypeName = ref('')

const tableHeaders = ref()
const tableData = ref([])
// echart 示例不能使用vue3定义成响应式的,会影响tips显示
// 在组件挂载完成后初始化 ECharts
onMounted(() => {
  const chartInstance = echarts.init(document.getElementById(props.chartId))

  currentChartType.value = businessStore.writerList.data.template_code

  // 初始化两个空数组
  const productArray: string[] = []
  const casesArray: number[] = []
  let currentChart = ''
  if (currentChartType.value === 'temp01') {
    // 表格
    tableData.value = businessStore.writerList.data.data
    tableHeaders.value = Object.keys(tableData.value[0])
    currentChartTypeName.value = '表格'
    return
  } else if (currentChartType.value === 'temp02') {
    // 饼图
    currentChart = 'pie'
    currentChartTypeName.value = '饼图'
  } else if (currentChartType.value === 'temp03') {
    // 柱状图
    currentChart = 'bar'
    currentChartTypeName.value = '柱状图'
    // 移除第一行标题
    const data = businessStore.writerList.data.data
    data.shift()

    // 遍历数据
    data.forEach((item) => {
      const product = item[0]
      const cases = Number.parseInt(item[1], 10)

      // 只当转换为数字成功时才添加到数组中
      if (!Number.isNaN(cases)) {
        productArray.push(product)
        casesArray.push(cases)
      }
    })
  } else if (currentChartType.value === 'temp04') {
    // 折线图
    currentChart = 'line'
    currentChartTypeName.value = '折线图'
  }

  const options = reactive({
    line: {
      animation: 'auto',
      animationDuration: 1000,
      animationDurationUpdate: 500,
      animationEasing: 'cubicInOut',
      animationEasingUpdate: 'cubicInOut',
      animationThreshold: 2000,
      progressiveThreshold: 3000,
      progressive: 400,
      hoverLayerThreshold: 3000,
      useUTC: false,
      tateAnimation: {
        duration: 300,
        easing: 'cubicOut',
      },
      tooltip: {
        axisPointer: {
          type: 'shadow',
        },
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        height: 400,
        containLabel: true,
      },
      xAxis: {
        type: 'category',
        data: businessStore.writerList.data.data[0],
        nameTextStyle: {
          fontSize: 14,
          fontFamily: 'PMingLiU',
          fontWeight: 400,
        },
      },

      yAxis: {
        type: 'value',
      },
      series: [
        {
          data: businessStore.writerList.data.data[1],
          type: 'line',
        },
      ],
    },
    bar: {
      animation: 'auto',
      animationDuration: 1000,
      animationDurationUpdate: 500,
      animationEasing: 'cubicInOut',
      animationEasingUpdate: 'cubicInOut',
      animationThreshold: 2000,
      progressiveThreshold: 3000,
      progressive: 400,
      hoverLayerThreshold: 3000,
      useUTC: false,
      // legend: {
      //     data: ['案件数量'],
      //     left: 'left',
      //     padding: [0, 20, 10, 0] //设置图例的内边距，顺序是 [上, 右, 下, 左]
      // },
      tateAnimation: {
        duration: 300,
        easing: 'cubicOut',
      },
      tooltip: {
        axisPointer: {
          type: 'shadow',
        },
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        height: 540,
        containLabel: true,
      },
      xAxis: {
        type: 'category',
        data: productArray,
        axisLabel: {
          // 展示角度
          rotate: 60,
        },
        nameTextStyle: {
          fontSize: 20,
          fontFamily: 'PMingLiU',
          fontWeight: 400,
        },
        axisTick: {
          alignWithLabel: true,
        },
      },
      yAxis: {
        type: 'value',
      },
      series: [
        {
          name: '案件数量',
          type: 'bar',
          barWidth: '60%',
          data: casesArray,
        },
      ],
    },
    pie: {
      legend: {
        top: '1',
        left: 'center',
        bottom: 10,
        itemGap: 12, // 图例的上下间距
        itemWidth: 10, // 图例左侧图块的长度
        textStyle: {
          fontSize: 14,
          fontFamily: 'PMingLiU',
          fontWeight: 400,
        },
      },
      series: [
        {
          name: '',
          type: 'pie',
          radius: ['40%', '70%'],
          avoidLabelOverlap: true,
          label: {
            show: true,
            position: 'center',
          },
          emphasis: {
            label: {
              show: true,
              fontSize: '40',
              fontWeight: 'bold',
            },
          },
          labelLine: {
            show: true,
          },
          data: businessStore.writerList.data.data,
        },
      ],
    },
  })

  chartInstance.setOption(options[currentChart])

  businessStore.clearWriterList()
  emit('chartRendered')
})
</script>

<template>
  <div style="background-color: #fff">
    <n-card
      :title="currentChartTypeName"
      embedded
      bordered
      :content-style="{
        'background-color': '#ffffff',
        'color': '#26244c',
      }"
      :header-style="{
        'color': '#26244c',
        'height': '10px',
        'background-color': '#f0effe',
        'text-align': 'left',
        'font-size': '14px',
        'font-family': 'PMingLiU',
      }"
      :footer-style="{
        'color': '#666',
        'background-color': '#ffffff',
        'text-align': 'left',
        'font-size': '14px',
        'font-family': 'PMingLiU',
      }"
    >
      <div
        :id="chartId"
        :style="{
          'width': `100%`,
          'height': `500px`,
          'padding': `30px`,
          'background-color': `#fff`,
        }"
      ></div>
      <template #footer>
        数据来源: 大模型生成的数据, 以上信息仅供参考
      </template>
    </n-card>
  </div>
</template>
