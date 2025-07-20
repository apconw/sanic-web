<script lang="ts" setup>
const hotSearchItems = [
  '① Text2SQL 文本转SQL 数据库查询',
  '② Echart图表增强且数据问答可视化',
  '③ Text2SQL与Echarts提供数据分析',
  '④ 结合Echarts和Text2SQL提升解读',
  '⑤ 基于大数据底座提升数据问答速度',
]
const headerTwoItems = [
  '① RAG模型优化问答系统流程',
  '② 向量技术提高数据检索效率',
  '③ 整合公网数据提升回答质量',
  '④ RAG框架实现精准通用问答',
  '⑤ 扩展方便对接三方开源系统',
]
const headerThreeItems = [
  '① 大模型解析文件实现智能问答',
  '② 统计学方法深入分析表格数据',
  '③ 表格数据结合大模型精准解读',
  '④ 支持更复杂表格计算增强统计',
  '⑤ 表格问答数据图表可视化展示',
]
</script>

<template>
  <div class="container">
    <div class="header">
      <div class="bg-primary rounded-50%">
        <div class="size-60 bg-white i-my-svg:system-logo"></div>
      </div>
      <div class="header-text">
        <h1 class="header-title">你的全能AI数据助手</h1>
        <p class="header-subtitle">基于大模型的数据问答小助手</p>
      </div>
    </div>
    <div class="content">
      <div class="card-one">
        <h4 class="title">通用问答</h4>
        <p class="card-subtitle">基于RAG大模型的通用问答</p>
        <ul>
          <li
            v-for="(item, index) in headerTwoItems"
            :key="index"
            class="list-item"
          >
            {{ item }}
          </li>
        </ul>
      </div>
      <div class="card-two">
        <h4 class="title">数据问答</h4>
        <p class="card-subtitle">基于大模型的数据问答</p>
        <ul>
          <li
            v-for="(item, index) in hotSearchItems"
            :key="index"
            class="list-item"
          >
            {{ item }}
          </li>
        </ul>
      </div>
      <div class="card-three">
        <h4 class="title">表格问答</h4>
        <p class="card-subtitle">基于大模型的表格问答</p>
        <ul>
          <li
            v-for="(item, index) in headerThreeItems"
            :key="index"
            class="list-item"
          >
            {{ item }}
          </li>
        </ul>
      </div>
    </div>
  </div>
</template>

<style scoped>
body {
  font-family: Arial, sans-serif;
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

.container {
  width: 100%;
  max-width: 900px;
  margin: 0 auto;
  padding: 20px;
}

.header {
  background-color: #f6f7fb;

  /* background: linear-gradient(to right, #f1f0fe, #f3f2ff); */

  border-radius: 5px;
  text-align: center;
  display: flex;
  align-items: center;
  gap: 16px;
  margin-top: 6%;
  justify-content: center;
  padding: 15px;
}

.header-text {
  flex: 1;
  text-align: left;
  flex-grow: 1;
}

.header-title {
  margin: 0; /* Remove default margin */
  font-size: 1.5em; /* Adjust as needed */
}

.header-subtitle {
  margin: 0; /* Remove default margin */
  font-size: 1em; /* Adjust as needed */
  margin-top: 3px; /* Add a small top margin for spacing between title and subtitle */
}

.header > div {
  text-align: left;
}

.content {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
  margin-top: 10px;
}

.card-one {
  border-radius: 10px;
  padding: 20px;
  background: linear-gradient(to bottom, #f0effe, #d4eefc);
}

.card-two {
  border-radius: 10px;
  padding: 20px;
  background: linear-gradient(to right, #f0effe, #d4eefc);
}

.card-three {
  border-radius: 10px;
  padding: 20px;
  background: linear-gradient(to left, #f0effe, #d4eefc);
}

.card-subtitle {
  margin: 0; /* Remove default margin */
  font-size: 13px; /* Adjust as needed */
  color: #6c757d; /* Optional: to match the color of .header-subtitle */
}

.title {
  color: #333;
  font-size: 17px;
  margin-bottom: 5px;
}

.list-item {
  list-style-type: none;
  margin-bottom: 20px;
  margin-top: 10px;
}
</style>
