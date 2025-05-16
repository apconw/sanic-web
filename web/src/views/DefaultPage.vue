<script lang="tsx" setup>
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
      <n-icon
        size="60"
        class="mr-15 bg-#716bea rounded-100%"
      >
        <svg
          t="1730167173147"
          class="icon"
          viewBox="0 0 1024 1024"
          version="1.1"
          xmlns="http://www.w3.org/2000/svg"
          p-id="29304"
        >
          <path
            d="M964.266667 609.848889a48.298667 48.298667 0 0 0-2.844445-3.982222l-113.379555-198.656 50.915555-89.315556a26.510222 26.510222 0 0 0 5.347556-32.711111l-64.967111-112.469333a25.144889 25.144889 0 0 0-21.959111-12.8H588.8l-52.053333-90.225778A25.144889 25.144889 0 0 0 514.503111 56.888889H395.776a25.6 25.6 0 0 0-24.177778 17.351111L257.991111 272.839111H152.291556a20.821333 20.821333 0 0 0-5.859556-0.568889 25.144889 25.144889 0 0 0-22.186667 13.084445L60.245333 398.449778a26.510222 26.510222 0 0 0 0 25.884444l114.460445 200.362667-51.2 90.567111a26.510222 26.510222 0 0 0 0 25.884444l59.164444 104.163556a25.201778 25.201778 0 0 0 27.306667 12.231111h226.304l55.352889 96.768a25.6 25.6 0 0 0 20.821333 12.8h1.706667l129.137778-0.568889c9.045333 0 17.408-4.892444 21.902222-12.8l114.176-200.647111h103.082666a26.396444 26.396444 0 0 0 21.959112-13.368889l59.335111-104.220444a26.510222 26.510222 0 0 0 0-25.884445l0.455111 0.284445z m-143.758223 12.515555l-62.407111-109.226666-244.167111 428.657778-62.748444-109.624889H206.279111l60.359111-105.813334h124.359111l-244.053333-427.52h125.838222L395.548444 83.626667l60.416 105.870222-62.407111 109.340444h486.855111l-61.952 108.430223 122.652445 215.210666h-120.832l0.227555-0.056889z"
            p-id="29305"
            fill="#ffffff"
          />
          <path
            d="M512.170667 663.722667l151.552-246.556445H360.277333z"
            p-id="29306"
            fill="#ffffff"
          />
        </svg>
      </n-icon>
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
  margin-top: 6%;
  justify-content: center;
  padding: 15px;
}

.header-text {
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
  flex-grow: 1;
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
