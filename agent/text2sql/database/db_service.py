import warnings

warnings.filterwarnings("ignore", message=".*pkg_resources.*deprecated.*")

import hashlib
import json
import logging
import os
import re
import time
from functools import lru_cache
from http import HTTPStatus
from typing import Dict, List, Tuple, Optional

import dashscope
import faiss
import jieba
import numpy as np
import pandas as pd
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sqlalchemy.inspection import inspect
from sqlalchemy.sql.expression import text

from agent.text2sql.state.agent_state import AgentState, ExecutionResult
from model.db_connection_pool import get_db_pool

# 日志配置
logger = logging.getLogger(__name__)

# 数据库连接池
db_pool = get_db_pool()

# 向量检索配置
USE_DASHSCOPE_EMBEDDING = True  # 使用阿里云嵌入模型
FORCE_REBUILD_VECTOR_INDEX = os.getenv("FORCE_REBUILD_VECTOR_INDEX", "false").lower() == "true"

# 向量索引存储路径
VECTOR_INDEX_DIR = "./vector_index"
os.makedirs(VECTOR_INDEX_DIR, exist_ok=True)

INDEX_FILE = os.path.join(VECTOR_INDEX_DIR, "schema.index")
METADATA_FILE = os.path.join(VECTOR_INDEX_DIR, "metadata.json")

# 重排模型
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME")
#  嵌入模型
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")

# 初始化 DashScope 客户端
if USE_DASHSCOPE_EMBEDDING:
    MODEL_API_KEY = os.getenv("SMALL_MODEL_API_KEY")
    MODEL_BASE_URL = os.getenv("SMALL_MODEL_BASE_URL")
    if not MODEL_API_KEY:
        raise ValueError("环境变量 MODEL_API_KEY 未设置，无法初始化嵌入模型客户端")
    client = OpenAI(api_key=MODEL_API_KEY, base_url=MODEL_BASE_URL)


class DatabaseService:
    """
    支持混合检索（BM25 + 向量）与索引持久化的数据库服务。
    提供表结构检索、SQL 执行、错误修正 SQL 执行等功能。

    核心执行流程：
    1. 初始化阶段：建立数据库连接池，配置检索参数
    2. 表结构检索：通过混合检索获取与用户查询最相关的表结构
       - 获取所有表结构信息
       - 构建文档文本用于检索
       - 初始化向量索引（加载或重建）
       - BM25 关键词匹配检索
       - 向量相似度检索
       - RRF 融合两种检索结果
       - 使用 DashScope 重排序器优化排序
    3. SQL 执行：执行查询语句并返回结果
    """

    def __init__(self):
        self._engine = db_pool.get_engine()
        self._faiss_index: Optional[faiss.Index] = None
        self._table_names: List[str] = []
        self._corpus: List[str] = []
        self._tokenized_corpus: List[List[str]] = []
        self._index_initialized: bool = False
        self.USE_RERANKER: bool = True  # 是否启用重排序器

    @staticmethod
    def _tokenize_text(text_str: str) -> List[str]:
        """
        对中文/英文文本进行分词，过滤标点符号。

        Args:
            text_str (str): 输入文本

        Returns:
            List[str]: 分词后的 token 列表
        """
        filtered_text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", " ", text_str)
        tokens = jieba.lcut(filtered_text, cut_all=False)
        return [token.strip() for token in tokens if token.strip()]

    def _get_table_comment(self, table_name: str) -> str:
        """
        从 information_schema 中获取指定表的注释。

        Args:
            table_name (str): 表名

        Returns:
            str: 表注释，若无则返回空字符串
        """
        try:
            query = text(
                """
                SELECT table_comment
                FROM information_schema.tables
                WHERE table_schema = DATABASE()
                  AND table_name = :table_name;
                """
            )
            with self._engine.connect() as conn:
                result = conn.execute(query, {"table_name": table_name})
                row = result.fetchone()
                return (row[0] or "").strip()
        except Exception as e:
            logger.warning(f"⚠️ 获取表 {table_name} 注释失败: {e}")
            return ""

    @staticmethod
    def _build_document(table_name: str, table_info: dict) -> str:
        """
        构建用于检索的文档文本（表名 + 注释 + 字段名 + 字段注释）。

        Args:
            table_name (str): 表名
            table_info (dict): 包含列、外键、注释等信息的字典

        Returns:
            str: 拼接后的文档文本
        """
        parts = [table_name]
        if table_info.get("table_comment"):
            parts.append(table_info["table_comment"])
        for col_name, col_info in table_info.get("columns", {}).items():
            parts.append(col_name)
            if col_info.get("comment"):
                parts.append(col_info["comment"])
        return " ".join(parts)

    @lru_cache(maxsize=1)
    def _fetch_all_table_info(self) -> Dict[str, Dict]:
        """
        获取数据库中所有表的结构信息（带 LRU 缓存）。

        Returns:
            Dict[str, Dict]: 表名 → 表结构信息的字典
        """
        start_time = time.time()
        inspector = inspect(self._engine)
        table_names = inspector.get_table_names()
        logger.info(f"🔍 开始加载 {len(table_names)} 张表的 schema 信息...")

        table_info = {}
        for table_name in table_names:
            try:
                columns = {}
                for col in inspector.get_columns(table_name):
                    columns[col["name"]] = {
                        "type": str(col["type"]),
                        "comment": str(col["comment"] or ""),
                    }

                foreign_keys = [
                    f"{fk['constrained_columns'][0]} -> {fk['referred_table']}.{fk['referred_columns'][0]}"
                    for fk in inspector.get_foreign_keys(table_name)
                ]

                table_comment = self._get_table_comment(table_name)

                table_info[table_name] = {
                    "columns": columns,
                    "foreign_keys": foreign_keys,
                    "table_comment": table_comment,
                }
            except Exception as e:
                logger.error(f"❌ 读取表 {table_name} 结构失败: {e}")

        elapsed = time.time() - start_time
        logger.info(f"✅ 成功加载 {len(table_info)} 张表，耗时 {elapsed:.2f}s")
        return table_info

    @staticmethod
    def _generate_schema_fingerprint(table_info: Dict[str, Dict]) -> str:
        """
        生成 schema 的指纹（MD5 哈希），用于检测变更。

        Args:
            table_info (Dict[str, Dict]): 表结构信息

        Returns:
            str: MD5 指纹
        """
        fingerprint_data = {}
        for table_name, info in table_info.items():
            fingerprint_data[table_name] = {
                "comment": info.get("table_comment", ""),
                "columns": sorted(
                    [f"{col_name}:{col_info.get('comment', '')}" for col_name, col_info in info["columns"].items()]
                ),
            }
        json_str = json.dumps(fingerprint_data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(json_str.encode("utf-8")).hexdigest()

    def _load_vector_index(self, table_info: Dict[str, Dict]) -> bool:
        """
        从磁盘加载 FAISS 向量索引和元数据。

        Args:
            table_info (Dict[str, Dict]): 当前表结构

        Returns:
            bool: 是否加载成功
        """
        if not (os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE)):
            logger.info("❌ 向量索引文件不存在，将重建")
            return False

        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            current_fingerprint = self._generate_schema_fingerprint(table_info)
            if metadata.get("fingerprint") != current_fingerprint:
                logger.info("🔄 数据库 schema 已变更，需重建向量索引")
                return False

            self._faiss_index = faiss.read_index(INDEX_FILE)
            self._table_names = metadata["table_names"]
            self._corpus = metadata["corpus"]

            logger.info(f"🎉 成功加载向量索引，包含 {len(self._table_names)} 张表")
            return True

        except Exception as e:
            logger.warning(f"⚠️ 加载向量索引失败: {e}，将重建")
            return False

    def _save_vector_index(self, table_info: Dict[str, Dict]):
        """
        将 FAISS 索引和元数据保存到磁盘。

        Args:
            table_info (Dict[str, Dict]): 当前表结构
        """
        if self._faiss_index is None:
            return

        faiss.write_index(self._faiss_index, INDEX_FILE)

        metadata = {
            "table_names": self._table_names,
            "corpus": self._corpus,
            "fingerprint": self._generate_schema_fingerprint(table_info),
            "updated_at": pd.Timestamp.now().isoformat(),
        }
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ 向量索引已保存至: {INDEX_FILE}")

    @staticmethod
    def _create_embeddings_with_dashscope(texts: List[str]) -> np.ndarray:
        """
        使用 DashScope API 生成文本嵌入向量。

        Args:
            texts (List[str]): 输入文本列表

        Returns:
            np.ndarray: 嵌入向量数组
        """
        logger.info("🌐 调用 DashScope 文本嵌入 API...")
        start_time = time.time()
        embeddings = []
        for doc in texts:
            try:
                response = client.embeddings.create(model=EMBEDDING_MODEL_NAME, input=doc)
                embeddings.append(response.data[0].embedding)
            except Exception as e:
                logger.error(f"❌ 嵌入生成失败 ({doc[:30]}...): {e}")
                embeddings.append(np.zeros(1024))  # 占位符

        embeddings = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings)
        logger.info(f"✅ 嵌入生成完成，耗时 {time.time() - start_time:.2f}s")
        return embeddings

    def _initialize_vector_index(self, table_info: Dict[str, Dict]):
        """
        初始化 FAISS 向量索引：加载或重建。

        Args:
            table_info (Dict[str, Dict]): 表结构信息
        """
        if self._index_initialized:
            return

        if FORCE_REBUILD_VECTOR_INDEX:
            logger.info("💡 强制重建向量索引（环境变量触发）")
        elif self._load_vector_index(table_info):
            self._index_initialized = True
            return

        # 构建新索引
        logger.info("🏗️ 开始构建新的向量索引...")
        start_time = time.time()

        self._table_names = list(table_info.keys())
        self._corpus = [self._build_document(name, info) for name, info in table_info.items()]

        # 生成嵌入
        embeddings = self._create_embeddings_with_dashscope(self._corpus)

        # 初始化 FAISS 索引
        dimension = embeddings.shape[1]
        self._faiss_index = faiss.IndexFlatIP(dimension)  # 内积 = 余弦相似度
        self._faiss_index.add(embeddings)

        # 保存索引
        self._save_vector_index(table_info)

        elapsed = time.time() - start_time
        logger.info(f"🎉 向量索引构建完成，共 {len(self._table_names)} 张表，耗时 {elapsed:.2f}s")
        self._index_initialized = True

    def _retrieve_by_vector(self, query: str, top_k: int = 10) -> List[int]:
        """
        使用向量相似度检索最相关的表。

        该方法通过调用 DashScope 的文本嵌入模型，将用户查询转换为向量，
        并使用 FAISS 向量索引搜索与之最相似的 top_k 个表结构文档。

        Faiss（Facebook AI Similarity Search）是由 Meta (Facebook) 开发的一个高效的相似性搜索库
        faiss-gpu/faiss-cpu 两个版本

        数据映射关系说明：
        - 构建映射：在索引初始化时，按照相同顺序维护 _table_names（表名列表）和 _corpus（文档列表）
        - 向量检索：FAISS 返回相似文档在 _corpus 中的索引位置
        - 索引转换：通过 _table_names[index] 将索引位置转换为具体的表名
        - 结果使用：使用这些表名从原始表信息中提取详细结构

        Args:
            query (str): 用户输入的自然语言查询，例如："查找最近一周的订单信息"
            top_k (int): 需要返回的最相似表的数量，默认值为 10

        Returns:
            List[int]: 与用户查询最相似的表在 corpus 中的索引列表，
                       按照相似度从高到低排序
        """
        try:
            response = client.embeddings.create(model=EMBEDDING_MODEL_NAME, input=query)
            query_vec = np.array([response.data[0].embedding]).astype("float32")
            faiss.normalize_L2(query_vec)
            _, indices = self._faiss_index.search(query_vec, top_k)
            return indices[0].tolist()
        except Exception as e:
            logger.error(f"❌ 向量检索失败: {e}")
            return []

    def _retrieve_by_bm25(self, table_info: Dict[str, Dict], user_query: str) -> List[int]:
        """
        使用 BM25 算法进行关键词匹配检索。

        Args:
            table_info (Dict[str, Dict]): 表结构
            user_query (str): 用户查询

        Returns:
            List[int]: 按相关性排序的索引列表
        """
        if not user_query or not table_info:
            return list(range(len(table_info)))

        logger.info("🔄 执行 BM25 检索...")
        self._corpus = [self._build_document(name, info) for name, info in table_info.items()]
        self._tokenized_corpus = [self._tokenize_text(doc) for doc in self._corpus]
        query_tokens = self._tokenize_text(user_query)

        bm25 = BM25Okapi(self._tokenized_corpus)
        doc_scores = bm25.get_scores(query_tokens)

        # 增强：若查询词出现在表注释中，则提升分数
        enhanced_scores = doc_scores.copy()
        table_comments = [info.get("table_comment", "") for info in table_info.values()]
        for i, (comment, score) in enumerate(zip(table_comments, doc_scores)):
            if score <= 0:
                continue
            comment_tokens = self._tokenize_text(comment)
            overlap = set(query_tokens) & set(comment_tokens)
            if overlap:
                overlap_ratio = len(overlap) / len(set(query_tokens))
                enhanced_scores[i] += score * overlap_ratio * 1.5

        scored_indices = sorted(enumerate(enhanced_scores), key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in scored_indices]

    @staticmethod
    def _rrf_fusion(bm25_indices: List[int], vector_indices: List[int], k: int = 60) -> List[int]:
        """
        使用 RRF（Reciprocal Rank Fusion）融合两种检索结果。

        Args:
            bm25_indices (List[int]): BM25 排序索引
            vector_indices (List[int]): 向量检索排序索引
            k (int): RRF 常数

        Returns:
            List[int]: 融合后排序的索引列表
        """
        scores = {}
        for rank, idx in enumerate(bm25_indices):
            scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
        for rank, idx in enumerate(vector_indices):
            scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
        sorted_indices = sorted(scores.items(), key=lambda x: -x[1])
        return [idx for idx, _ in sorted_indices]

    def _rerank_with_dashscope(self, query: str, candidate_tables: Dict[str, Dict]) -> List[Tuple[str, float]]:
        """
        使用 DashScope GTE-Rerank-V2 对候选表进行重排序。

        Args:
            query (str): 用户查询
            candidate_tables (Dict[str, Dict]): 候选表及其信息

        Returns:
            List[Tuple[str, float]]: (表名, 相关性分数) 列表，按分数降序
        """
        if not self.USE_RERANKER:
            logger.debug("⏭️ Reranker 已禁用，跳过重排序")
            return [(name, 1.0) for name in candidate_tables.keys()]

        try:
            documents = []
            name_to_text = {}
            for table_name, info in candidate_tables.items():
                doc_text = self._build_document(table_name, info)
                documents.append(doc_text)
                name_to_text[table_name] = doc_text

            if not documents:
                return []

            logger.info("🔁 调用 GTE-Rerank-V2 进行重排序...")
            response = dashscope.TextReRank.call(
                api_key=MODEL_API_KEY,
                model=RERANK_MODEL_NAME,
                query=query,
                documents=documents,
                top_n=len(documents),
                return_documents=False,
            )

            if response.status_code == HTTPStatus.OK:
                results = []
                for item in response.output.results:
                    table_name = next(name for name, text in name_to_text.items() if text == documents[item.index])
                    results.append((table_name, item.relevance_score))
                results.sort(key=lambda x: x[1], reverse=True)
                logger.info("✅ Rerank 完成")
                return results
            else:
                logger.warning(f"⚠️ Rerank API 调用失败: {response.message}")
                return [(name, 1.0) for name in candidate_tables.keys()]

        except Exception as e:
            logger.error(f"❌ Rerank 过程出错: {e}")
            return [(name, 1.0) for name in candidate_tables.keys()]

    def get_table_schema(self, state: AgentState) -> AgentState:
        """
        根据用户查询，通过混合检索筛选出最相关的数据库表结构。

        Args:
            state (AgentState): 当前状态，包含 user_query

        Returns:
            AgentState: 更新后的状态，包含 db_info
        """
        try:
            logger.info("🔍 开始获取数据库表 schema 信息")
            all_table_info = self._fetch_all_table_info()

            user_query = state.get("user_query", "").strip()
            if not user_query:
                state["db_info"] = all_table_info
                logger.info(f"ℹ️ 无用户查询，返回全部 {len(all_table_info)} 张表")
                return state

            # 初始化向量索引
            self._initialize_vector_index(all_table_info)

            # 混合检索
            logger.info("🔍 开始混合检索：BM25 + 向量检索")
            bm25_top_indices = self._retrieve_by_bm25(all_table_info, user_query)
            logger.info(f"📊 BM25检索返回 {len(bm25_top_indices)} 个结果")
            vector_top_indices = self._retrieve_by_vector(user_query, top_k=20)
            logger.info(f"🔗 向量检索返回 {len(vector_top_indices)} 个结果")

            # 过滤：仅保留同时在 BM25 前 50 和向量结果中的表
            valid_bm25_set = set(bm25_top_indices[:50])
            candidate_indices = [idx for idx in vector_top_indices if idx in valid_bm25_set]
            logger.info(f"🎯 初步筛选后保留 {len(candidate_indices)} 个候选表")

            if not candidate_indices:
                candidate_indices = bm25_top_indices[:4]  # 降级
                logger.info("⚠️ 候选表为空，降级使用BM25前4个结果")

            fused_indices = self._rrf_fusion(bm25_top_indices, candidate_indices, k=60)
            logger.info(f"🔄 RRF融合后得到 {len(fused_indices)} 个结果")

            # 评分筛选
            selected_indices = []
            for idx in fused_indices:
                bm25_rank = bm25_top_indices.index(idx) + 1 if idx in bm25_top_indices else len(all_table_info) + 1
                vector_rank = (
                    vector_top_indices.index(idx) + 1 if idx in vector_top_indices else len(all_table_info) + 1
                )
                score = 1 / (60 + bm25_rank) + 1 / (60 + vector_rank)
                if score >= 0.01 and len(selected_indices) < 10:
                    selected_indices.append(idx)

            candidate_table_names = [self._table_names[i] for i in selected_indices]
            candidate_table_info = {name: all_table_info[name] for name in candidate_table_names}

            # 重排序
            reranked_results = self._rerank_with_dashscope(user_query, candidate_table_info)
            final_table_names = [name for name, _ in reranked_results][:4]  # 取 top 4

            # 构建输出
            filtered_info = {name: all_table_info[name] for name in final_table_names}

            # 打印结果摘要
            print(f"\n🔍 用户查询: {user_query}")
            print("📊 检索与排序结果:")
            for i, (table_name, score) in enumerate(reranked_results):
                bm25_idx = self._table_names.index(table_name) if table_name in self._table_names else -1
                bm25_rank = bm25_top_indices.index(bm25_idx) + 1 if bm25_idx in bm25_top_indices else "-"
                vector_rank = vector_top_indices.index(bm25_idx) + 1 if bm25_idx in vector_top_indices else "-"
                print(
                    f"  {i + 1}. {table_name:<15} | BM25: {bm25_rank:>2} | Vector: {vector_rank:>2} | Rerank: {score:.3f}"
                )

            state["db_info"] = filtered_info
            logger.info(f"✅ 最终筛选出 {len(filtered_info)} 个相关表: {list(filtered_info.keys())}")

        except Exception as e:
            logger.error(f"❌ 获取数据库表信息失败: {e}", exc_info=True)
            state["db_info"] = {}
            state["execution_result"] = ExecutionResult(success=False, error="无法连接数据库或获取元数据")

        return state

    @staticmethod
    def execute_sql(state: AgentState) -> AgentState:
        """
        执行生成的 SQL 语句。

        Args:
            state (AgentState): 包含 generated_sql 的状态

        Returns:
            AgentState: 更新执行结果
        """
        generated_sql = state.get("generated_sql", "").strip()
        if not generated_sql:
            error_msg = "SQL 为空，无法执行"
            logger.warning(error_msg)
            state["execution_result"] = ExecutionResult(success=False, error=error_msg)
            return state

        logger.info("▶️ 执行 SQL 语句")
        try:
            with db_pool.get_session() as session:
                result = session.execute(text(generated_sql))
                result_data = result.fetchall()
                columns = result.keys()
                frame = pd.DataFrame(result_data, columns=columns)
                state["execution_result"] = ExecutionResult(success=True, data=frame.to_dict(orient="records"))
                logger.info(f"✅ SQL 执行成功，返回 {len(result_data)} 条记录")
        except Exception as e:
            error_msg = f"执行 SQL 失败: {e}"
            logger.error(error_msg, exc_info=True)
            state["execution_result"] = ExecutionResult(success=False, error=str(e))
        return state

    @staticmethod
    def execute_correction_sql(state: AgentState) -> AgentState:
        """
        执行修正后的 SQL 语句。

        Args:
            state (AgentState): 包含 correction_result 的状态

        Returns:
            AgentState: 更新执行结果
        """
        correction_result = state.get("correction_result")
        if not correction_result or not hasattr(correction_result, "corrected_sql_query"):
            error_msg = "无修正后的 SQL 可执行"
            logger.warning(error_msg)
            state["execution_result"] = ExecutionResult(success=False, error=error_msg)
            return state

        corrected_sql = getattr(correction_result, "corrected_sql_query", "").strip()
        if not corrected_sql:
            error_msg = "修正后的 SQL 为空"
            logger.warning(error_msg)
            state["execution_result"] = ExecutionResult(success=False, error=error_msg)
            return state

        logger.info("🔧 执行修正后的 SQL 语句")
        try:
            with db_pool.get_session() as session:
                result = session.execute(text(corrected_sql))
                result_data = result.fetchall()
                columns = result.keys()
                frame = pd.DataFrame(result_data, columns=columns)
                state["execution_result"] = ExecutionResult(success=True, data=frame.to_dict(orient="records"))
                logger.info(f"✅ 修正 SQL 执行成功，返回 {len(result_data)} 条记录")
        except Exception as e:
            error_msg = f"执行修正 SQL 失败: {e}"
            logger.error(error_msg, exc_info=True)
            state["execution_result"] = ExecutionResult(success=False, error=str(e))
        return state
