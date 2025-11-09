"""
上下文管理实验 目前未使用
压缩、裁剪、卸载、保存上下文
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import trim_messages

# 条件导入，根据平台使用不同的嵌入实现
USE_DASHSCOPE = os.getenv("MODEL_TYPE").lower() == "qwen"

if USE_DASHSCOPE:
    import dashscope
    from http import HTTPStatus
else:
    from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)


class DashScopeEmbeddings:
    """阿里云 DashScope 嵌入模型实现"""

    def __init__(self, model, api_key):
        self.model = model
        self.api_key = api_key

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """为文档列表生成嵌入向量"""
        embeddings = []
        for text in texts:
            resp = dashscope.TextEmbedding.call(model=self.model, api_key=self.api_key, input=text)
            if resp.status_code == HTTPStatus.OK:
                embeddings.append(resp.output["embeddings"][0]["embedding"])
            else:
                logger.error(f"DashScope embedding failed: {resp.message}")
                # 返回零向量作为回退
                embeddings.append([0.0] * 1024)  # 假设维度为1024
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """为查询文本生成嵌入向量"""
        resp = dashscope.TextEmbedding.call(model=self.model, api_key=self.api_key, input=text)
        if resp.status_code == HTTPStatus.OK:
            return resp.output["embeddings"][0]["embedding"]
        else:
            logger.error(f"DashScope embedding failed: {resp.message}")
            # 返回零向量作为回退
            return [0.0] * 1024


class ConversationHistoryManager:
    """
    对话历史管理器，负责对话历史的存储、检索、清理等操作
    """

    def __init__(self):
        """初始化对话历史管理器"""
        # 配置参数
        self.max_age_days = int(os.getenv("CONVERSATION_MAX_AGE_DAYS", 7))
        self.max_sessions = int(os.getenv("CONVERSATION_MAX_SESSIONS", 1000))
        self.max_rounds_per_session = int(os.getenv("CONVERSATION_MAX_ROUNDS", 5))
        self.max_tokens_per_session = int(os.getenv("CONVERSATION_MAX_TOKENS", 30000))
        self.model = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v4")
        self.model_api_key = os.getenv("MODEL_API_KEY")
        self.model_base_url = os.getenv("MODEL_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

        """启动时加载所有会话存储"""
        # 检查是否已经加载过
        self.conversations_loaded = False

        # 使用与 db_service 相同的根目录，但使用不同的子目录
        self.storage_root_dir = os.getenv("VECTOR_INDEX_DIR", "./vector_index")
        self.conversation_storage_dir = os.path.join(self.storage_root_dir, "conversation_history")

        # 确保目录存在
        Path(self.conversation_storage_dir).mkdir(parents=True, exist_ok=True)

        # 初始化向量存储
        if USE_DASHSCOPE:
            # 使用阿里云 DashScope API
            self.embeddings = DashScopeEmbeddings(self.model, self.model_api_key)
        else:
            # 使用 OpenAI API 兼容的实现
            self.embeddings = OpenAIEmbeddings(
                api_key=self.model_api_key, base_url=self.model_base_url, model=self.model
            )

        # 存储结构
        self.conversation_store: Dict[str, FAISS] = {}  # 会话向量存储
        self.conversation_metadata: Dict[str, Dict[str, Any]] = {}  # 会话元数据
        self.conversation_cache: Dict[str, List[BaseMessage]] = {}  # 会话缓存

        # 启动定期清理任务
        self.cleanup_task = None

    def start_cleanup_task(self):
        """启动定期清理任务"""
        if self.cleanup_task is None or self.cleanup_task.done():
            self.cleanup_task = asyncio.create_task(self._periodic_cleanup())

    async def _periodic_cleanup(self):
        """定期清理过期会话"""
        while True:
            try:
                self.cleanup_old_conversations()
            except Exception as e:
                logger.error(f"清理过期会话时出错: {e}")
            # 每小时执行一次清理
            await asyncio.sleep(3600)

    def update_conversation_metadata(self, session_id: str):
        """更新会话元数据"""
        if session_id not in self.conversation_metadata:
            self.conversation_metadata[session_id] = {
                "created_at": datetime.now(),
                "last_accessed": datetime.now(),
                "message_count": 0,
            }
        else:
            self.conversation_metadata[session_id]["last_accessed"] = datetime.now()
            self.conversation_metadata[session_id]["message_count"] += 1

    def store_conversation_history(self, session_id: str, messages: List[BaseMessage]):
        """将对话历史存储到向量数据库"""
        if not messages:
            return

        try:
            # 过滤出有内容的消息
            valid_messages = [msg for msg in messages if msg.content]

            if session_id not in self.conversation_store:
                # 创建新的向量存储
                texts = [f"{msg.type}:{msg.content}" for msg in valid_messages]
                if texts:
                    self.conversation_store[session_id] = FAISS.from_texts(texts=texts, embedding=self.embeddings)
            else:
                # 添加新消息到现有向量存储
                texts = [
                    f"{msg.type}:{msg.content}"
                    for msg in valid_messages
                    if f"{msg.type}:{msg.content}"
                    not in str(self.conversation_store[session_id].docstore._dict.values())
                ]
                if texts:
                    self.conversation_store[session_id].add_texts(texts=texts)

            # 更新缓存
            self.conversation_cache[session_id] = messages.copy()

            # 保存到磁盘
            self.save_conversation_store(session_id)

        except Exception as e:
            logger.error(f"存储对话历史时出错: {e}")

    def save_conversation_store(self, session_id: str):
        """保存会话向量存储到磁盘"""
        if session_id in self.conversation_store:
            try:
                session_dir = os.path.join(self.conversation_storage_dir, session_id)
                # 确保会话目录存在
                Path(session_dir).mkdir(parents=True, exist_ok=True)
                # 保存 FAISS 索引和文档
                self.conversation_store[session_id].save_local(session_dir)
                logger.info(f"会话 {session_id} 的向量存储已保存到 {session_dir}")
            except Exception as e:
                logger.error(f"保存会话 {session_id} 向量存储时出错: {e}")

    def load_conversation_store(self, session_id: str) -> bool:
        """从磁盘加载会话向量存储"""
        session_dir = os.path.join(self.conversation_storage_dir, session_id)
        if os.path.exists(session_dir):
            try:
                # 从本地加载 FAISS 索引
                if USE_DASHSCOPE:
                    vector_store = FAISS.load_local(session_dir, self.embeddings, allow_dangerous_deserialization=True)
                else:
                    vector_store = FAISS.load_local(session_dir, self.embeddings, allow_dangerous_deserialization=True)
                self.conversation_store[session_id] = vector_store
                logger.info(f"会话 {session_id} 的向量存储已从 {session_dir} 加载")
                return True
            except Exception as e:
                logger.error(f"加载会话 {session_id} 向量存储时出错: {e}")
        return False

    def retrieve_relevant_history(self, session_id: str, query: str, k: int = 3) -> List[str]:
        """根据查询召回相关对话历史"""
        # 如果会话存储未加载，尝试从磁盘加载
        if session_id not in self.conversation_store:
            if not self.load_conversation_store(session_id):
                return []

        try:
            docs = self.conversation_store[session_id].similarity_search(query, k=k)
            return [doc.page_content for doc in docs]
        except Exception as e:
            logger.error(f"召回对话历史时出错: {e}")
            return []

    def trim_messages_with_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        模型调用前的消息清理，结合上下文管理
        :param state: 状态对象，包含对话消息
        :return: 修剪后的消息列表
        """
        messages = state["messages"]

        # 保留最近的对话轮次
        human_message_indices = [i for i, msg in enumerate(messages) if msg.type == "human"]

        if len(human_message_indices) > self.max_rounds_per_session:
            # 从倒数第max_rounds轮开始保留
            start_index = human_message_indices[-self.max_rounds_per_session]
            recent_messages = messages[start_index:]
        else:
            recent_messages = messages

        # 应用token限制
        trimmed_messages = trim_messages(
            messages=recent_messages,
            max_tokens=self.max_tokens_per_session,
            token_counter=lambda messages: sum(len(msg.content or "") for msg in messages),
            strategy="last",
            allow_partial=False,
            start_on="human",
            include_system=True,
            text_splitter=None,
        )

        return {"llm_input_messages": trimmed_messages}

    def cleanup_old_conversations(self):
        """清理过期会话"""
        now = datetime.now()
        expired_sessions = []

        # 按时间清理
        for session_id, metadata in self.conversation_metadata.items():
            if (now - metadata["last_accessed"]).days > self.max_age_days:
                expired_sessions.append(session_id)

        # 如果仍然超过最大会话数，按访问时间清理最旧的
        if len(self.conversation_metadata) > self.max_sessions:
            sorted_sessions = sorted(self.conversation_metadata.items(), key=lambda x: x[1]["last_accessed"])
            excess_count = len(sorted_sessions) - self.max_sessions
            expired_sessions.extend([sess[0] for sess in sorted_sessions[:excess_count]])

        # 执行清理
        for session_id in set(expired_sessions):
            self.remove_conversation(session_id)

        logger.info(f"清理了 {len(expired_sessions)} 个过期会话")

    def remove_conversation(self, session_id: str):
        """移除指定会话的所有数据"""
        if session_id in self.conversation_store:
            del self.conversation_store[session_id]
        if session_id in self.conversation_metadata:
            del self.conversation_metadata[session_id]
        if session_id in self.conversation_cache:
            del self.conversation_cache[session_id]

        # 删除磁盘上的存储文件
        session_dir = os.path.join(self.conversation_storage_dir, session_id)
        if os.path.exists(session_dir):
            import shutil

            shutil.rmtree(session_dir)
            logger.info(f"已删除会话 {session_id} 的磁盘存储文件")

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话信息"""
        return self.conversation_metadata.get(session_id)

    def get_all_sessions(self) -> List[str]:
        """获取所有会话ID"""
        return list(self.conversation_metadata.keys())

    def clear_all(self):
        """清空所有会话数据"""
        self.conversation_store.clear()
        self.conversation_metadata.clear()
        self.conversation_cache.clear()

        # 删除所有磁盘上的存储文件
        if os.path.exists(self.conversation_storage_dir):
            import shutil

            shutil.rmtree(self.conversation_storage_dir)
            # 重新创建目录
            Path(self.conversation_storage_dir).mkdir(parents=True, exist_ok=True)

    def format_history_context(self, session_id: str, query: str) -> str:
        """格式化历史上下文用于输入到模型"""
        relevant_history = self.retrieve_relevant_history(session_id, query)
        if relevant_history:
            history_context = "\n\n相关历史对话:\n" + "\n".join(relevant_history)
            return history_context
        return ""

    def cache_messages(self, session_id: str, messages: List[BaseMessage]):
        """缓存消息"""
        self.conversation_cache[session_id] = messages.copy()

    def get_cached_messages(self, session_id: str) -> Optional[List[BaseMessage]]:
        """获取缓存的消息"""
        return self.conversation_cache.get(session_id)

    def load_all_conversations(self):
        """启动时加载所有会话存储"""
        # 检查是否已经加载过
        if self.conversations_loaded:
            logger.info("会话存储已加载，跳过重复加载")
            return

        try:
            # 遍历存储目录中的所有会话文件夹
            if os.path.exists(self.conversation_storage_dir):
                for session_id in os.listdir(self.conversation_storage_dir):
                    session_dir = os.path.join(self.conversation_storage_dir, session_id)
                    if os.path.isdir(session_dir):
                        # 加载会话存储
                        if self.load_conversation_store(session_id):
                            # 初始化元数据
                            self.conversation_metadata[session_id] = {
                                "created_at": datetime.now(),
                                "last_accessed": datetime.now(),
                                "message_count": 0,
                            }
                            logger.info(f"发现会话存储: {session_id}")
            logger.info(f"已完成会话存储加载，共加载 {len(self.conversation_metadata)} 个会话")
            self.conversations_loaded = True
        except Exception as e:
            logger.error(f"加载所有会话存储时出错: {e}")
