"""本地 bge embedding 封装，聚类用（无 API）。"""
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from src.config import EMBED_MODEL


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


class BgeEmbeddings:
    """简化版：只暴露 encode_documents,Clusterer 直接用 numpy 数组。
    没有 LangChain Embeddings 接口包装(Agent 项目不需要),避免依赖耦合。
    """

    def encode_documents(self, texts: list[str]):
        """返回 numpy 数组 (N, dim)，已 L2 归一化。直接相乘即为 cosine 相似度。"""
        return _model().encode(texts, normalize_embeddings=True, batch_size=32)

    def encode_query(self, text: str):
        prefix = "Represent this sentence for searching relevant passages: "
        return _model().encode(prefix + text, normalize_embeddings=True)
