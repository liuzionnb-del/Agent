"""集中配置：路径、LLM、质量过滤超参。"""
from pathlib import Path
from dotenv import load_dotenv
import os

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# 大模型（兼容 OpenAI 接口的国内厂商）
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# 本地 embedding（聚类用）
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# Semantic Scholar API key（可选，提速率上限）
S2_API_KEY = os.getenv("S2_API_KEY", "")

# 质量过滤超参
MIN_CITATIONS = int(os.getenv("MIN_CITATIONS", "3"))
MAX_AGE_YEARS = int(os.getenv("MAX_AGE_YEARS", "5"))

# 检索每源默认数
DEFAULT_PER_SOURCE = 5

# 报告输出目录
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)
