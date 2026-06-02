"""统一数据模型：Paper(检索结果) + Finding/Cluster/Subquestion(后续阶段用)。"""
from dataclasses import dataclass, field, asdict
from typing import Literal


@dataclass
class Paper:
    """统一论文表示，三个 Tool 都返回这个。"""
    source: Literal["arxiv", "semantic_scholar", "pubmed"]
    paper_id: str                       # source-specific id
    title: str
    abstract: str
    year: int | None
    citation_count: int = 0
    venue: str | None = None
    url: str = ""
    authors: list[str] = field(default_factory=list)
    external_ids: dict = field(default_factory=dict)  # {"DOI":..., "ArXivId":..., ...}

    def dedupe_key(self) -> str:
        """跨源去重键：DOI > arxiv id > title 前 60 字符。"""
        return (
            self.external_ids.get("DOI")
            or self.external_ids.get("ArXivId")
            or self.title[:60].strip().lower()
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Subquestion:
    id: str
    text: str
    keywords: list[str]


@dataclass
class Finding:
    """Reader 提炼出的一条结构化结论。"""
    id: str
    paper_id: str
    paper_title: str
    paper_url: str
    paper_year: int
    paper_citation_count: int
    key_point: str
    quote: str
    relevance: float
    subquestion_id: str


@dataclass
class Cluster:
    """Clusterer 合并同质 finding 后的聚类。"""
    id: str
    findings: list[Finding]
    representative_id: str          # 内部 citation_count 最高的 finding id
    cluster_size: int
    central_claim: str
