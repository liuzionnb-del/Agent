"""跨源去重 + 按 citation/年份做质量过滤。"""
from datetime import datetime
from src.config import MIN_CITATIONS, MAX_AGE_YEARS
from src.models import Paper


def dedupe_papers(papers: list[Paper]) -> list[Paper]:
    """按 dedupe_key 跨源去重；保留 citation_count 最高的。"""
    bucket: dict[str, Paper] = {}
    for p in papers:
        k = p.dedupe_key()
        if k not in bucket or p.citation_count > bucket[k].citation_count:
            bucket[k] = p
    return list(bucket.values())


def filter_papers(papers: list[Paper], *,
                  min_citations: int = MIN_CITATIONS,
                  max_age_years: int = MAX_AGE_YEARS,
                  new_paper_grace: bool = True) -> list[Paper]:
    """质量过滤：
    - 缺年份 -> 丢
    - 太老(> max_age_years) -> 丢
    - 近 1 年 + new_paper_grace -> 放行(避免新论文 citation=0 被误杀)
    - 其余按 min_citations 阈值
    arXiv / PubMed 不返回 citation_count(默认 0),会被这层硬卡掉,
    所以靠 s2_search 提供引用数,arxiv/pubmed 主要负责补充覆盖面。
    实践上调用方常分别设阈值。
    """
    current = datetime.now().year
    out = []
    for p in papers:
        if p.year is None:
            continue
        age = current - p.year
        if age > max_age_years:
            continue
        if new_paper_grace and age <= 1:
            out.append(p)
            continue
        if p.citation_count < min_citations:
            continue
        out.append(p)
    return out


if __name__ == "__main__":
    # 简单 sanity check
    from src.models import Paper
    today = datetime.now().year
    sample = [
        Paper("arxiv", "a1", "old high-cit", "", year=today - 6, citation_count=200),
        Paper("arxiv", "a2", "recent low-cit", "", year=today, citation_count=0),
        Paper("arxiv", "a3", "midage ok-cit", "", year=today - 3, citation_count=10),
        Paper("arxiv", "a4", "midage low-cit", "", year=today - 3, citation_count=1),
    ]
    kept = filter_papers(sample)
    print("kept:")
    for p in kept:
        print(f"  {p.title}")
    # 预期: 'recent low-cit'(grace 放行) + 'midage ok-cit'(过 cit 阈值)
