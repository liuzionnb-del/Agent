"""Semantic Scholar 检索 Tool。跨学科主力，免 key（也可填 key 提速率上限）。
带 citation count / venue / externalIds，是质量过滤的关键数据源。
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import requests
from src.config import S2_API_KEY
from src.models import Paper

BASE = "https://api.semanticscholar.org/graph/v1"
FIELDS = "title,abstract,year,citationCount,venue,url,authors,externalIds"


def _headers() -> dict:
    return {"x-api-key": S2_API_KEY} if S2_API_KEY else {}


def s2_search(query: str, max_results: int = 5,
              year_from: int | None = None, min_citations: int = 0) -> list[Paper]:
    params = {"query": query, "limit": min(max_results * 2, 20), "fields": FIELDS}
    if year_from:
        params["year"] = f"{year_from}-"
    # 指数退避重试 429(无 key 共享 100 req/5min,易撞峰)
    delays = [5, 15, 45]
    for attempt in range(len(delays) + 1):
        r = requests.get(f"{BASE}/paper/search", params=params, headers=_headers(), timeout=20)
        if r.status_code != 429:
            break
        if attempt < len(delays):
            wait = int(r.headers.get("Retry-After", delays[attempt]))
            print(f"  [s2 429] sleeping {wait}s (attempt {attempt+1}/{len(delays)})", file=sys.stderr)
            time.sleep(wait)
    r.raise_for_status()
    out = []
    for item in r.json().get("data", []):
        if not item.get("abstract"):        # 摘要必备
            continue
        if (item.get("citationCount") or 0) < min_citations:
            continue
        out.append(Paper(
            source="semantic_scholar",
            paper_id=item.get("paperId", ""),
            title=(item.get("title") or "").strip(),
            abstract=(item.get("abstract") or "").strip(),
            year=item.get("year"),
            citation_count=item.get("citationCount") or 0,
            venue=item.get("venue") or None,
            url=item.get("url") or "",
            authors=[a.get("name", "") for a in (item.get("authors") or [])],
            external_ids=item.get("externalIds") or {},
        ))
        if len(out) >= max_results:
            break
    return out


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "retrieval augmented generation"
    print(f"Query: {q}\n")
    for p in s2_search(q, max_results=5):
        cit = f"cit={p.citation_count}"
        ven = f" | {p.venue}" if p.venue else ""
        print(f"  [{p.year}] {cit:<10}{ven}")
        print(f"    {p.title[:90]}")
        print(f"    {p.url}")
