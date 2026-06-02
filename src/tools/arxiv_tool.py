"""arXiv 检索 Tool。免 key，CS/Physics/Math 主力源。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import arxiv
from src.models import Paper


def arxiv_search(query: str, max_results: int = 5) -> list[Paper]:
    client = arxiv.Client(page_size=max_results, delay_seconds=1.0, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    out = []
    for r in client.results(search):
        out.append(Paper(
            source="arxiv",
            paper_id=r.get_short_id(),
            title=r.title.strip(),
            abstract=(r.summary or "").strip(),
            year=r.published.year if r.published else None,
            citation_count=0,                    # arXiv 不暴露引用数
            venue=None,
            url=r.entry_id,
            authors=[a.name for a in r.authors],
            external_ids={"ArXivId": r.get_short_id()},
        ))
    return out


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "retrieval augmented generation"
    print(f"Query: {q}\n")
    for p in arxiv_search(q, max_results=5):
        print(f"  [{p.year}] {p.title[:90]}")
        print(f"          {p.url}")
