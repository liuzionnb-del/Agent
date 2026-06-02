"""Reader:批量读 5 篇 abstract,提炼结构化 findings(LLM 调用降 5×)。

输入:papers(已经过质量过滤)+ subquestion(用于打相关分)
输出:list[Finding],每篇 paper 一条 finding
"""
import json
import re
import sys
import uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage
from src.llm import get_llm
from src.models import Paper, Subquestion, Finding

BATCH_SIZE = 5

READER_SYS = (
    "You extract structured findings from academic abstracts to support a literature review.\n"
    "For each numbered abstract below, output ONE finding with:\n"
    "- key_point: a single concise sentence summarizing the paper's contribution relevant to the subquestion\n"
    "- quote: a direct verbatim snippet from the abstract (3-25 words) supporting the key_point\n"
    "- relevance: 0.0-1.0 — how directly the paper addresses the subquestion (be strict)\n\n"
    "If a paper is clearly off-topic give relevance < 0.3 but still produce a finding.\n"
    "Output ONLY a JSON object:\n"
    '{"findings":[{"paper_idx":0,"key_point":"...","quote":"...","relevance":0.85}, ...]}'
)


def _short_id() -> str:
    return uuid.uuid4().hex[:6]


def _read_one_batch(papers: list[Paper], subq: Subquestion) -> list[Finding]:
    batch_text = ""
    for i, p in enumerate(papers):
        batch_text += f"[{i}] {p.title} ({p.year})\nAbstract: {p.abstract}\n\n"
    user = f"Subquestion: {subq.text}\n\nPapers:\n{batch_text}"
    msg = get_llm(temperature=0.2).invoke([
        SystemMessage(content=READER_SYS),
        HumanMessage(content=user),
    ])
    m = re.search(r"\{.*\}", msg.content, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except Exception:
        return []
    findings = []
    for item in data.get("findings", []):
        idx = item.get("paper_idx")
        if not isinstance(idx, int) or idx >= len(papers):
            continue
        p = papers[idx]
        findings.append(Finding(
            id=f"F{_short_id()}",
            paper_id=p.paper_id,
            paper_title=p.title,
            paper_url=p.url,
            paper_year=p.year or 0,
            paper_citation_count=p.citation_count,
            key_point=item.get("key_point", "").strip(),
            quote=item.get("quote", "").strip(),
            relevance=float(item.get("relevance", 0.0)),
            subquestion_id=subq.id,
        ))
    return findings


def read_papers(papers: list[Paper], subq: Subquestion,
                batch_size: int = BATCH_SIZE) -> list[Finding]:
    """对 papers 分批喂给 LLM,每批 batch_size 篇。"""
    out = []
    for i in range(0, len(papers), batch_size):
        out.extend(_read_one_batch(papers[i:i + batch_size], subq))
    return out


if __name__ == "__main__":
    # 端到端冒烟:topic -> planner -> s2_search -> reader
    from src.agents.planner import plan
    from src.tools.s2_tool import s2_search

    topic = sys.argv[1] if len(sys.argv) > 1 else "retrieval augmented generation"
    print(f"Topic: {topic}\n")
    print("[1/3] Planner...")
    subqs = plan(topic)
    for sq in subqs:
        print(f"  {sq.id}: {sq.text}")
    sq0 = subqs[0]
    print(f"\n[2/3] Searcher (s2) for {sq0.id}...")
    papers = s2_search(sq0.keywords[0], max_results=5)
    print(f"  got {len(papers)} papers")
    print(f"\n[3/3] Reader batch ({len(papers)} papers, 1 LLM call)...")
    findings = read_papers(papers, sq0)
    for f in findings:
        print(f"  [{f.id}] rel={f.relevance:.2f}  cit={f.paper_citation_count}")
        print(f"        point: {f.key_point}")
        print(f"        quote: \"{f.quote[:100]}\"")
