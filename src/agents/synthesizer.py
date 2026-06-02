"""Synthesizer(P1 单段版):由 clusters 一次生成 4 段综述。

P3 升级为 4 段并行 + Merger。P1 先用单 LLM 调用串起来,验证可读性。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage
from src.llm import get_llm
from src.models import Cluster

SYNTH_SYS = (
    "You are writing a concise literature review based ONLY on the findings provided. "
    "Output well-formatted Markdown with EXACTLY these 4 sections (h2 headings):\n"
    "  ## Background\n  ## Main Methods\n  ## Challenges\n  ## Trends\n\n"
    "Hard constraints:\n"
    "- Every non-trivial claim MUST cite cluster IDs in brackets like [C001] or [C001, C003].\n"
    "- Each section must contain at least 2 citations.\n"
    "- Use ONLY information in the findings — do NOT add outside knowledge.\n"
    "- If evidence is thin for a section, say so briefly rather than fabricate.\n"
    "- Total length: 500-800 words.\n"
)


def synthesize(topic: str, clusters: list[Cluster]) -> str:
    if not clusters:
        return "No findings collected; report skipped."

    # 把 cluster 压成上下文:每条 1 行,带 cluster_size 作"独立证据数"信号
    ctx_lines = []
    for c in clusters:
        finding_ids = ", ".join(f.id for f in c.findings)
        year_range = sorted({f.paper_year for f in c.findings if f.paper_year})
        max_cit = max((f.paper_citation_count for f in c.findings), default=0)
        meta = f"size={c.cluster_size}, years={year_range}, max_citations={max_cit}, findings={finding_ids}"
        ctx_lines.append(f"[{c.id} | {meta}]\n  claim: {c.central_claim}")
    ctx = "\n\n".join(ctx_lines)

    user = f"Topic: {topic}\n\nClusters (use [CXXX] for citations):\n{ctx}"
    msg = get_llm(temperature=0.3).invoke([
        SystemMessage(content=SYNTH_SYS),
        HumanMessage(content=user),
    ])
    return msg.content.strip()


if __name__ == "__main__":
    # 单元测试:用手造 clusters 调一次
    from src.models import Finding
    fs = [
        Finding("F1", "p1", "Lewis 2020", "u", 2020, 14000,
                "RAG combines neural retrieval over dense indices with a seq2seq generator", "...", 0.95, "SQ1"),
        Finding("F2", "p2", "Survey 2023", "u", 2023, 3000,
                "Surveys recent advances in retrieval augmented generation for LLMs", "...", 0.9, "SQ5"),
        Finding("F3", "p3", "RAGAs 2023", "u", 2023, 600,
                "Provides reference-free metrics for evaluating RAG faithfulness and relevance", "...", 0.85, "SQ4"),
    ]
    cs = [
        Cluster("C001", [fs[0]], fs[0].id, 1, fs[0].key_point),
        Cluster("C002", [fs[1]], fs[1].id, 1, fs[1].key_point),
        Cluster("C003", [fs[2]], fs[2].id, 1, fs[2].key_point),
    ]
    print(synthesize(sys.argv[1] if len(sys.argv) > 1 else "Retrieval Augmented Generation", cs))
