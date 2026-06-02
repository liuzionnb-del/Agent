"""Clusterer:bge 向量 + cosine 阈值贪心聚类,合并同质 findings(无 LLM)。

为什么聚类:5 篇论文如果都讲"RAG 检索阶段用 dense embedding 提升召回",
原始 finding 数会虚高,真实"独立证据"只有 1 条。聚类后用 cluster_size 作覆盖度。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from src.embeddings import BgeEmbeddings
from src.models import Finding, Cluster


def cluster_findings(findings: list[Finding], threshold: float = 0.85) -> list[Cluster]:
    if not findings:
        return []
    e = BgeEmbeddings()
    vecs = np.asarray(e.encode_documents([f.key_point for f in findings]))
    sims = vecs @ vecs.T                                    # 已归一化,点积 = cosine
    n = len(findings)
    visited: set[int] = set()
    clusters: list[Cluster] = []

    for i in range(n):
        if i in visited:
            continue
        members_idx = [i]
        visited.add(i)
        for j in range(i + 1, n):
            if j not in visited and sims[i][j] >= threshold:
                members_idx.append(j)
                visited.add(j)
        members = [findings[k] for k in members_idx]
        rep = max(members, key=lambda f: f.paper_citation_count)
        clusters.append(Cluster(
            id=f"C{len(clusters) + 1:03d}",
            findings=members,
            representative_id=rep.id,
            cluster_size=len(members),
            central_claim=rep.key_point,        # P1 简单取代表;P3 可让 LLM 综合
        ))
    return clusters


if __name__ == "__main__":
    # 构造几条手动 findings 做 sanity check
    samples = [
        Finding("F1", "p1", "Paper1", "u", 2023, 100,
                "RAG uses dense retrieval to find relevant passages for LLM context", "...", 0.9, "SQ1"),
        Finding("F2", "p2", "Paper2", "u", 2023, 50,
                "Retrieval augmented generation retrieves documents via dense embeddings to ground LLM responses", "...", 0.9, "SQ1"),
        Finding("F3", "p3", "Paper3", "u", 2023, 30,
                "Sparse BM25 keyword retrieval can complement dense methods", "...", 0.8, "SQ2"),
        Finding("F4", "p4", "Paper4", "u", 2024, 5,
                "Cross-encoder reranking improves retrieval precision after initial recall", "...", 0.85, "SQ2"),
    ]
    cs = cluster_findings(samples, threshold=0.85)
    for c in cs:
        members_ids = [f.id for f in c.findings]
        print(f"  {c.id} size={c.cluster_size}  members={members_ids}")
        print(f"      rep claim: {c.central_claim[:90]}")
