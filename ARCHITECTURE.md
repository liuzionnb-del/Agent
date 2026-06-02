# 学术文献调研 Agent · 技术架构文档 v2

> 状态:**v2 已审核通过**(整合架构评审的 8 项反馈)。可进入 P1 实施。
> 对应简历:Project A · 多 Agent + 引用网络 + Reflection + 多源 Tool Calling + Memory

---

## 1. 项目定位

**输入**:一个研究主题(例如 `"in-context learning in LLMs"`)
**输出**:带引用的结构化文献综述(背景 / 主流方法 / 挑战 / 趋势 + 引用列表)
**目标用户**:**英文学术圈** AI / CS / 医学 / 生命科学研究生 / 研究员
**作用域边界(明确划界)**:
- ✅ 覆盖:arXiv(CS/Math/Physics)、Semantic Scholar(全学科)、PubMed(生命医学)
- ❌ **不做** CNKI / 万方:都需付费 API,personal project 不烧;爬虫法律灰区,不进作品集。**Tool 系统留接口,未来可接**

**自我相关性**:你是 AI 研究生,这就是为你自己课题前期调研做的工具——面试可讲"我是真用户"。

---

## 2. 整体架构(8 个 Node + 多 Tool)

```
                          [Topic]
                            │
              ┌─────────────┘
              ▼
          ┌─Planner─┐ ◄────── ReflectionReport(迭代时)
          │ 主题分解 │
          └────┬────┘
            subqs[]
              │
              ▼
        ┌──Searcher──┐  Tools(质量过滤后返回):
        │ 多源关键词搜  │   · arxiv_search
        │             │   · semantic_scholar_search  ← 主力(跨学科)
        │             │   · pubmed_search
        └──────┬──────┘
            papers[]  (已按 citations / year 过滤)
              │
              ▼
        ┌──Reader──┐   ※ 批量:每次塞 5 篇 abstract
        │ 提炼+评分  │
        └────┬─────┘
          findings[]
              │
              ▼
       ┌─Clusterer─┐   ※ bge 向量聚类去重(无 LLM)
       │ 合并同质点  │
       └────┬──────┘
          clusters[]
              │
              ▼                                  ┌─── (P2 加)
       ┌─CitationExplorer─┐  Tools: ─────────────┤  · s2_references
       │ 对 top-k cluster  │                     │  · s2_citations
       │ 追前后引用链      │                     └─── 把扩出来的 paper
       └────────┬─────────┘                          回 Reader → Clusterer
             扩展 findings
                │
                ▼
       ┌──Reflector──┐  输出结构化 ReflectionReport:
       │ 诊断覆盖/冲突 │  { covered_subqs, missing_subqs,
       │ 决定下一步    │    conflicts, evidence_quality,
       └──┬──────────┘    suggested_actions }
       充分 │  不充分 & iter<3
            │     └──→ 回 Planner(带 gap)
            ▼
   ┌─── Synthesizer(并行 4 个 LLM 调用) ───┐
   │  · Background      · Methods           │
   │  · Challenges      · Trends            │
   └──────────────────┬────────────────────┘
                       ▼
                ┌── Merger ──┐
                │ 拼接 + 统一引用编号 │
                └──────┬──────┘
                       ▼
                [Final Report]
```

**核心范式**:Plan → Multi-source Search → Read(batch)→ Cluster → CitationExplore → Reflect → Section-Synthesize → Merge

---

## 3. Node 职责定义

| Node | 输入 | 处理 | 输出 | LLM 调用 |
|------|------|------|------|---------|
| **Planner** | topic + ReflectionReport(迭代时) | 拆主题为 3-5 子问题及关键词 | `subqs[]` | 1 次 |
| **Searcher** | subqs | 对每个 subq 跑 3 个 Tool,合并去重,按质量过滤 | `papers[]` | 0(纯 Tool) |
| **Reader** | papers + subqs | **批量 5 篇/次**,结构化输出 findings 数组 | `findings[]` | ⌈N/5⌉ 次 |
| **Clusterer** | findings | bge 嵌入 → cosine ≥ 0.85 合并为 cluster | `clusters[]` | 0(纯向量) |
| **CitationExplorer**(P2) | top-k clusters | 对最强 cluster 的种子 paper 调 s2 引用网,补 papers | 扩展 papers | 0(纯 Tool) |
| **Reflector** | clusters + topic + iter | 输出结构化诊断报告 | `ReflectionReport` | 1 次 |
| **Synthesizer** | clusters | **4 段并行**生成(Background/Methods/Challenges/Trends) | `4 段 markdown` | 4 次 |
| **Merger** | 4 段 + clusters | 衔接 + 统一引用编号 + 格式化 | `final_report` | 1 次 |

**单次完整运行 LLM 调用估算**(不迭代):`1 + 0 + ⌈20/5⌉ + 0 + 1 + 4 + 1 ≈ 11 次`(原方案 20-30 次)

---

## 4. 数据流与状态

```python
class ResearchState(TypedDict):
    topic: str
    subquestions: list[Subquestion]
    findings: list[Finding]              # 累积
    clusters: list[Cluster]              # Clusterer 输出
    iteration: int
    max_iterations: int                  # = 3
    reflection_history: list[ReflectionReport]
    sections: dict[str, str]             # {"Background": "...", ...}
    final_report: str | None
```

```python
class Finding:
    id: str                  # F001...
    paper_id: str            # arXiv ID / S2 ID(供 CitationExplorer 用)
    paper_title: str
    paper_url: str
    paper_year: int
    paper_citation_count: int
    paper_venue: str | None
    key_point: str           # Reader 提炼
    quote: str               # 原文片段
    relevance: float         # 0-1
    subquestion_id: str

class Cluster:
    id: str                  # C001...
    findings: list[Finding]  # 同质点合并后的集合
    representative_finding: Finding  # 引用计数最高的代表
    cluster_size: int        # = len(findings),作"独立证据数"
    central_claim: str       # Clusterer 抽出的共同论点

class ReflectionReport:
    iteration: int
    covered_subqs: dict[str, int]        # {subq_id: cluster_count}
    missing_subqs: list[str]
    conflicts: list[Conflict]             # 不同 cluster 矛盾论点
    evidence_quality: Literal["low","mid","high"]
    decision: Literal["continue","iterate","abort"]
    suggested_actions: list[Action]      # ["search:'X'", "expand_citations:C012", ...]
```

---

## 5. Tool 定义

| Tool | 入参 | 出参 | 阶段 | 备注 |
|------|------|------|------|------|
| `arxiv_search` | `query, max_results=5` | `list[Paper]` | P1 | CS/Physics/Math,免 key |
| `semantic_scholar_search` | `query, max_results=5, year_from, min_citations` | `list[Paper]` | **P1 主力** | 跨学科,免 key,带 citations / venue |
| `pubmed_search` | `query, max_results=5` | `list[Paper]` | P1 | 医学/生命科学 |
| `s2_references` | `paper_id` | `list[Paper]` | **P2(新)** | 这篇论文引了谁(向后) |
| `s2_citations` | `paper_id, max=20` | `list[Paper]` | **P2(新)** | 谁引了这篇(向前) |
| `paper_section_extract` | `pdf_url, sections=["intro","related","conclusion"]` | `dict[str, str]` | **P2(新)** | arXiv PDF 拉下来分段抽取 |
| `cnki_search` *(预留)* | — | — | 未来 | **接口预留,v1 不实现** |

**质量过滤(Searcher 内嵌,P1 必须)**:
```python
class QualityFilter:
    min_citations: int = 3         # 老论文 (>2 年) citation 下限
    max_age_years: int = 5         # 默认只看近 5 年
    new_paper_grace: bool = True   # 近 1 年内 citation = 0 也保留
    venues_tier1: set[str] = {...} # 可选硬过滤
```

---

## 6. 记忆系统

**短期(会话内,P1)**:
- 全部存在 `ResearchState`
- findings / clusters / reflection_history 累积
- iteration 防死循环

**长期(跨任务,P5 选做)**:
- 调研完成后把 clusters + report 入 FAISS(沿用 RAG 项目的 embedding)
- 新主题进来时先检索,相似度 > 0.85 直接复用旧 clusters,可减半数 LLM 调用
- 面试讲点:"记忆级联跨会话复用"

---

## 7. Reflector 结构化决策

不再 continue/iterate/abort 三选一,而是先**诊断**再**给具体动作**:

| 诊断维度 | 数据来源 | 阈值 |
|---------|---------|------|
| 子问题覆盖 | `covered_subqs` | 至少 N-1 子问题有 ≥2 cluster |
| 证据质量 | 平均 citation count + relevance | mid 以上才走 Synthesizer |
| 冲突识别 | cluster 之间论点对比 | 标注但不阻塞 |
| 迭代上限 | iter < 3 | 硬约束 |

**iterate 时的 suggested_actions**(消除震荡):
- `"search: <new keyword>"` → Searcher
- `"expand_citations: <cluster_id>"` → CitationExplorer
- `"fetch_fulltext: <paper_id>"` → paper_section_extract(P2)

Planner 收到 actions 后**直接执行**,不再从主题重拆。

---

## 8. 技术栈

| 层 | 选型 | 理由 |
|----|------|------|
| Agent 编排 | **LangGraph** | 状态机显式,条件分支可视化 |
| LLM | DeepSeek / 通义 / 智谱(OpenAI 兼容) | 沿用 RAG 项目 |
| 检索 Tool | `arxiv`(pip 包)+ `requests` 直调 S2/PubMed REST | 都是免 key REST API |
| 引用网络 Tool | S2 `/paper/{id}/references`、`/citations` | 免费,官方 |
| PDF 抽取(P2) | `pypdf` + `unstructured` | 解决 intro/related/conclusion 分段 |
| Embedding(聚类用) | `bge-small-en-v1.5`(本地) | **复用 RAG 项目的实现** |
| 状态类型 | `TypedDict` + `pydantic` | 类型安全 |
| API 服务 | FastAPI + **SSE 流式** | 长任务必须流式 |
| 持久化(P5 选) | FAISS | 复用 RAG 项目 |
| 评估 | LLM-as-a-Judge | 复用 RAG 项目 |

---

## 9. 评估方案

**评测集**:12 题研究主题
- 4 题 CS(覆盖 LLM/RAG/Agent/视觉)
- 4 题生命科学(覆盖蛋白质/基因/神经)
- 2 题物理 / 数学
- 2 题跨学科(交叉课题)

**对比 baseline**:LLM 不调研、不调用 Tool,直接生成同一主题的"综述"

**5 维度 LLM-as-a-Judge(0-1)**:
| 维度 | 定义 |
|------|------|
| 引用密度 | 每 100 字含多少有效引用 |
| 覆盖度 | 子方向是否覆盖(对照评测集预定义的 must-cover 点) |
| 可追溯性 | 每论断是否能追到具体 finding/paper |
| **证据多样性** | 不同 cluster 占比(防 5 篇说同一事) |
| 冗余度 | 报告是否被低质量 finding 注水 |
| **运行时长**(独立指标) | topic → report 端到端,目标 < 5 分钟 |

预期:相比 baseline,引用密度 +5×、覆盖度 +50%、可追溯性 ≥ 0.85。

---

## 10. 阶段计划(改后)

| 阶段 | 内容 | 工期 | 主要 LLM 调用源 |
|------|------|------|----------------|
| **P1** 多源骨架 | Planner / Searcher(arXiv+S2+PubMed)/ Reader 批量 / 质量过滤 / Clusterer / 单段 Synthesizer / CLI | **5-6 天** | Reader, Planner, Synthesizer |
| **P2** 引用网 + Reflection + PDF | LangGraph 状态机 / Reflector 结构化输出 / CitationExplorer Tool / paper_section_extract / 迭代循环 | **4-5 天** | + Reflector, Reader v2 |
| **P3** 分段 Synth + 评估 | Synthesizer 4 段并行 + Merger / 12 题评测集 + 5 维 LLM-as-a-Judge + baseline 对比 | **3-4 天** | + Synthesizer ×4, Judge |
| **P4** 生产化 | FastAPI + SSE 流式 + README + 架构图 + GitHub | **1-2 天** | — |
| **P5** 长期记忆(选) | FAISS 历史 cluster 复用 + 相似主题快速回放 | **1-2 天** | — |

**总工期**:13-17 天。每阶段都能独立交付一个可演示版本(P1 完就有命令行 demo)。

---

## 11. 风险与应对

| 风险 | 等级 | 应对 |
|------|------|------|
| S2 API 速率限制(无 key 时 100 req/5 min) | 中 | 节流 + 失败重试 + 请求批量 |
| 长任务超时(可能 5-8 分钟) | **高** | **SSE 流式**,每个 node 完成都推进度 |
| LLM token 成本 | 中 | DeepSeek + Reader 批量(降 5×)+ Reflector 短输出 |
| Reflection 死循环 | 中 | `max_iterations=3` 硬上限 + Reflector 强制 abort 路径 |
| 评估主观性 | 中 | 评测集主题固定,must-cover 点预定义,Judge prompt 严格 |
| PDF 解析失败(P2) | 中 | 失败自动回退 abstract,不阻塞主流程 |
| 聚类阈值不准 | 低 | 0.85 起步,评测时调,可学到的超参 |

---

## 12. 复用 RAG 项目的资产

- LLM 客户端封装(`llm.py`)
- bge embedding 封装(用于 Clusterer)
- `.env` 配置体系
- LLM-as-a-Judge 评估框架
- FastAPI + lifespan 模板
- 项目目录结构与 git 流程
- gold set 设计思路(分层 + tuning/test 切分)

**新建能力**(差异化):
- LangGraph 多 Agent 状态机
- 真实跨域 Tool Calling(3 源检索 + 引用网)
- Reflection 结构化自我修正
- 向量聚类去重(用于"覆盖度"指标)
- 4 段并行生成 + Merger
- SSE 流式长任务输出
- (可选)跨会话长期记忆

---

## 13. 简历升级版描述(假设 P1-P4 完成)

> **跨学科学术文献调研 Agent** ｜ github.com/你/research-agent
>
> **项目简介**:面向研究生/研究员的英文学术文献自动调研系统,输入主题自动产出带引用的结构化综述,覆盖 arXiv / Semantic Scholar / PubMed 三源。
>
> **技术栈**:Python、LangGraph、FastAPI(SSE)、Semantic Scholar / arXiv / PubMed API、bge 嵌入、LLM-as-a-Judge
>
> **技术亮点**:
> - **多 Agent 状态机**:基于 LangGraph 编排 Planner→Searcher→Reader→Clusterer→Reflector→Synthesizer→Merger 八节点流水线,显式状态可观测,iteration 硬上限防死循环。
> - **多源检索 + 引用网络**:3 个学术源 Tool + 双向引用追溯(Semantic Scholar references/citations),按 citation/年份/会议级别质量过滤。
> - **批量阅读 + 向量聚类去重**:Reader 批量 5 篇/次降 5 倍调用成本;后接 bge 嵌入聚类合并同质 finding,避免"5 篇讲同一事"的虚高覆盖度。
> - **结构化反思自修正**:Reflector 输出 missing_subqs/conflicts/suggested_actions,Planner 按 actions 精准迭代,消除阈值震荡。
> - **分段并行生成**:综述拆为背景/方法/挑战/趋势 4 段并行 LLM 调用,Merger 统一引用编号,长综述不超 token。
> - **评估体系**:12 题跨学科评测集,LLM-as-a-Judge 5 维度评分,baseline 对照下引用密度 +5×、覆盖度 +50%。
> - **生产化**:FastAPI + SSE 流式接口,长任务每 node 推进度,避免 HTTP 超时。

---

**v2 架构确认。开 P1 实施。**
