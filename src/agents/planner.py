"""Planner:把研究主题拆为 3-5 个子问题,每个带 2-3 个搜索关键词。

P2 起,Reflector 不充分时会回传 gap,Planner 用 gap 引导新一轮拆解。
"""
import json
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage
from src.llm import get_llm
from src.models import Subquestion

PLANNER_SYS = (
    "You are an academic literature research planner. "
    "Given a research topic, output 3-5 subquestions that together cover the topic's "
    "major sub-aspects (e.g., definition / mainstream methods / applications / "
    "challenges / recent trends). For each subquestion provide 2-3 short search "
    "keywords suitable for academic search engines.\n\n"
    "Output ONLY a JSON object:\n"
    '{"subquestions": ['
    '{"id":"SQ1","text":"...","keywords":["..",".."]},'
    '{"id":"SQ2",...}]}\n'
    "Do not include explanations outside the JSON."
)


def plan(topic: str, prior_gap: str = "") -> list[Subquestion]:
    user = f"Topic: {topic}"
    if prior_gap:
        user += f"\n\nPrior iteration found these gaps to address: {prior_gap}"
    msg = get_llm(temperature=0.3).invoke([
        SystemMessage(content=PLANNER_SYS),
        HumanMessage(content=user),
    ])
    m = re.search(r"\{.*\}", msg.content, re.S)
    if not m:
        raise ValueError(f"Planner output not JSON: {msg.content[:200]}")
    data = json.loads(m.group(0))
    return [Subquestion(id=sq["id"], text=sq["text"], keywords=sq["keywords"])
            for sq in data["subquestions"]]


if __name__ == "__main__":
    topic = sys.argv[1] if len(sys.argv) > 1 else "retrieval augmented generation"
    print(f"Topic: {topic}\n")
    for sq in plan(topic):
        print(f"  [{sq.id}] {sq.text}")
        print(f"        keywords: {sq.keywords}")
