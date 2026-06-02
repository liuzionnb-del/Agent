"""PubMed 检索 Tool。生命医学场景必备，NCBI E-utilities，免 key。"""
import sys
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import requests
from xml.etree import ElementTree as ET
from src.models import Paper

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _esearch_ids(query: str, max_results: int) -> list[str]:
    r = requests.get(
        f"{EUTILS}/esearch.fcgi",
        params={"db": "pubmed", "term": query, "retmax": max_results,
                "retmode": "json", "sort": "relevance"},
        timeout=20,
    )
    r.raise_for_status()
    return r.json().get("esearchresult", {}).get("idlist", []) or []


def _efetch_papers(ids: list[str]) -> list[Paper]:
    if not ids:
        return []
    r = requests.get(
        f"{EUTILS}/efetch.fcgi",
        params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml"},
        timeout=30,
    )
    r.raise_for_status()
    root = ET.fromstring(r.text)
    out = []
    for art in root.findall(".//PubmedArticle"):
        pmid = (art.findtext(".//PMID") or "").strip()
        title = (art.findtext(".//ArticleTitle") or "").strip()
        # 摘要可能分多段
        abs_parts = [t.text for t in art.findall(".//AbstractText") if t.text]
        abstract = " ".join(abs_parts).strip()
        if not abstract:
            continue
        year_text = art.findtext(".//PubDate/Year") or art.findtext(".//PubDate/MedlineDate") or ""
        year_match = re.search(r"\d{4}", year_text)
        year = int(year_match.group(0)) if year_match else None
        venue = (art.findtext(".//Journal/Title") or "").strip() or None
        authors = []
        for a in art.findall(".//Author"):
            ln = a.findtext("LastName") or ""
            fn = a.findtext("ForeName") or ""
            full = f"{fn} {ln}".strip()
            if full:
                authors.append(full)
        doi = ""
        for el_id in art.findall(".//ArticleId"):
            if el_id.attrib.get("IdType") == "doi":
                doi = (el_id.text or "").strip()
                break
        out.append(Paper(
            source="pubmed",
            paper_id=pmid,
            title=title,
            abstract=abstract,
            year=year,
            citation_count=0,                # PubMed 不直接给引用数
            venue=venue,
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            authors=authors,
            external_ids={"PubMedId": pmid, **({"DOI": doi} if doi else {})},
        ))
    return out


def pubmed_search(query: str, max_results: int = 5) -> list[Paper]:
    return _efetch_papers(_esearch_ids(query, max_results))


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "CRISPR base editing"
    print(f"Query: {q}\n")
    for p in pubmed_search(q, max_results=5):
        ven = f" | {p.venue[:40]}" if p.venue else ""
        print(f"  [{p.year}]{ven}")
        print(f"    {p.title[:90]}")
        print(f"    {p.url}")
