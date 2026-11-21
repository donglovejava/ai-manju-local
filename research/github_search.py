"""GitHub 调研：搜索"小说/文本 → 漫画/漫剧/视频"相关项目。

拆成纯函数（build_search_url / parse_items / rank）+ 一个 main，
便于 tests/test_research.py 离线回归，不必每次真打 API。
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request

GITHUB_SEARCH = "https://api.github.com/search/repositories"
HEADERS = {"User-Agent": "manju-research", "Accept": "application/vnd.github+json"}

QUERIES = [
    "novel to anime",
    "novel to comic",
    "ai comic generator",
    "manga generator ai",
    "storyboard generator ai",
    "comfyui storyboard",
    "light novel video",
    "text to comic",
    "webtoon generator",
    "manhwa generator",
    "小说 漫画",
    "ai manga pipeline",
]


def build_search_url(query: str, per_page: int = 6) -> str:
    """构造按 star 降序的搜索 URL。"""
    params = urllib.parse.urlencode(
        {"q": query, "sort": "stars", "order": "desc", "per_page": per_page}
    )
    return f"{GITHUB_SEARCH}?{params}"


def parse_items(payload: dict) -> list[dict]:
    """把 GitHub 的仓库条目标准化；容忍字段缺失与空结果。"""
    out: list[dict] = []
    for it in payload.get("items", []) or []:
        out.append(
            {
                "name": it.get("full_name", ""),
                "stars": int(it.get("stargazers_count") or 0),
                "language": it.get("language") or "-",
                "description": (it.get("description") or "")[:95],
                "url": it.get("html_url", ""),
            }
        )
    return out


def rank(items: list[dict]) -> list[dict]:
    """按 star 降序。"""
    return sorted(items, key=lambda x: -x["stars"])


def dedupe(items: list[dict]) -> list[dict]:
    """按仓库名去重，保留首次出现。"""
    seen: set[str] = set()
    out: list[dict] = []
    for it in items:
        if it["name"] in seen:
            continue
        seen.add(it["name"])
        out.append(it)
    return out


def fetch(url: str, timeout: int = 25) -> dict:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> None:
    all_items: list[dict] = []
    for q in QUERIES:
        try:
            payload = fetch(build_search_url(q))
        except Exception as e:  # noqa: BLE001 - 限速/网络错误不中断整体调研
            print(f"\n## {q}  -> ERROR {e}")
            continue
        items = parse_items(payload)
        print(f"\n## {q}  ({payload.get('total_count', 0)} hits)")
        for it in items:
            print(f"  {it['stars']:>6}★  {it['name']:<45} {it['language']:<10} {it['description']}")
        all_items.extend(items)
        time.sleep(1.2)

    uniq = rank(dedupe(all_items))
    print(f"\n\n=== 去重后共 {len(uniq)} 个仓库 ===\n")
    for it in uniq[:20]:
        print(f"{it['stars']:>6}★  {it['name']:<45} {it['url']}")


if __name__ == "__main__":
    main()
