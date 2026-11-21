"""GitHub 调研脚本的离线回归测试（不真打 API）。"""
from github_search import build_search_url, dedupe, parse_items, rank


def test_build_search_url_has_expected_params():
    url = build_search_url("novel to comic", per_page=5)
    assert url.startswith("https://api.github.com/search/repositories?")
    assert "q=novel+to+comic" in url
    assert "sort=stars" in url
    assert "order=desc" in url
    assert "per_page=5" in url


def test_build_search_url_encodes_chinese():
    url = build_search_url("小说 漫画")
    assert "%E5%B0%8F%E8%AF%B4" in url  # "小说" 的 UTF-8 百分号编码


def test_parse_items_normalizes_missing_fields():
    payload = {
        "items": [
            {
                "full_name": "a/b",
                "stargazers_count": 12,
                "language": None,
                "description": None,
                "html_url": "https://github.com/a/b",
            },
            {"full_name": "c/d", "stargazers_count": 0},
        ]
    }
    items = parse_items(payload)
    assert len(items) == 2
    assert items[0]["language"] == "-"
    assert items[0]["description"] == ""
    assert items[1]["stars"] == 0
    assert items[1]["url"] == ""


def test_parse_items_tolerates_empty_payload():
    assert parse_items({}) == []
    assert parse_items({"items": None}) == []


def test_rank_and_dedupe():
    items = [
        {"name": "a/b", "stars": 5},
        {"name": "c/d", "stars": 100},
        {"name": "a/b", "stars": 5},
    ]
    uniq = dedupe(items)
    assert len(uniq) == 2
    assert [x["name"] for x in rank(uniq)] == ["c/d", "a/b"]
