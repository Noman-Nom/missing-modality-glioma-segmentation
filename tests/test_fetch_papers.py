"""ADR 0003: fetch_papers.py on toy answers. No test here touches the network."""
import csv
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PATH = REPO / "experiments" / "0006-literature" / "fetch_papers.py"
SPEC = importlib.util.spec_from_file_location("fetch_papers", PATH)
fp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fp)


def work(n, url=None, cited=1, pdf=None):
    return {"id": f"https://openalex.org/W{n}", "doi": f"https://doi.org/10.1/{n}",
            "title": f"Paper  {n}", "publication_year": 2024, "cited_by_count": cited,
            "primary_location": {"source": {"display_name": "A Journal"}},
            "open_access": {"oa_url": url}, "best_oa_location": {"pdf_url": pdf}}


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_the_direct_pdf_link_comes_first():
    assert fp.pdf_link(work(1, url="https://a.org/page", pdf="https://a.org/x.pdf")).endswith("x.pdf")


def test_an_arxiv_page_becomes_its_pdf():
    assert fp.pdf_link(work(1, url="https://arxiv.org/abs/2409.19366")) == (
        "https://arxiv.org/pdf/2409.19366")


def test_a_closed_paper_is_listed_for_the_human():
    row = fp.to_row(work(1), "q")
    assert (row["id"], row["url"], row["status"]) == ("W1", "", "no_open_copy")
    assert row["title"] == "Paper 1"


def test_a_paper_found_twice_is_kept_once():
    first, new = fp.merge([], [work(1, url="https://a.org/1")], "q1")
    second, new_again = fp.merge(first, [work(1, url="https://a.org/1"), work(2)], "q2")
    assert (new, new_again) == (1, 1)
    assert [r["id"] for r in second] == ["W1", "W2"]
    assert second[0]["query"] == "q1"


def test_search_writes_candidates_and_one_log_line(tmp_path, monkeypatch):
    answer = {"meta": {"count": 7}, "results": [work(1, url="https://a.org/1"), work(2)]}
    monkeypatch.setattr(fp, "fetch", lambda url, limit=None: json.dumps(answer).encode())
    assert fp.main(["--lit", str(tmp_path), "search", "glioma"]) == 0
    assert len(rows(tmp_path / "candidates.csv")) == 2
    log = rows(tmp_path / "search-log.csv")
    assert [(x["query"], x["hits"], x["returned"], x["new"]) for x in log] == [
        ("glioma", "7", "2", "2")]


def test_the_top_are_the_most_cited_waiting_papers():
    table = [{"id": "a", "cited_by": "1", "status": "new"},
             {"id": "b", "cited_by": "9", "status": "ok"},
             {"id": "c", "cited_by": "5", "status": "new"}]
    assert [r["id"] for r in fp.pick(table, set(), 1)] == ["c"]
    assert [r["id"] for r in fp.pick(table, {"b"}, 1)] == ["b"]


def test_only_a_real_pdf_is_kept(tmp_path, monkeypatch):
    row = {"id": "W1", "url": "https://a.org/1"}
    monkeypatch.setattr(fp, "fetch", lambda url, limit=None: b"<html>sign in</html>")
    assert fp.download(row, tmp_path) == "not_pdf"
    assert not list(tmp_path.iterdir())
    monkeypatch.setattr(fp, "fetch", lambda url, limit=None: b"%PDF-1.7 toy")
    assert fp.download(row, tmp_path) == "ok"
    assert (tmp_path / "W1.pdf").read_bytes() == b"%PDF-1.7 toy"


def test_a_plain_http_link_is_not_followed(tmp_path, monkeypatch):
    def refuse(url, limit=None):
        raise AssertionError("must not fetch")
    monkeypatch.setattr(fp, "fetch", refuse)
    assert fp.download({"id": "W1", "url": "http://a.org/1"}, tmp_path) == "no_open_copy"


def test_a_failed_fetch_is_a_status_not_a_crash(tmp_path, monkeypatch):
    def broken(url, limit=None):
        raise OSError("no route")
    monkeypatch.setattr(fp, "fetch", broken)
    assert fp.download({"id": "W1", "url": "https://a.org/1"}, tmp_path) == "failed"
