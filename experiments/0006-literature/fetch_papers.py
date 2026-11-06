"""Find papers and download their open access PDFs, with a log of every search (ADR 0003).

    uv run python experiments/0006-literature/fetch_papers.py search "missing modality glioma"
    uv run python experiments/0006-literature/fetch_papers.py download --top 10
Search asks OpenAlex, adds new papers to docs/literature/candidates.csv and one line to
search-log.csv. Download takes the open access link OpenAlex gave, keeps a file only if it really
is a PDF, and writes a text copy with pdftotext. PDFs stay in ~/literature-pdfs, outside git. A
paper with no open link gets status no_open_copy, which is the list for the human to fetch. It uses
no key, no personal email and no shadow library. Standard library only.
"""
import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LIT = REPO / "docs" / "literature"
PDFS = Path.home() / "literature-pdfs"
API = "https://api.openalex.org/works"
AGENT = {"User-Agent": "qarar-research-literature/1.0"}
FIELDS = "id,doi,title,publication_year,cited_by_count,primary_location,open_access,best_oa_location"
MAX_PDF_BYTES = 60_000_000
PAUSE_S = 1.0
CANDIDATE_COLUMNS = ["id", "doi", "year", "title", "venue", "cited_by", "url", "query", "status"]
LOG_COLUMNS = ["utc", "query", "from_year", "hits", "returned", "new"]


def search_url(query, from_year, limit):
    """The OpenAlex request for one search, relevance order by default."""
    args = {"search": query, "per-page": limit, "select": FIELDS,
            "filter": f"from_publication_date:{from_year}-01-01"}
    return f"{API}?{urllib.parse.urlencode(args)}"


def pdf_link(work):
    """The best open link of an OpenAlex work, or '' if it has none. An arXiv page becomes its PDF."""
    best = work.get("best_oa_location") or {}
    url = best.get("pdf_url") or (work.get("open_access") or {}).get("oa_url") or ""
    match = re.fullmatch(r"https?://arxiv\.org/abs/(.+)", url)
    return f"https://arxiv.org/pdf/{match[1]}" if match else url


def to_row(work, query):
    """One candidates.csv row. The id is the OpenAlex one, without its address."""
    source = (work.get("primary_location") or {}).get("source") or {}
    url = pdf_link(work)
    return {"id": work["id"].rsplit("/", 1)[-1], "doi": work.get("doi") or "",
            "year": work.get("publication_year") or "", "title": " ".join((work.get("title") or "").split()),
            "venue": source.get("display_name") or "", "cited_by": work.get("cited_by_count") or 0,
            "url": url, "query": query, "status": "new" if url else "no_open_copy"}


def read_rows(path):
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def merge(rows, works, query):
    """rows plus the works not yet in it, matched on the OpenAlex id. Returns (rows, number new)."""
    seen = {r["id"] for r in rows}
    fresh = [r for r in (to_row(w, query) for w in works) if r["id"] not in seen]
    return rows + fresh, len(fresh)


def fetch(url, limit=None):
    """The bytes at url. With a limit, raises ValueError on a larger answer."""
    request = urllib.request.Request(url, headers=AGENT)
    with urllib.request.urlopen(request, timeout=60) as answer:
        data = answer.read(limit + 1) if limit else answer.read()
    if limit and len(data) > limit:
        raise ValueError(f"over {limit} bytes")
    return data


def cmd_search(a):
    answer = json.loads(fetch(search_url(a.query, a.from_year, a.limit)))
    path = a.lit / "candidates.csv"
    rows, new = merge(read_rows(path), answer["results"], a.query)
    write_rows(path, CANDIDATE_COLUMNS, rows)
    log = read_rows(a.lit / "search-log.csv") + [{
        "utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "query": a.query,
        "from_year": a.from_year, "hits": answer["meta"]["count"],
        "returned": len(answer["results"]), "new": new}]
    write_rows(a.lit / "search-log.csv", LOG_COLUMNS, log)
    print(f"{answer['meta']['count']} hits, {len(answer['results'])} returned, {new} new, "
          f"{len(rows)} candidates in {path}")
    return 0


def pick(rows, ids, top):
    """The rows to download: the ids asked for, else the top most cited not yet tried."""
    if ids:
        return [r for r in rows if r["id"] in ids]
    waiting = [r for r in rows if r["status"] == "new"]
    return sorted(waiting, key=lambda r: -int(r["cited_by"]))[:top]


def download(row, folder):
    """Try one paper and return its status: ok, not_pdf, failed or no_open_copy."""
    url = row["url"]
    if not url.startswith("https://"):
        return "no_open_copy"
    try:
        data = fetch(url, MAX_PDF_BYTES)
    except (OSError, ValueError, urllib.error.URLError):
        return "failed"
    if not data.startswith(b"%PDF"):
        return "not_pdf"
    pdf = folder / f"{row['id']}.pdf"
    pdf.write_bytes(data)
    if shutil.which("pdftotext"):
        subprocess.run(["pdftotext", "-layout", str(pdf), str(pdf.with_suffix(".txt"))],
                       check=False, capture_output=True)
    return "ok"


def cmd_download(a):
    path = a.lit / "candidates.csv"
    rows = read_rows(path)
    chosen = pick(rows, set(a.ids), a.top)
    a.pdfs.mkdir(parents=True, exist_ok=True)
    for row in chosen:
        row["status"] = download(row, a.pdfs)
        print(f"{row['status']:13s} {row['id']} {row['title'][:70]}")
        write_rows(path, CANDIDATE_COLUMNS, rows)  # a stop keeps what finished
        time.sleep(PAUSE_S)
    missing = [r for r in rows if r["status"] in ("no_open_copy", "not_pdf", "failed")]
    print(f"{sum(r['status'] == 'ok' for r in rows)} downloaded, {len(missing)} need the human")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lit", type=Path, default=LIT)
    ap.add_argument("--pdfs", type=Path, default=PDFS)
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("--from-year", type=int, default=2016)
    s.add_argument("--limit", type=int, default=50, choices=range(1, 201), metavar="1-200")
    s.set_defaults(run=cmd_search)
    d = sub.add_parser("download")
    d.add_argument("--ids", nargs="*", default=[])
    d.add_argument("--top", type=int, default=10)
    d.set_defaults(run=cmd_download)
    a = ap.parse_args(argv)
    return a.run(a)


if __name__ == "__main__":
    sys.exit(main())
