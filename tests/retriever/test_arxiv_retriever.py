"""Tests for ArxivRetriever."""

import time
from types import SimpleNamespace

from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever, _run_with_hard_timeout
import zotero_arxiv_daily.retriever.arxiv_retriever as arxiv_retriever


def _sleep_and_return(value: str, delay_seconds: float) -> str:
    time.sleep(delay_seconds)
    return value


def _raise_runtime_error() -> None:
    raise RuntimeError("boom")


def test_arxiv_retriever_returns_metadata_only_when_generation_disabled(config, monkeypatch):
    raw_papers = [
        SimpleNamespace(
            title="Paper A",
            authors=[SimpleNamespace(name="Author 1"), SimpleNamespace(name="Author 2")],
            summary="Abstract A",
            entry_id="https://arxiv.org/abs/1234.5678",
            pdf_url="https://arxiv.org/pdf/1234.5678.pdf",
        ),
        SimpleNamespace(
            title="Paper B",
            authors=[SimpleNamespace(name="Author 3")],
            summary="Abstract B",
            entry_id="https://arxiv.org/abs/2345.6789",
            pdf_url="https://arxiv.org/pdf/2345.6789.pdf",
        ),
    ]

    monkeypatch.setattr(ArxivRetriever, "_retrieve_raw_papers", lambda self: raw_papers)

    config.executor.show_tldr = False
    config.executor.show_affiliations = False
    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()

    assert len(papers) == len(raw_papers)
    assert [paper.title for paper in papers] == ["Paper A", "Paper B"]
    assert papers[0].authors == ["Author 1", "Author 2"]
    assert papers[1].abstract == "Abstract B"
    assert all(paper.full_text is None for paper in papers)


def test_arxiv_retriever_include_cross_list(config, mock_feedparser, monkeypatch):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    all_entries = list(mock_feedparser.entries)
    new_entries = [
        entry for entry in all_entries
        if entry.get("arxiv_announce_type", "new") in {"new", "cross"}
    ]

    fake_results = []
    for entry in new_entries:
        pid = entry.id.removeprefix("oai:arXiv.org:")
        fake_results.append(SimpleNamespace(
            title=entry.title,
            authors=[SimpleNamespace(name="Test Author")],
            summary="Test abstract",
            pdf_url=f"https://arxiv.org/pdf/{pid}",
            entry_id=f"https://arxiv.org/abs/{pid}",
            source_url=lambda pid=pid: f"https://arxiv.org/e-print/{pid}",
        ))

    class FakeClient:
        def __init__(self, **kw):
            pass

        def results(self, search):
            return iter(fake_results)

    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)

    config.source.arxiv.include_cross_list = True
    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()

    assert len(papers) == len(new_entries)
    assert set(p.title for p in papers) == set(e.title for e in new_entries)


def test_run_with_hard_timeout_returns_value():
    result = _run_with_hard_timeout(
        _sleep_and_return,
        ("done", 0.01),
        timeout=1,
        operation="test op",
        paper_title="paper",
    )
    assert result == "done"


def test_run_with_hard_timeout_returns_none_on_timeout(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _sleep_and_return,
        ("done", 1.0),
        timeout=0.01,
        operation="test op",
        paper_title="paper",
    )
    assert result is None
    assert "timed out" in warnings[0]


def test_run_with_hard_timeout_returns_none_on_failure(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _raise_runtime_error,
        (),
        timeout=1,
        operation="test op",
        paper_title="paper",
    )
    assert result is None
    assert "boom" in warnings[0]
