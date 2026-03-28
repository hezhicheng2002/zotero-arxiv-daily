import io
import time
from types import SimpleNamespace
from urllib.error import HTTPError

from omegaconf import open_dict

from zotero_arxiv_daily.protocol import Paper
import zotero_arxiv_daily.retriever.arxiv_retriever as arxiv_retriever
from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever, _run_with_hard_timeout
from zotero_arxiv_daily.retriever.base import BaseRetriever, register_retriever


def _sleep_and_return(value: str, delay_seconds: float) -> str:
    time.sleep(delay_seconds)
    return value


def _raise_runtime_error() -> None:
    raise RuntimeError("boom")


def test_arxiv_retriever(config, monkeypatch):
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

    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()

    assert len(papers) == len(raw_papers)
    assert [paper.title for paper in papers] == ["Paper A", "Paper B"]
    assert papers[0].authors == ["Author 1", "Author 2"]
    assert papers[1].abstract == "Abstract B"
    assert all(paper.full_text is None for paper in papers)


@register_retriever("failing_test")
class FailingTestRetriever(BaseRetriever):
    def _retrieve_raw_papers(self) -> list[dict[str, str]]:
        return [
            {"title": "good paper", "mode": "ok"},
            {"title": "bad paper", "mode": "fail"},
        ]

    def convert_to_paper(self, raw_paper: dict[str, str]) -> Paper | None:
        if raw_paper["mode"] == "fail":
            raise HTTPError(
                url="https://example.com/paper.pdf",
                code=404,
                msg="not found",
                hdrs=None,
                fp=io.BufferedReader(io.BytesIO(b"missing")),
            )
        return Paper(
            source=self.name,
            title=raw_paper["title"],
            authors=[],
            abstract="",
            url=f"https://example.com/{raw_paper['mode']}",
        )


@register_retriever("serial_test")
class SerialTestRetriever(BaseRetriever):
    def __init__(self, config, seen_titles: list[str]):
        super().__init__(config)
        self.seen_titles = seen_titles

    def _retrieve_raw_papers(self) -> list[dict[str, str]]:
        return [
            {"title": "paper 1"},
            {"title": "paper 2"},
            {"title": "paper 3"},
        ]

    def convert_to_paper(self, raw_paper: dict[str, str]) -> Paper:
        self.seen_titles.append(raw_paper["title"])
        return Paper(
            source=self.name,
            title=raw_paper["title"],
            authors=[],
            abstract="",
            url=f"https://example.com/{raw_paper['title']}",
        )


def test_retrieve_papers_skips_conversion_errors(config):
    with open_dict(config.source):
        config.source.failing_test = {}

    retriever = FailingTestRetriever(config)
    papers = retriever.retrieve_papers()

    assert [paper.title for paper in papers] == ["good paper"]


def test_retrieve_papers_runs_serially(config):
    with open_dict(config.source):
        config.source.serial_test = {}

    seen_titles: list[str] = []
    retriever = SerialTestRetriever(config, seen_titles)
    papers = retriever.retrieve_papers()

    expected_titles = ["paper 1", "paper 2", "paper 3"]
    assert seen_titles == expected_titles
    assert [paper.title for paper in papers] == expected_titles


def test_run_with_hard_timeout_returns_value():
    result = _run_with_hard_timeout(
        _sleep_and_return,
        ("done", 0.01),
        timeout=1,
        operation="test operation",
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
        operation="test operation",
        paper_title="paper",
    )

    assert result is None
    assert warnings == ["test operation timed out for paper after 0.01 seconds"]


def test_run_with_hard_timeout_returns_none_on_failure(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))

    result = _run_with_hard_timeout(
        _raise_runtime_error,
        (),
        timeout=1,
        operation="test operation",
        paper_title="paper",
    )

    assert result is None
    assert warnings == ["test operation failed for paper: RuntimeError: boom"]
