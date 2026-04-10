"""Tests for zotero_arxiv_daily.executor: normalize_path_patterns, filter_corpus, fetch_zotero_corpus, E2E."""

from datetime import datetime

import pytest
from omegaconf import OmegaConf

from zotero_arxiv_daily.executor import Executor, normalize_path_patterns
from zotero_arxiv_daily.protocol import CorpusPaper, Paper


def test_normalize_path_patterns_rejects_single_string_for_include_path():
    with pytest.raises(TypeError, match="config.zotero.include_path must be a list"):
        normalize_path_patterns("2026/survey/**", "include_path")


def test_normalize_path_patterns_accepts_list_config_for_include_path():
    include_path = OmegaConf.create(["2026/survey/**", "2026/reading-group/**"])
    assert normalize_path_patterns(include_path, "include_path") == [
        "2026/survey/**",
        "2026/reading-group/**",
    ]


def test_normalize_path_patterns_rejects_single_string_for_ignore_path():
    with pytest.raises(TypeError, match="config.zotero.ignore_path must be a list"):
        normalize_path_patterns("archive/**", "ignore_path")


def test_normalize_path_patterns_accepts_list_config_for_ignore_path():
    ignore_path = OmegaConf.create(["archive/**", "2025/**"])
    assert normalize_path_patterns(ignore_path, "ignore_path") == ["archive/**", "2025/**"]


def test_normalize_path_patterns_accepts_empty_list():
    assert normalize_path_patterns([], "ignore_path") == []


def test_normalize_path_patterns_accepts_none():
    assert normalize_path_patterns(None, "include_path") is None


def _make_executor(include_patterns=None, ignore_patterns=None):
    executor = Executor.__new__(Executor)
    executor.include_path_patterns = normalize_path_patterns(include_patterns, "include_path") if include_patterns else None
    executor.ignore_path_patterns = normalize_path_patterns(ignore_patterns, "ignore_path") if ignore_patterns else None
    return executor


def test_filter_corpus_matches_any_path_against_any_pattern():
    executor = _make_executor(include_patterns=["2026/survey/**", "2026/reading-group/**"])
    corpus = [
        CorpusPaper(title="Survey Paper", abstract="", added_date=datetime(2026, 1, 1), paths=["2026/survey/topic-a", "archive/misc"]),
        CorpusPaper(title="Reading Group Paper", abstract="", added_date=datetime(2026, 1, 2), paths=["notes/inbox", "2026/reading-group/week-1"]),
        CorpusPaper(title="Excluded Paper", abstract="", added_date=datetime(2026, 1, 3), paths=["2025/other/topic"]),
    ]
    filtered = executor.filter_corpus(corpus)
    assert [p.title for p in filtered] == ["Survey Paper", "Reading Group Paper"]


def test_filter_corpus_excludes_papers_matching_ignore_path():
    executor = _make_executor(ignore_patterns=["archive/**", "2025/**"])
    corpus = [
        CorpusPaper(title="Active Paper", abstract="", added_date=datetime(2026, 1, 1), paths=["2026/survey/topic-a"]),
        CorpusPaper(title="Archived Paper", abstract="", added_date=datetime(2026, 1, 2), paths=["archive/misc"]),
        CorpusPaper(title="Old Paper", abstract="", added_date=datetime(2026, 1, 3), paths=["2025/other/topic"]),
    ]
    filtered = executor.filter_corpus(corpus)
    assert [p.title for p in filtered] == ["Active Paper"]


def test_filter_corpus_ignore_path_takes_precedence_over_include_path():
    executor = _make_executor(include_patterns=["2026/**"], ignore_patterns=["2026/ignore/**"])
    corpus = [
        CorpusPaper(title="Included Paper", abstract="", added_date=datetime(2026, 1, 1), paths=["2026/survey/topic-a"]),
        CorpusPaper(title="Ignored Paper", abstract="", added_date=datetime(2026, 1, 2), paths=["2026/ignore/topic-b"]),
    ]
    filtered = executor.filter_corpus(corpus)
    assert [p.title for p in filtered] == ["Included Paper"]


def test_filter_corpus_no_filters_returns_all():
    executor = _make_executor()
    corpus = [
        CorpusPaper(title="Paper A", abstract="", added_date=datetime(2026, 1, 1), paths=["foo"]),
        CorpusPaper(title="Paper B", abstract="", added_date=datetime(2026, 1, 2), paths=["bar"]),
    ]
    filtered = executor.filter_corpus(corpus)
    assert filtered == corpus


def test_fetch_zotero_corpus(config, monkeypatch):
    from tests.canned_responses import make_stub_zotero_client

    stub_zot = make_stub_zotero_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    executor = Executor.__new__(Executor)
    executor.config = config
    corpus = executor.fetch_zotero_corpus()

    assert len(corpus) == 2
    assert corpus[0].title == "Stub Paper 1"
    assert "survey/topic-a" in corpus[0].paths[0]


def test_fetch_zotero_corpus_paper_with_zero_collections(config, monkeypatch):
    from tests.canned_responses import make_stub_zotero_client

    items = [
        {
            "data": {
                "title": "No Collection Paper",
                "abstractNote": "Abstract.",
                "dateAdded": "2026-03-01T00:00:00Z",
                "collections": [],
            }
        }
    ]
    stub_zot = make_stub_zotero_client(items=items)
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    executor = Executor.__new__(Executor)
    executor.config = config
    corpus = executor.fetch_zotero_corpus()

    assert len(corpus) == 1
    assert corpus[0].paths == []


def test_run_end_to_end(config, monkeypatch):
    import smtplib

    from omegaconf import open_dict

    from tests.canned_responses import make_stub_openai_client, make_stub_smtp, make_stub_zotero_client, make_sample_paper

    with open_dict(config):
        config.executor.source = ["arxiv"]
        config.executor.reranker = "api"
        config.executor.send_empty = False

    stub_zot = make_stub_zotero_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    stub_client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: stub_client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)
    retrieved = [
        make_sample_paper(title="E2E Paper 1", score=None),
        make_sample_paper(title="E2E Paper 2", score=None),
    ]

    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401
    from zotero_arxiv_daily.retriever.base import registered_retrievers

    monkeypatch.setattr(registered_retrievers["arxiv"], "retrieve_papers", lambda self: retrieved)

    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    executor = Executor(config)
    executor.run()

    assert len(sent) == 1
    _, _, email_body = sent[0]
    assert "text/html" in email_body


def test_run_no_papers_send_empty_false(config, monkeypatch):
    import smtplib

    from omegaconf import open_dict

    from tests.canned_responses import make_stub_openai_client, make_stub_smtp, make_stub_zotero_client

    with open_dict(config):
        config.executor.source = ["arxiv"]
        config.executor.reranker = "api"
        config.executor.send_empty = False

    stub_zot = make_stub_zotero_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    stub_client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: stub_client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)

    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401
    from zotero_arxiv_daily.retriever.base import registered_retrievers

    monkeypatch.setattr(registered_retrievers["arxiv"], "retrieve_papers", lambda self: [])

    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    executor = Executor(config)
    executor.run()

    assert len(sent) == 0


def test_run_no_papers_send_empty_true(config, monkeypatch):
    import smtplib

    from omegaconf import open_dict

    from tests.canned_responses import make_stub_openai_client, make_stub_smtp, make_stub_zotero_client

    with open_dict(config):
        config.executor.source = ["arxiv"]
        config.executor.reranker = "api"
        config.executor.send_empty = True

    stub_zot = make_stub_zotero_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    stub_client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: stub_client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)

    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401
    from zotero_arxiv_daily.retriever.base import registered_retrievers

    monkeypatch.setattr(registered_retrievers["arxiv"], "retrieve_papers", lambda self: [])

    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    executor = Executor(config)
    executor.run()

    assert len(sent) == 1
    _, _, body = sent[0]
    assert "text/html" in body


def test_executor_only_enriches_top_ranked_papers(config, monkeypatch):
    config.executor.source = ["arxiv"]
    config.executor.show_tldr = True
    config.executor.show_affiliations = False
    config.executor.max_paper_num = 1

    class FakeRetriever:
        name = "arxiv"

        def __init__(self, config):
            self.enriched_titles = []

        def retrieve_papers(self):
            return [
                Paper(
                    source="arxiv",
                    title="Top paper",
                    authors=["Author A"],
                    abstract="Top abstract",
                    url="https://example.com/top",
                    pdf_url="https://example.com/top.pdf",
                ),
                Paper(
                    source="arxiv",
                    title="Lower paper",
                    authors=["Author B"],
                    abstract="Lower abstract",
                    url="https://example.com/lower",
                    pdf_url="https://example.com/lower.pdf",
                ),
            ]

        def enrich_papers(self, papers):
            self.enriched_titles = [paper.title for paper in papers]
            for paper in papers:
                paper.full_text = f"full text for {paper.title}"
            return papers

    class FakeReranker:
        def __init__(self, config):
            pass

        def rerank(self, candidates, corpus):
            return list(candidates)

    monkeypatch.setattr("zotero_arxiv_daily.executor.get_retriever_cls", lambda _: FakeRetriever)
    monkeypatch.setattr("zotero_arxiv_daily.executor.get_reranker_cls", lambda _: FakeReranker)
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        Executor,
        "fetch_zotero_corpus",
        lambda self: [
            CorpusPaper(
                title="Corpus paper",
                abstract="Corpus abstract",
                added_date=datetime(2026, 3, 10),
                paths=["/test"],
            )
        ],
    )
    monkeypatch.setattr(Executor, "filter_corpus", lambda self, corpus: corpus)
    monkeypatch.setattr("zotero_arxiv_daily.executor.render_email", lambda papers, **kwargs: "email")
    monkeypatch.setattr("zotero_arxiv_daily.executor.send_email", lambda config, content: None)

    generated_titles = []

    def fake_generate_tldr(self, *args, **kwargs):
        generated_titles.append(self.title)
        self.tldr = self.abstract
        return self.tldr

    monkeypatch.setattr(Paper, "generate_tldr", fake_generate_tldr)

    executor = Executor(config)
    fake_retriever = executor.retrievers["arxiv"]
    executor.run()

    assert fake_retriever.enriched_titles == ["Top paper"]
    assert generated_titles == ["Top paper"]


def test_executor_does_not_initialize_llm_when_generation_is_disabled(config, monkeypatch):
    config.executor.source = ["arxiv"]
    config.executor.show_tldr = False
    config.executor.show_affiliations = False
    config.executor.max_paper_num = 1

    class FakeRetriever:
        name = "arxiv"

        def __init__(self, config):
            pass

        def retrieve_papers(self):
            return [
                Paper(
                    source="arxiv",
                    title="Only paper",
                    authors=["Author A"],
                    abstract="Abstract",
                    url="https://example.com/paper",
                    pdf_url="https://example.com/paper.pdf",
                )
            ]

        def enrich_papers(self, papers):
            raise AssertionError("enrich_papers should not run when generation is disabled")

    class FakeReranker:
        def __init__(self, config):
            pass

        def rerank(self, candidates, corpus):
            return list(candidates)

    def fail_openai(*args, **kwargs):
        raise AssertionError("OpenAI client should not initialize when TLDR and affiliations are disabled")

    monkeypatch.setattr("zotero_arxiv_daily.executor.get_retriever_cls", lambda _: FakeRetriever)
    monkeypatch.setattr("zotero_arxiv_daily.executor.get_reranker_cls", lambda _: FakeReranker)
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", fail_openai)
    monkeypatch.setattr(
        Executor,
        "fetch_zotero_corpus",
        lambda self: [
            CorpusPaper(
                title="Corpus paper",
                abstract="Corpus abstract",
                added_date=datetime(2026, 3, 10),
                paths=["/test"],
            )
        ],
    )
    monkeypatch.setattr(Executor, "filter_corpus", lambda self, corpus: corpus)
    monkeypatch.setattr("zotero_arxiv_daily.executor.render_email", lambda papers, **kwargs: "email")
    monkeypatch.setattr("zotero_arxiv_daily.executor.send_email", lambda config, content: None)

    executor = Executor(config)
    executor.run()
