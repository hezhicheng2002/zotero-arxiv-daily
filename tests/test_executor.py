from datetime import datetime

from zotero_arxiv_daily.executor import Executor
from zotero_arxiv_daily.protocol import CorpusPaper, Paper


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
