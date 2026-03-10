from types import SimpleNamespace

from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever


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
