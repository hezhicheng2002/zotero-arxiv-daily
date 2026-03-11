from concurrent.futures import ProcessPoolExecutor
import os
from tempfile import TemporaryDirectory

import arxiv
from arxiv import Result as ArxivResult
import feedparser
from loguru import logger
import requests
from tqdm import tqdm

from .base import BaseRetriever, register_retriever
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf


def _download_and_extract_full_text(title: str, pdf_url: str | None, timeout_seconds: int = 60) -> str | None:
    if not pdf_url:
        return None

    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        try:
            response = requests.get(pdf_url, timeout=timeout_seconds)
            response.raise_for_status()
            with open(path, "wb") as file:
                file.write(response.content)
            return extract_markdown_from_pdf(path)
        except Exception as error:
            logger.warning(f"Failed to extract full text of {title}: {error}")
            return None


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")

    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        client = arxiv.Client(num_retries=10, delay_seconds=10)
        query = "+".join(self.config.source.arxiv.category)
        # Get the latest paper from arxiv rss feed.
        rss_url = f"https://rss.arxiv.org/atom/{query}"
        rss_timeout = self.config.executor.get("network_timeout_seconds", 60)
        response = requests.get(rss_url, timeout=rss_timeout)
        response.raise_for_status()
        feed = feedparser.parse(response.content)
        if "Feed error for query" in feed.feed.title:
            raise Exception(f"Invalid ARXIV_QUERY: {query}.")
        raw_papers = []
        all_paper_ids = [
            item.id.removeprefix("oai:arXiv.org:")
            for item in feed.entries
            if item.get("arxiv_announce_type", "new") == "new"
        ]
        if self.config.executor.debug:
            all_paper_ids = all_paper_ids[:10]

        # Get full metadata for each paper from the arxiv API.
        bar = tqdm(total=len(all_paper_ids))
        for index in range(0, len(all_paper_ids), 20):
            search = arxiv.Search(id_list=all_paper_ids[index:index + 20])
            batch = list(client.results(search))
            bar.update(len(batch))
            raw_papers.extend(batch)
        bar.close()

        return raw_papers

    def convert_to_paper(self, raw_paper:ArxivResult) -> Paper:
        return Paper(
            source=self.name,
            title=raw_paper.title,
            authors=[author.name for author in raw_paper.authors],
            abstract=raw_paper.summary,
            url=raw_paper.entry_id,
            pdf_url=raw_paper.pdf_url,
            full_text=None,
        )

    def retrieve_papers(self) -> list[Paper]:
        raw_papers = self._retrieve_raw_papers()
        logger.info("Processing papers...")
        need_full_text = self.config.executor.get("show_tldr", True) or self.config.executor.get("show_affiliations", True)
        if need_full_text:
            logger.info("Initial retrieval uses metadata only. Full text extraction will run after reranking for the final selected papers.")
        else:
            logger.info("TLDR and affiliations are both disabled - skipping PDF download and full text extraction.")
        papers = [self.convert_to_paper(raw_paper) for raw_paper in raw_papers]
        return [paper for paper in papers if paper is not None]

    def enrich_papers(self, papers:list[Paper]) -> list[Paper]:
        need_full_text = self.config.executor.get("show_tldr", True) or self.config.executor.get("show_affiliations", True)
        if not need_full_text or len(papers) == 0:
            return papers

        logger.info(f"Extracting full text for {len(papers)} top-ranked arxiv papers...")
        max_workers = min(self.config.executor.max_workers, len(papers))
        if max_workers <= 1:
            full_texts = [
                _download_and_extract_full_text(
                    paper.title,
                    paper.pdf_url,
                    self.config.executor.get("network_timeout_seconds", 60),
                )
                for paper in tqdm(papers)
            ]
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as exec_pool:
                full_texts = list(
                    tqdm(
                        exec_pool.map(
                            _download_and_extract_full_text,
                            [paper.title for paper in papers],
                            [paper.pdf_url for paper in papers],
                            [self.config.executor.get("network_timeout_seconds", 60)] * len(papers),
                        ),
                        total=len(papers),
                    )
                )

        for paper, full_text in zip(papers, full_texts):
            paper.full_text = full_text
        return papers
