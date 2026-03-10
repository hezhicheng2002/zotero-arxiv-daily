from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf
from tempfile import TemporaryDirectory
import feedparser
from urllib.request import urlretrieve
from tqdm import tqdm
import os
from loguru import logger

@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")

    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        client = arxiv.Client(num_retries=10,delay_seconds=10)
        query = '+'.join(self.config.source.arxiv.category)
        # Get the latest paper from arxiv rss feed
        feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
        if 'Feed error for query' in feed.feed.title:
            raise Exception(f"Invalid ARXIV_QUERY: {query}.")
        raw_papers = []
        all_paper_ids = [i.id.removeprefix("oai:arXiv.org:") for i in feed.entries if i.get("arxiv_announce_type","new") == 'new']
        if self.config.executor.debug:
            all_paper_ids = all_paper_ids[:10]

        # Get full information of each paper from arxiv api
        bar = tqdm(total=len(all_paper_ids))
        for i in range(0,len(all_paper_ids),20):
            search = arxiv.Search(id_list=all_paper_ids[i:i+20])
            batch = list(client.results(search))
            bar.update(len(batch))
            raw_papers.extend(batch)
        bar.close()

        return raw_papers

    def convert_to_paper(self, raw_paper, *, need_full_text=False) -> Paper:
        title = raw_paper.title
        authors = [a.name for a in raw_paper.authors]
        abstract = raw_paper.summary
        pdf_url = raw_paper.pdf_url

        full_text = None
        if need_full_text:
            with TemporaryDirectory() as temp_dir:
                path = os.path.join(temp_dir, "paper.pdf")
                urlretrieve(pdf_url, path)
                try:
                    full_text = extract_markdown_from_pdf(path)
                except Exception as e:
                    logger.warning(f"Failed to extract full text of {title}: {e}")

        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper.entry_id,
            pdf_url=pdf_url,
            full_text=full_text
        )

    def retrieve_papers(self) -> list[Paper]:
        raw_papers = self._retrieve_raw_papers()
        papers = []
        logger.info("Processing papers...")
        need_full_text = self.config.executor.get("show_tldr", True) or self.config.executor.get("show_affiliations", True)
        if not need_full_text:
            logger.info("TLDR and affiliations are both disabled — skipping PDF download and full text extraction.")
        with ProcessPoolExecutor(max_workers=self.config.executor.max_workers) as exec_pool:
            from functools import partial
            papers = list(exec_pool.map(partial(self.convert_to_paper, need_full_text=need_full_text), raw_papers))
        return [p for p in papers if p is not None]
