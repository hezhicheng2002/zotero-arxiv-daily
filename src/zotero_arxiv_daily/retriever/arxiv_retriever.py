from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from queue import Empty
from tempfile import TemporaryDirectory
from typing import Any, Callable, TypeVar

import arxiv
from arxiv import Result as ArxivResult
import feedparser
from loguru import logger
import requests
from tqdm import tqdm

from .base import BaseRetriever, register_retriever
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar

T = TypeVar("T")

DEFAULT_REQUEST_TIMEOUT = 60
HTML_EXTRACTION_TIMEOUT = 60
FULL_TEXT_EXTRACTION_TIMEOUT = 180


def _download_file(url: str, path: str, timeout_seconds: int = DEFAULT_REQUEST_TIMEOUT) -> None:
    with requests.get(url, stream=True, timeout=(10, timeout_seconds)) as response:
        response.raise_for_status()
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def _run_in_subprocess(
    result_queue: Any,
    func: Callable[..., T | None],
    args: tuple[Any, ...],
) -> None:
    try:
        result_queue.put(("ok", func(*args)))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_with_hard_timeout(
    func: Callable[..., T | None],
    args: tuple[Any, ...],
    *,
    timeout: float,
    operation: str,
    paper_title: str,
) -> T | None:
    start_methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in start_methods else start_methods[0])
    result_queue = context.Queue()
    process = context.Process(target=_run_in_subprocess, args=(result_queue, func, args))
    process.start()

    try:
        status, payload = result_queue.get(timeout=timeout)
    except Empty:
        if process.is_alive():
            process.kill()
        process.join(5)
        result_queue.close()
        result_queue.join_thread()
        logger.warning(f"{operation} timed out for {paper_title} after {timeout} seconds")
        return None

    process.join(5)
    result_queue.close()
    result_queue.join_thread()

    if status == "ok":
        return payload

    logger.warning(f"{operation} failed for {paper_title}: {payload}")
    return None


def _extract_text_from_html_worker(html_url: str) -> str | None:
    import trafilatura

    downloaded = trafilatura.fetch_url(html_url)
    if downloaded is None:
        raise ValueError(f"Failed to download HTML from {html_url}")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text:
        raise ValueError(f"No text extracted from {html_url}")
    return text


def _extract_text_from_pdf_worker(pdf_url: str, timeout_seconds: int) -> str:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        _download_file(pdf_url, path, timeout_seconds)
        return extract_markdown_from_pdf(path)


def _extract_text_from_tar_worker(source_url: str, paper_id: str, timeout_seconds: int) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        _download_file(source_url, path, timeout_seconds)
        file_contents = extract_tex_code_from_tar(path, paper_id)
        if not file_contents or "all" not in file_contents:
            raise ValueError("Main tex file not found.")
        return file_contents["all"]


def _extract_full_text(
    title: str,
    entry_url: str,
    pdf_url: str | None,
    timeout_seconds: int = DEFAULT_REQUEST_TIMEOUT,
) -> str | None:
    html_url = entry_url.replace("/abs/", "/html/")
    full_text = _run_with_hard_timeout(
        _extract_text_from_html_worker,
        (html_url,),
        timeout=HTML_EXTRACTION_TIMEOUT,
        operation="HTML extraction",
        paper_title=title,
    )
    if full_text:
        return full_text

    if pdf_url:
        full_text = _run_with_hard_timeout(
            _extract_text_from_pdf_worker,
            (pdf_url, timeout_seconds),
            timeout=FULL_TEXT_EXTRACTION_TIMEOUT,
            operation="PDF extraction",
            paper_title=title,
        )
        if full_text:
            return full_text

    source_url = entry_url.replace("/abs/", "/e-print/")
    return _run_with_hard_timeout(
        _extract_text_from_tar_worker,
        (source_url, entry_url, timeout_seconds),
        timeout=FULL_TEXT_EXTRACTION_TIMEOUT,
        operation="Tar extraction",
        paper_title=title,
    )


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")

    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        client = arxiv.Client(num_retries=10, delay_seconds=10)
        query = "+".join(self.config.source.arxiv.category)
        include_cross_list = self.config.source.arxiv.get("include_cross_list", False)

        rss_url = f"https://rss.arxiv.org/atom/{query}"
        rss_timeout = self.config.executor.get("network_timeout_seconds", DEFAULT_REQUEST_TIMEOUT)
        response = requests.get(rss_url, timeout=rss_timeout)
        response.raise_for_status()
        feed = feedparser.parse(response.content)
        if "Feed error for query" in feed.feed.title:
            raise Exception(f"Invalid ARXIV_QUERY: {query}.")

        allowed_announce_types = {"new", "cross"} if include_cross_list else {"new"}
        all_paper_ids = [
            item.id.removeprefix("oai:arXiv.org:")
            for item in feed.entries
            if item.get("arxiv_announce_type", "new") in allowed_announce_types
        ]
        if self.config.executor.debug:
            all_paper_ids = all_paper_ids[:10]

        raw_papers = []
        bar = tqdm(total=len(all_paper_ids))
        for index in range(0, len(all_paper_ids), 20):
            search = arxiv.Search(id_list=all_paper_ids[index:index + 20])
            batch = list(client.results(search))
            bar.update(len(batch))
            raw_papers.extend(batch)
        bar.close()
        return raw_papers

    def convert_to_paper(self, raw_paper: ArxivResult) -> Paper:
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

    def enrich_papers(self, papers: list[Paper]) -> list[Paper]:
        need_full_text = self.config.executor.get("show_tldr", True) or self.config.executor.get("show_affiliations", True)
        if not need_full_text or len(papers) == 0:
            return papers

        logger.info(f"Extracting full text for {len(papers)} top-ranked arxiv papers...")
        timeout_seconds = self.config.executor.get("network_timeout_seconds", DEFAULT_REQUEST_TIMEOUT)
        max_workers = min(self.config.executor.get("max_workers", 4), len(papers))

        if max_workers <= 1:
            full_texts = [
                _extract_full_text(paper.title, paper.url, paper.pdf_url, timeout_seconds)
                for paper in tqdm(papers)
            ]
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as exec_pool:
                full_texts = list(
                    tqdm(
                        exec_pool.map(
                            _extract_full_text,
                            [paper.title for paper in papers],
                            [paper.url for paper in papers],
                            [paper.pdf_url for paper in papers],
                            [timeout_seconds] * len(papers),
                        ),
                        total=len(papers),
                    )
                )

        for paper, full_text in zip(papers, full_texts):
            paper.full_text = full_text
        return papers
