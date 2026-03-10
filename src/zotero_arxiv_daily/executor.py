from loguru import logger
from pyzotero import zotero
from omegaconf import DictConfig
from .utils import glob_match
from .retriever import get_retriever_cls
from .protocol import CorpusPaper
import random
import time
from datetime import datetime
from .reranker import get_reranker_cls
from .construct_email import render_email
from .utils import send_email
from openai import OpenAI
from tqdm import tqdm
class Executor:
    def __init__(self, config:DictConfig):
        self.config = config
        self.retrievers = {
            source: get_retriever_cls(source)(config) for source in config.executor.source
        }
        self.reranker = get_reranker_cls(config.executor.reranker)(config)
        self.openai_client = None
        self.fallback_openai_client = None
        self.fallback_llm_config = None
        self._llm_clients_initialized = False

    def _ensure_llm_clients(self) -> None:
        if self._llm_clients_initialized:
            return

        self.openai_client = OpenAI(
            api_key=self.config.llm.api.key,
            base_url=self.config.llm.api.base_url,
        )

        fallback_cfg = self.config.llm.get("fallback", None)
        if fallback_cfg and fallback_cfg.get("api", None):
            fallback_key = fallback_cfg.api.get("key", None)
            fallback_base_url = fallback_cfg.api.get("base_url", None)
            fallback_model = fallback_cfg.get("generation_kwargs", {}).get("model", None)
            if fallback_key and fallback_base_url and fallback_model:
                self.fallback_openai_client = OpenAI(api_key=fallback_key, base_url=fallback_base_url)
                self.fallback_llm_config = fallback_cfg
                logger.info("Configured fallback LLM provider")

        self._llm_clients_initialized = True

    def _is_retryable_zotero_error(self, error: Exception) -> bool:
        message = str(error).lower()
        retryable_markers = [
            "code: 429",
            "code: 502",
            "code: 503",
            "code: 504",
            "bad gateway",
            "timeout",
            "timed out",
            "connection",
            "temporarily unavailable",
            "service unavailable",
        ]
        return any(marker in message for marker in retryable_markers)

    def fetch_zotero_corpus(self) -> list[CorpusPaper]:
        logger.info("Fetching zotero corpus")
        zot = zotero.Zotero(self.config.zotero.user_id, 'user', self.config.zotero.api_key)
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                collections = zot.everything(zot.collections())
                collections = {c['key']: c for c in collections}
                corpus = zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint'))
                break
            except Exception as error:
                if attempt < max_retries and self._is_retryable_zotero_error(error):
                    backoff_seconds = 2 ** (attempt - 1)
                    logger.warning(
                        f"Transient Zotero API error (attempt {attempt}/{max_retries}): {error}. "
                        f"Retrying in {backoff_seconds}s..."
                    )
                    time.sleep(backoff_seconds)
                    continue
                raise
        corpus = [c for c in corpus if c['data']['abstractNote'] != '']
        def get_collection_path(col_key:str) -> str:
            if p := collections[col_key]['data']['parentCollection']:
                return get_collection_path(p) + '/' + collections[col_key]['data']['name']
            else:
                return collections[col_key]['data']['name']
        for c in corpus:
            paths = [get_collection_path(col) for col in c['data']['collections']]
            c['paths'] = paths
        logger.info(f"Fetched {len(corpus)} zotero papers")
        return [CorpusPaper(
            title=c['data']['title'],
            abstract=c['data']['abstractNote'],
            added_date=datetime.strptime(c['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'),
            paths=c['paths']
        ) for c in corpus]
    
    def filter_corpus(self, corpus:list[CorpusPaper]) -> list[CorpusPaper]:
        if not self.config.zotero.include_path:
            return corpus
        new_corpus = []
        logger.info(f"Selecting zotero papers matching include_path: {self.config.zotero.include_path}")
        for c in corpus:
            match_results = [glob_match(p, self.config.zotero.include_path) for p in c.paths]
            if any(match_results):
                new_corpus.append(c)
        samples = random.sample(new_corpus, min(5, len(new_corpus)))
        samples = '\n'.join([c.title + ' - ' + '\n'.join(c.paths) for c in samples])
        logger.info(f"Selected {len(new_corpus)} zotero papers:\n{samples}\n...")
        return new_corpus

    
    def run(self):
        show_tldr = self.config.executor.get("show_tldr", True)
        show_affiliations = self.config.executor.get("show_affiliations", True)
        corpus = self.fetch_zotero_corpus()
        corpus = self.filter_corpus(corpus)
        if len(corpus) == 0:
            logger.error(f"No zotero papers found. Please check your zotero settings:\n{self.config.zotero}")
            return
        all_papers = []
        for source, retriever in self.retrievers.items():
            logger.info(f"Retrieving {source} papers...")
            papers = retriever.retrieve_papers()
            if len(papers) == 0:
                logger.info(f"No {source} papers found")
                continue
            logger.info(f"Retrieved {len(papers)} {source} papers")
            all_papers.extend(papers)
        logger.info(f"Total {len(all_papers)} papers retrieved from all sources")
        reranked_papers = []
        if len(all_papers) > 0:
            logger.info("Reranking papers...")
            reranked_papers = self.reranker.rerank(all_papers, corpus)
            reranked_papers = reranked_papers[:self.config.executor.max_paper_num]
            if show_tldr or show_affiliations:
                logger.info(f"Preparing full text for {len(reranked_papers)} top-ranked papers...")
                papers_by_source = {}
                for paper in reranked_papers:
                    papers_by_source.setdefault(paper.source, []).append(paper)
                for source, papers in papers_by_source.items():
                    self.retrievers[source].enrich_papers(papers)
            if show_tldr and show_affiliations:
                logger.info("Generating TLDR and affiliations...")
            elif show_tldr and not show_affiliations:
                logger.info("Generating TLDR (affiliations disabled)...")
            elif (not show_tldr) and show_affiliations:
                logger.info("Generating affiliations (TLDR disabled)...")
            else:
                logger.info("TLDR and affiliations disabled. Skipping LLM generation.")
            if show_tldr or show_affiliations:
                self._ensure_llm_clients()
            for p in tqdm(reranked_papers):
                if show_tldr:
                    p.generate_tldr(
                        self.openai_client,
                        self.config.llm,
                        fallback_openai_client=self.fallback_openai_client,
                        fallback_llm_params=self.fallback_llm_config,
                    )
                if show_affiliations:
                    p.generate_affiliations(
                        self.openai_client,
                        self.config.llm,
                        fallback_openai_client=self.fallback_openai_client,
                        fallback_llm_params=self.fallback_llm_config,
                    )
        elif not self.config.executor.send_empty:
            logger.info("No new papers found. No email will be sent.")
            return
        logger.info("Sending email...")
        email_content = render_email(reranked_papers, show_tldr=show_tldr, show_affiliations=show_affiliations)
        send_email(self.config, email_content)
        logger.info("Email sent successfully")
