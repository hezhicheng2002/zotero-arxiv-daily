from dataclasses import dataclass
from typing import Optional, TypeVar
from datetime import datetime
import re
import tiktoken
from openai import OpenAI
from loguru import logger
import json
import ast
RawPaperItem = TypeVar('RawPaperItem')

@dataclass
class Paper:
    source: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    pdf_url: Optional[str] = None
    full_text: Optional[str] = None
    tldr: Optional[str] = None
    affiliations: Optional[list[str]] = None
    score: Optional[float] = None

    def _generate_tldr_with_llm(self, openai_client:OpenAI,llm_params:dict) -> str:
        lang = llm_params.get('language', 'English')
        prompt = f"Given the following information of a paper, generate a one-sentence TLDR summary in {lang}:\n\n"
        if self.title:
            prompt += f"Title:\n {self.title}\n\n"

        if self.abstract:
            prompt += f"Abstract: {self.abstract}\n\n"

        if self.full_text:
            prompt += f"Preview of main content:\n {self.full_text}\n\n"

        if not self.full_text and not self.abstract:
            logger.warning(f"Neither full text nor abstract is provided for {self.url}")
            return "Failed to generate TLDR. Neither full text nor abstract is provided"
        
        # use gpt-4o tokenizer for estimation
        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)
        prompt_tokens = prompt_tokens[:4000]  # truncate to 4000 tokens
        prompt = enc.decode(prompt_tokens)
        
        response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"You are an assistant who perfectly summarizes scientific paper, and gives the core idea of the paper to the user. Your answer should be in {lang}.",
                },
                {"role": "user", "content": prompt},
            ],
            **llm_params.get('generation_kwargs', {})
        )
        tldr = response.choices[0].message.content
        return tldr

    def _classify_llm_error(self, error: Exception) -> str:
        message = str(error).lower()
        if any(keyword in message for keyword in ["401", "unauthorized", "authentication", "api key", "invalid_api_key"]):
            return "auth"
        if any(keyword in message for keyword in ["403", "forbidden", "permission", "insufficient_quota"]):
            return "permission_or_quota"
        if any(keyword in message for keyword in ["404", "model", "does not exist", "not found"]):
            return "model_or_endpoint"
        if any(keyword in message for keyword in ["429", "rate limit", "too many requests"]):
            return "rate_limit"
        if any(keyword in message for keyword in ["timeout", "timed out", "connection", "dns", "ssl", "network"]):
            return "network_or_timeout"
        if any(keyword in message for keyword in ["400", "invalid", "bad request", "max_tokens", "context length"]):
            return "request_params"
        return "unknown"
    
    def generate_tldr(
        self,
        openai_client: OpenAI,
        llm_params: dict,
        fallback_openai_client: Optional[OpenAI] = None,
        fallback_llm_params: Optional[dict] = None,
    ) -> str:
        try:
            tldr = self._generate_tldr_with_llm(openai_client,llm_params)
            self.tldr = tldr
            return tldr
        except Exception as e:
            category = self._classify_llm_error(e)
            lang = llm_params.get('language', 'English')
            model = llm_params.get('generation_kwargs', {}).get('model', None)
            logger.error(
                f"[TLDR][LLM_ERROR][{category}] url={self.url} lang={lang} model={model} error={type(e).__name__}: {e}"
            )
            if fallback_openai_client is not None and fallback_llm_params is not None:
                try:
                    logger.warning(f"[TLDR][FALLBACK_LLM] url={self.url} reason=primary_failed")
                    tldr = self._generate_tldr_with_llm(fallback_openai_client, fallback_llm_params)
                    self.tldr = tldr
                    return tldr
                except Exception as fallback_error:
                    fallback_category = self._classify_llm_error(fallback_error)
                    fallback_lang = fallback_llm_params.get('language', 'English')
                    fallback_model = fallback_llm_params.get('generation_kwargs', {}).get('model', None)
                    logger.error(
                        f"[TLDR][FALLBACK_LLM_ERROR][{fallback_category}] url={self.url} lang={fallback_lang} model={fallback_model} "
                        f"error={type(fallback_error).__name__}: {fallback_error}"
                    )
            logger.warning(
                f"[TLDR][FALLBACK_ABSTRACT] url={self.url} reason=llm_generation_failed"
            )
            tldr = self.abstract
            self.tldr = tldr
            return tldr

    def _generate_affiliations_with_llm(self, openai_client:OpenAI,llm_params:dict) -> Optional[list[str]]:
        if self.full_text is not None:
            prompt = f"Given the beginning of a paper, extract the affiliations of the authors in a python list format, which is sorted by the author order. If there is no affiliation found, return an empty list '[]':\n\n{self.full_text}"
            # use gpt-4o tokenizer for estimation
            enc = tiktoken.encoding_for_model("gpt-4o")
            prompt_tokens = enc.encode(prompt)
            prompt_tokens = prompt_tokens[:2000]  # truncate to 2000 tokens
            prompt = enc.decode(prompt_tokens)
            generation_kwargs = dict(llm_params.get('generation_kwargs', {}))
            generation_kwargs.setdefault("max_tokens", 256)
            generation_kwargs.setdefault("temperature", 0)

            affiliations = openai_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant who perfectly extracts affiliations of authors from a paper. You should return a python list of affiliations sorted by the author order, like [\"TsingHua University\",\"Peking University\"]. If an affiliation is consisted of multi-level affiliations, like 'Department of Computer Science, TsingHua University', you should return the top-level affiliation 'TsingHua University' only. Do not contain duplicated affiliations. If there is no affiliation found, you should return an empty list [ ]. You should only return the final list of affiliations, and do not return any intermediate results.",
                    },
                    {"role": "user", "content": prompt},
                ],
                **generation_kwargs
            )
            affiliations = affiliations.choices[0].message.content

            affiliations = self._parse_affiliations_output(affiliations)
            affiliations = list(set(affiliations))
            affiliations = [str(a) for a in affiliations]

            return affiliations

    def _parse_affiliations_output(self, output: str) -> list[str]:
        output = output.strip()
        matched = re.search(r'\[.*?\]', output, flags=re.DOTALL)
        if matched:
            output = matched.group(0)

        try:
            parsed = json.loads(output)
        except Exception:
            parsed = ast.literal_eval(output)

        if not isinstance(parsed, list):
            raise ValueError(f"Affiliations output is not a list: {type(parsed)}")

        return [str(item).strip() for item in parsed if str(item).strip()]
    
    def generate_affiliations(
        self,
        openai_client: OpenAI,
        llm_params: dict,
        fallback_openai_client: Optional[OpenAI] = None,
        fallback_llm_params: Optional[dict] = None,
    ) -> Optional[list[str]]:
        try:
            affiliations = self._generate_affiliations_with_llm(openai_client,llm_params)
            self.affiliations = affiliations
            return affiliations
        except Exception as e:
            logger.warning(f"Failed to generate affiliations of {self.url}: {e}")
            if fallback_openai_client is not None and fallback_llm_params is not None:
                try:
                    logger.warning(f"[AFFILIATIONS][FALLBACK_LLM] url={self.url} reason=primary_failed")
                    affiliations = self._generate_affiliations_with_llm(fallback_openai_client, fallback_llm_params)
                    self.affiliations = affiliations
                    return affiliations
                except Exception as fallback_error:
                    logger.warning(f"[AFFILIATIONS][FALLBACK_LLM_ERROR] url={self.url} error={fallback_error}")
            self.affiliations = None
            return None
@dataclass
class CorpusPaper:
    title: str
    abstract: str
    added_date: datetime
    paths: list[str]