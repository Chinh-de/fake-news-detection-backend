import os
import re
import random
import time
import wikipedia
import duckdb
import numpy as np
from bs4 import BeautifulSoup
from curl_cffi import requests as curl_requests
from ddgs import DDGS
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sqlalchemy import text
from app.config import settings
from app.services.text_processor import clean_text_transformer, preprocess_text
import logging
from app.config import settings

# Logger for retrieval
logger = logging.getLogger("retrieval_service")
level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
logger.setLevel(level)

# Configure trust domains
TRUST_DOMAINS = [
    "baochinhphu.vn",
    "vov.vn",
    "vtv.vn",
    "vnexpress.net",
    "tuoitre.vn",
    "thanhnien.vn",
    "vietnamnet.vn",
    "plo.vn"
]

ALL_SYNONYM_LABELS = [
    # Tin thật (6 từ)
    "Xác thực", "Đáng tin", "Chính xác", "Thực tế", "Tin cậy", "Trung thực",
    
    # Tin giả (6 từ)
    "Giả mạo", "Sai sự thật", "Gây hiểu lầm", "Tin đồn", "Lừa dối", "Bịa đặt"
]

class RetrievalService:
    def __init__(self):
        self._encoder = None

    def get_encoder(self) -> SentenceTransformer:
        """Lazy load and cache the embedding model on disk."""
        if self._encoder is None:
            os.makedirs(settings.EMBEDDING_MODEL_CACHE_DIR, exist_ok=True)
            logger.info("Loading embedding model (multilingual-e5-small) from cache/disk at %s...", settings.EMBEDDING_MODEL_CACHE_DIR)
            self._encoder = SentenceTransformer(
                "intfloat/multilingual-e5-small",
                cache_folder=settings.EMBEDDING_MODEL_CACHE_DIR
            )
            logger.info("Embedding model loaded.")
        return self._encoder

    def query_wikipedia(self, entity: str, lang: str = "vi", fetch_full: bool = False) -> str:
        """Query Wikipedia for a given entity definition."""
        try:
            wikipedia.set_lang(lang)
            if fetch_full:
                page = wikipedia.page(entity, auto_suggest=False)
                return page.content
            else:
                return wikipedia.summary(entity, auto_suggest=False)
        except Exception:
            return "Not found"

    async def get_wiki_definitions(self, entities: list[str]) -> dict[str, str]:
        """Fetch Wikipedia definitions for a list of entities."""
        res = {}
        for ent in entities:
            if not ent or not ent.strip():
                continue
            definition = self.query_wikipedia(ent.strip(), lang="vi", fetch_full=False)
            if "Not found" not in definition:
                res[ent.strip()] = definition
        return res

    async def search_postgres_corpus(self, db, query_text: str, limit: int = 8) -> list[dict]:
        """Search local news corpus using pg_trgm similarity."""
        # Using pg_trgm distance operator <->
        stmt = text("""
            SELECT text, label_id, source_dataset 
            FROM news_corpus 
            ORDER BY text <-> :query_text 
            LIMIT :limit
        """)
        try:
            result = await db.execute(stmt, {"query_text": query_text, "limit": limit})
            rows = result.fetchall()
            return [
                {
                    "text": row[0],
                    "label_id": row[1],
                    "source": row[2] or "DB/Corpus"
                }
                for row in rows
            ]
        except Exception as e:
            logger.error("Error querying postgres corpus: %s", e)
            return []

    def _search_ddgs_news_with_backend(self, ddgs: DDGS, query: str, max_results: int) -> list[dict]:
        news_items = []
        results_gen = ddgs.news(
            query=query,
            safesearch="off",
            timelimit=None,
            max_results=max_results,
            backend="bing",
        )
        for i, result in enumerate(results_gen):
            if i >= max_results:
                break
            title = result.get("title", "")
            body = result.get("body", "")
            url = result.get("url", result.get("href", ""))

            cleaned_body = clean_text_transformer(body)
            if cleaned_body:
                news_items.append({
                    "text": f"{title}\n{cleaned_body}",
                    "label_id": random.choice([0, 1]),
                    "source": url or "Internet/DDG"
                })
        return news_items

    def _search_ddgs_text_with_backend(self, ddgs: DDGS, query: str, max_results: int, backend: str) -> list[dict]:
        results = []
        results_gen = ddgs.text(
            query,
            backend=backend,
            max_results=max_results
        )
        for i, r in enumerate(results_gen):
            if i >= max_results:
                break
            title = r.get("title", "")
            url = r.get("href", r.get("url", ""))
            snippet = r.get("body", r.get("snippet", ""))
            results.append({
                "title": title,
                "url": url,
                "snippet": snippet
            })
        return results

    def search_internet_news(self, query: str, max_results: int = 10, max_retries: int = 3) -> list[dict]:
        """Tìm kiếm tin tức internet, chỉ dùng backend='bing', thử lại tối đa max_retries lần."""
        logger.debug("Internet news search: query=%s max_results=%d", query, max_results)
        last_exception = None
        for attempt in range(1, max_retries + 1):
            try:
                with DDGS(timeout=20) as ddgs:
                    news_items = self._search_ddgs_news_with_backend(ddgs, query, max_results)
                    if news_items:
                        logger.info("Internet news success on attempt %d: %d items", attempt, len(news_items))
                        return news_items
                    logger.warning("Internet news attempt %d returned empty results", attempt)
            except Exception as e:
                last_exception = e
                logger.warning("Internet news attempt %d failed: %s", attempt, e)
                if attempt < max_retries:
                    time.sleep(0.5)

        logger.error("Internet news failed after %d attempts. Last error: %s", max_retries, last_exception)
        return []

    async def retrieve_fewshot_examples(self, db, post_text: str, search_query: str) -> list[dict]:
        """
        Retrieve 4 few-shot examples from local DB news corpus + Internet search.
        Rerank using BM25.
        Assign random synonym labels.
        """
        # 1. Retrieve top 8 from local news corpus
        db_candidates = await self.search_postgres_corpus(db, post_text, limit=8)
        logger.info("Fewshot local corpus candidates=%d", len(db_candidates))
        
        # 2. Search internet news
        import asyncio
        loop = asyncio.get_event_loop()
        internet_candidates = await loop.run_in_executor(
            None, self.search_internet_news, search_query, 10
        )
        logger.info("Fewshot internet candidates=%d", len(internet_candidates))
        
        candidates = db_candidates + internet_candidates
        logger.info("Fewshot total candidates=%d", len(candidates))
        if not candidates:
            return []
            
        # 3. BM25 Reranking using post_text as query
        corpus_texts = [preprocess_text(c["text"]) for c in candidates]
        tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
        
        bm25 = BM25Okapi(tokenized_corpus)
        query_tokens = preprocess_text(post_text).lower().split()
        scores = bm25.get_scores(query_tokens)
        
        scored_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in scored_indices[:4]]
        logger.info("Fewshot reranked selected=%d", len(top_indices))
        
        fewshots = []
        for idx in top_indices:
            cand = candidates[idx]
            # Prefer a combined title + body when available to avoid truncated context
            raw_text = cand.get("text", "")
            title = cand.get("title", "")
            combined = (title.strip() + " - " + raw_text.strip()) if title else raw_text.strip()
 
            # Trim to last full sentence to avoid cutting mid-sentence in examples
            def trim_to_sentence(s: str, max_len: int = 1000) -> str:
                s = s.strip()
                if len(s) <= max_len:
                    return s
                s = s[:max_len]
                # find last sentence-ending punctuation
                pos = max(s.rfind('.'), s.rfind('!'), s.rfind('?'), s.rfind('…'))
                if pos != -1:
                    return s[: pos + 1].strip()
                return s.strip()
 
            final_text = trim_to_sentence(combined, max_len=1000)
 
            fewshots.append({
                "text": final_text,
                # Assign a random synonym label from ALL_SYNONYM_LABELS
                "label": random.choice(ALL_SYNONYM_LABELS),
                "source": cand.get("source")
            })
            
        return fewshots

    def search_trusted_articles(self, query: str, max_urls: int = 5) -> list[dict]:
        """Tìm kiếm trusted domains, thử xen kẽ yahoo -> bing -> yahoo -> bing."""
        site_filter = " OR ".join([f"site:{d}" for d in TRUST_DOMAINS])
        full_query = f"{query} ({site_filter})".strip()
        logger.debug("Trusted search full query: %s", full_query)

        backends = ["bing", "yahoo"]
        max_attempts = 2

        for attempt in range(1, max_attempts + 1):
            with DDGS(timeout=20) as ddgs:
                for backend in backends:
                    try:
                        results = self._search_ddgs_text_with_backend(ddgs, full_query, max_urls, backend)
                        if results:
                            logger.info(
                                "Trusted search SUCCESS with backend=%s (attempt %d): %d results",
                                backend,
                                attempt,
                                len(results),
                            )
                            return results
                        logger.warning("Trusted search backend=%s attempt %d returned empty", backend, attempt)
                    except Exception as e:
                        logger.error("Trusted search backend=%s attempt %d failed: %s", backend, attempt, e)
                    time.sleep(0.5)

        logger.info("Trusted search returned no results for query=%s", query)
        return []

    def scrape_article_text(self, url: str) -> str:
        """Scrape text from an URL."""
        browsers = ["chrome", "chrome110", "edge99", "safari15_3"]
        browser_choice = random.choice(browsers)
        try:
            response = curl_requests.get(url, impersonate=browser_choice, timeout=10)
            if response.status_code != 200:
                return ""
            soup = BeautifulSoup(response.text, "html.parser")
            for elem in soup(["script", "style", "nav", "footer", "header", "aside", "form", "button", "iframe"]):
                elem.decompose()
            paragraphs = soup.find_all("p")
            cleaned_p = []
            for p in paragraphs:
                txt = p.get_text(separator=" ", strip=True)
                if len(txt.split()) > 8:
                    cleaned_p.append(txt)
            text_content = " ".join(cleaned_p)
            text_content = re.sub(r"http[s]?://\S+", "", text_content)
            text_content = re.sub(r"\[.*?\]", "", text_content)
            out = re.sub(r"\s+", " ", text_content).strip()
            try:
                logger.debug("Scraped URL=%s status=%d len=%d", url, response.status_code, len(out))
            except Exception:
                pass
            return out
        except Exception:
            logger.debug("Failed scraping URL=%s", url)
            return ""

    def chunk_text(self, text_str: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
        """
        Chunk article text: length approx 800 chars, cut at nearest sentence boundary,
        with 150 chars overlap. Fallback hard split if no boundary in ±50 chars.
        """
        if not text_str:
            return []

        # Split into sentences using simple regex that keeps sentence-ending punctuation
        sentences = re.split(r'(?<=[\.\!\?…])\s+', text_str)

        # Accumulate sentences into chunks without breaking sentences
        temp_chunks = []
        current = ''
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if not current:
                current = s
            elif len(current) + 1 + len(s) <= chunk_size:
                current = current + ' ' + s
            else:
                temp_chunks.append(current.strip())
                current = s
        if current:
            temp_chunks.append(current.strip())

        # Apply overlap by prefixing each chunk (except first) with the last `overlap` chars
        final_chunks = []
        for i, c in enumerate(temp_chunks):
            if i == 0:
                final_chunks.append(c)
            else:
                prev = final_chunks[-1]
                prefix = prev[-overlap:] if len(prev) > overlap else prev
                # Ensure we don't exceed reasonable length too much; truncate prefix if needed
                candidate = (prefix + ' ' + c).strip()
                final_chunks.append(candidate)

        # Filter out very short chunks
        return [c for c in final_chunks if len(c.strip()) > 30]

    async def retrieve_rag_evidence(self, query_text: str, post_normalized_text: str) -> list[dict]:
        """
        Search trusted sites, scrape articles, chunk them, embed them,
        and perform semantic search with query = normalized post text to get top 4 chunks.
        """
        # 1. Search trusted sites
        logger.debug("retrieve_rag_evidence query=%s", query_text)
        results = self.search_trusted_articles(query_text, max_urls=5)
        logger.info("RAG trusted search results=%d", len(results))
        if not results:
            return []
            
        # 2. Scrape and chunk articles in parallel or sequentially
        import asyncio
        loop = asyncio.get_event_loop()
        
        def _scrape_and_chunk():
            all_chunks = []
            for r in results:
                scraped_text = self.scrape_article_text(r["url"])
                # Fallback to snippet if scraping fails or is too short
                if not scraped_text or len(scraped_text.split()) < 30:
                    scraped_text = r.get("snippet", "")
                if not scraped_text:
                    continue
                # Clean scraped text before chunking to normalize whitespace, remove noise
                scraped_text = clean_text_transformer(scraped_text)
                if not scraped_text:
                    continue

                chunks = self.chunk_text(scraped_text)
                for chunk in chunks:
                    all_chunks.append({
                        "chunk_text": chunk,
                        "title": r["title"],
                        "url": r["url"],
                        "source": "trusted_internet"
                    })
            return all_chunks
            
        chunks = await loop.run_in_executor(None, _scrape_and_chunk)
        logger.info("RAG scraped chunks=%d", len(chunks))
        if not chunks:
            return []
            
        # 3. Embed chunks and perform semantic search using multilingual-e5-small
        encoder = self.get_encoder()
        chunk_texts = [preprocess_text(c["chunk_text"]) for c in chunks]
        
        def _compute_embeddings():
            # e5 model format requires "query: " and "passage: " prefixes for best results
            query_emb = encoder.encode(f"query: {post_normalized_text}", normalize_embeddings=True)
            passage_texts = [f"passage: {ct}" for ct in chunk_texts]
            corpus_embs = encoder.encode(passage_texts, normalize_embeddings=True, batch_size=32)
            scores = (query_emb @ corpus_embs.T)
            return scores
            
        scores = await loop.run_in_executor(None, _compute_embeddings)
        
        # Sort and select top 4
        scored_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in scored_indices[:4]]
        logger.info("RAG selected chunks=%d", len(top_indices))
        
        evidence = []
        for idx in top_indices:
            chunk_info = chunks[idx]
            evidence.append({
                "score": float(scores[idx]),
                "chunk_text": chunk_info["chunk_text"],
                "title": chunk_info["title"],
                "url": chunk_info["url"],
                "source": chunk_info["source"]
            })
        logger.debug("selected evidence count=%d", len(evidence))
        try:
            if settings.ENABLE_ANALYSIS_LOG:
                for e in evidence:
                    logger.debug("Selected evidence: url=%s score=%s title=%s", e.get("url"), e.get("score"), e.get("title"))
        except Exception:
            pass
            
        return evidence

# Global singleton instance
retrieval_service = RetrievalService()
