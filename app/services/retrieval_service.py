import os
import asyncio
import re
import random
import time
import wikipedia
# Override default User-Agent to bypass Wikipedia's robot detection/blocking
wikipedia.wikipedia.USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
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
        self._wiki_session = None

    def _get_wiki_session(self):
        """Lazy load the curl_cffi AsyncSession for Wikipedia requests to keep connections alive and prevent 429s."""
        if self._wiki_session is None:
            self._wiki_session = curl_requests.AsyncSession()
            self._wiki_session.headers.update({
                "User-Agent": "FakeNewsDetectionBot/1.0 (https://github.com/Fake-news-detection; contact@fakenewsdetection.com) curl-cffi/0.6"
            })
        return self._wiki_session

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

    async def query_wikipedia(self, entity: str, lang: str = "vi", fetch_full: bool = False, max_retries: int = 2) -> str:
        """Query Wikipedia for a given entity definition using curl_cffi AsyncSession to bypass WAF blocks/rate-limits."""
        url = f"https://{lang}.wikipedia.org/w/api.php"
        session = self._get_wiki_session()
        
        for attempt in range(1, max_retries + 1):
            try:
                # 1. Search for the entity using action=opensearch (auto-complete / prefix matching)
                search_params = {
                    "action": "opensearch",
                    "format": "json",
                    "search": entity,
                    "limit": 1
                }
                
                r = await session.get(url, params=search_params, timeout=10)
                if r.status_code != 200:
                    raise Exception(f"HTTP error {r.status_code}")
                
                data = r.json()
                if len(data) < 2 or not data[1]:
                    return "Not found"
                
                title = data[1][0]
                
                # 2. Get the extract (summary) for the exact title
                query_params = {
                    "action": "query",
                    "format": "json",
                    "prop": "extracts",
                    "exintro": True,
                    "explaintext": True,
                    "titles": title,
                    "redirects": 1
                }
                
                r = await session.get(url, params=query_params, timeout=10)
                if r.status_code != 200:
                    raise Exception(f"HTTP error {r.status_code}")
                
                pages = r.json().get("query", {}).get("pages", {})
                for page_id, page_data in pages.items():
                    if page_id == "-1":
                        continue
                    extract = page_data.get("extract", "").strip()
                    if extract:
                        return extract
                
                return "Not found"
            except Exception as e:
                logger.warning("Wikipedia query for '%s' failed (attempt %d/%d): %s", entity, attempt, max_retries, e)
                if attempt < max_retries:
                    # Sleep longer on rate limit / 429 to let WAF clear, using asyncio.sleep instead of time.sleep
                    sleep_time = 3.0 * attempt if "429" in str(e) else 0.5 * attempt
                    await asyncio.sleep(sleep_time)
        return "Not found"


    async def get_wiki_definitions(self, entities: list[str]) -> dict[str, str]:
        """Fetch Wikipedia definitions for a list of entities concurrently using curl_cffi AsyncSession."""
        if not entities:
            return {}
        
        # Deduplicate and filter out empty entities
        unique_entities = sorted(list(set(ent.strip() for ent in entities if ent and ent.strip())))
        if not unique_entities:
            return {}
            
        tasks = [self.query_wikipedia(ent, lang="vi", fetch_full=False) for ent in unique_entities]
        
        # Execute concurrently
        results = await asyncio.gather(*tasks)
        
        res = {}
        for ent, definition in zip(unique_entities, results):
            if definition and "Not found" not in definition:
                res[ent] = definition
        return res

    async def search_postgres_corpus(self, db, query_text: str, limit: int = 8) -> list[dict]:
        """Search local news corpus using pg_trgm similarity."""
        # Using pg_trgm distance operator <->
        stmt = text("""
            SELECT text, source_dataset 
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
                    "source": row[1] or "DB/Corpus"
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
            region="vn-vi",
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
                    "source": url or "Internet/DDG"
                })
        return news_items

    def _search_ddgs_text_with_backend(self, ddgs: DDGS, query: str, max_results: int, backend: str) -> list[dict]:
        results = []
        results_gen = ddgs.text(
            query,
            region="vn-vi",
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

    async def search_internet_news(self, query: str, max_results: int = 10, max_retries: int = 3) -> list[dict]:
        """Tìm kiếm tin tức internet, chỉ dùng backend='bing', thử lại tối đa max_retries lần."""
        logger.warning("Internet news search: query=%s, using backend='bing' (attempting up to %d times)", query, max_retries)
        last_exception = None
        loop = asyncio.get_event_loop()
        for attempt in range(1, max_retries + 1):
            try:
                def _do_search():
                    with DDGS(timeout=20) as ddgs:
                        return self._search_ddgs_news_with_backend(ddgs, query, max_results)
                
                news_items = await loop.run_in_executor(None, _do_search)
                if news_items:
                    logger.warning("Internet news success on attempt %d: %d items", attempt, len(news_items))
                    for idx, item in enumerate(news_items, 1):
                        logger.warning("Internet news result #%d: source=%s, text_preview=%s", idx, item.get("source"), item.get("text", "")[:120].replace('\n', ' ') + "...")
                    return news_items
                logger.warning("Internet news attempt %d returned empty results", attempt)
            except Exception as e:
                last_exception = e
                logger.warning("Internet news attempt %d failed: %s", attempt, e)
                if attempt < max_retries:
                    await asyncio.sleep(0.5)

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
        internet_candidates = await self.search_internet_news(search_query, 10)
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

    async def search_trusted_articles(self, query: str, max_urls: int = 10) -> list[dict]:
        """Tìm kiếm trusted domains, thử xen kẽ yahoo -> bing -> yahoo -> bing."""
        site_filter = " OR ".join([f"site:{d}" for d in TRUST_DOMAINS])
        full_query = f"({site_filter}) {query}".strip()
        logger.warning("Trusted search full query: %s", full_query)

        backends = ["yahoo", "brave", "bing", "auto"]
        max_attempts = 2
        loop = asyncio.get_event_loop()

        for attempt in range(1, max_attempts + 1):
            for backend in backends:
                try:
                    logger.warning("Initiating trusted search using backend='%s' (attempt %d/%d) with query: %s", backend, attempt, max_attempts, full_query)
                    def _do_search():
                        with DDGS(timeout=20) as ddgs:
                            return self._search_ddgs_text_with_backend(ddgs, full_query, max_urls, backend)
                    
                    results = await loop.run_in_executor(None, _do_search)
                    if results:
                        logger.warning(
                            "Trusted search SUCCESS with backend=%s (attempt %d): %d results",
                            backend,
                            attempt,
                            len(results),
                        )
                        for idx, r in enumerate(results, 1):
                            logger.warning("Trusted search result #%d: title=%s, url=%s, snippet=%s", idx, r.get("title"), r.get("url"), r.get("snippet", "")[:120].replace('\n', ' ') + "...")
                        return results
                    logger.warning("Trusted search backend=%s attempt %d returned empty", backend, attempt)
                except Exception as e:
                    logger.error("Trusted search backend=%s attempt %d failed: %s", backend, attempt, e)
                await asyncio.sleep(0.5)

        logger.warning("Trusted search returned no results for query=%s", query)
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
                logger.warning("Scraped URL=%s status=%d len=%d", url, response.status_code, len(out))
            except Exception:
                pass
            return out
        except Exception:
            logger.warning("Failed scraping URL=%s", url)
            return ""

    def chunk_text(self, text_str: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
        """
        Chunk article text: length approx chunk_size (default 1000) chars,
        prioritizing keeping whole sentences, with overlap (default 150) chars.
        Ensure overlap is done at sentence boundaries (no chopped sentences/words).
        """
        if not text_str:
            return []

        # Split into sentences using simple regex that keeps sentence-ending punctuation
        raw_sentences = re.split(r'(?<=[\.\!\?…])\s+', text_str)
        sentences = []
        for s in raw_sentences:
            s = s.strip()
            if not s:
                continue
            # If a single sentence is extremely long, we split it by words to avoid issues
            if len(s) > chunk_size:
                words = s.split()
                curr_words = []
                curr_len = 0
                for w in words:
                    if curr_words and curr_len + 1 + len(w) > chunk_size - overlap:
                        sentences.append(" ".join(curr_words))
                        curr_words = [w]
                        curr_len = len(w)
                    else:
                        curr_words.append(w)
                        curr_len += len(w) + (1 if curr_len > 0 else 0)
                if curr_words:
                    sentences.append(" ".join(curr_words))
            else:
                sentences.append(s)

        if not sentences:
            return []

        chunks = []
        current_sentences = []
        current_length = 0

        for s in sentences:
            s_len = len(s)
            # If adding this sentence exceeds chunk_size
            if current_sentences and current_length + 1 + s_len > chunk_size:
                # Add current chunk
                chunks.append(" ".join(current_sentences))
                
                # Calculate overlap sentences
                overlap_sentences = []
                overlap_len = 0
                for os in reversed(current_sentences):
                    # We want to stop if we exceed overlap
                    if overlap_len >= overlap:
                        break
                    # Also avoid carrying over the entire chunk
                    if len(overlap_sentences) == len(current_sentences) - 1:
                        break
                    overlap_sentences.append(os)
                    overlap_len += len(os) + (1 if overlap_len > 0 else 0)
                
                overlap_sentences.reverse()
                current_sentences = list(overlap_sentences)
                current_length = sum(len(os) for os in current_sentences) + len(current_sentences) - 1
                if current_length < 0:
                    current_length = 0

            if current_sentences:
                current_sentences.append(s)
                current_length += 1 + s_len
            else:
                current_sentences.append(s)
                current_length = s_len

        if current_sentences:
            chunks.append(" ".join(current_sentences))

        # Filter out very short chunks
        return [c.strip() for c in chunks if len(c.strip()) > 30]

    def is_trusted_url(self, url: str) -> bool:
        """Check if a URL belongs to one of the trusted domains."""
        if not url:
            return False
        url_lower = url.strip().lower()
        domain = re.sub(r'^https?://(www\.)?', '', url_lower)
        host = domain.split('/')[0].split('?')[0].split(':')[0]
        for d in TRUST_DOMAINS:
            if host == d or host.endswith("." + d):
                return True
        return False
    def is_valid_article_url(self, url: str) -> bool:
        # Loại bỏ file sitemap, file ảnh, hoặc các trang định dạng xml
        if url.lower().endswith(('.xml', '.png', '.jpg', '.jpeg', '.gif')):
            return False
        # Loại bỏ các trang sitemap của vnexpress
        if "sitemap" in url.lower():
            return False
        return True


    async def retrieve_rag_evidence(self, query_text: str, post_normalized_text: str) -> list[dict]:
        """
        Search trusted sites using both LLM query and clean sliced input query in parallel,
        scrape articles, chunk them, embed them, and perform semantic search.
        """
        MAX_LEN = 265

        loop = asyncio.get_event_loop()
        # Clean input query: slice to max 265 chars

        input_raw_text = post_normalized_text.strip()
    
        if len(input_raw_text) <= MAX_LEN:
            full_query = input_raw_text
        else:
            # cắt trong giới hạn rồi lùi về khoảng trắng gần nhất
            cut = input_raw_text[:MAX_LEN]

            last_space = cut.rfind(" ")

            # nếu không có space (case hiếm) thì fallback hard cut
            if last_space == -1:
                full_query = cut
            else:
                full_query = cut[:last_space]

        cleaned_input_query = full_query.strip()
        
        logger.warning("Running parallel trusted search. Query 1 (LLM): '%s', Query 2 (Clean Input): '%s'", query_text, cleaned_input_query)
        
        # Run both searches in parallel
        task1 = self.search_trusted_articles(query_text, 10)
        task2 = self.search_trusted_articles(cleaned_input_query, 10)
        
        results1, results2 = await asyncio.gather(task1, task2)
        
        # Combine and deduplicate by URL
        results = []
        seen_urls = set()
        for r in (results1 + results2):
            url = r.get("url")
            if url:
                if not self.is_trusted_url(url):
                    logger.warning("RAG search returned untrusted domain URL: %s, filtering out", url)
                    continue
                if not self.is_valid_article_url(url):
                    logger.warning("RAG search returned non-article URL: %s, filtering out", url)
                    continue

                # Normalize URL for deduplication check
                norm_url = url.strip().lower()
                norm_url = re.sub(r'^https?://(www\.)?', '', norm_url)
                norm_url = norm_url.rstrip('/')
                if '?' in norm_url:
                    base, query = norm_url.split('?', 1)
                    params = [p for p in query.split('&') if not p.startswith('utm_')]
                    if params:
                        norm_url = base + '?' + '&'.join(params)
                    else:
                        norm_url = base
                
                if norm_url not in seen_urls:
                    seen_urls.add(norm_url)
                    results.append(r)
                
        logger.warning("RAG trusted search combined results=%d (LLM=%d, Clean Input=%d)", len(results), len(results1), len(results2))
        if not results:
            return []

        # Scrape all articles in parallel using the default thread pool executor
        scrape_tasks = [
            loop.run_in_executor(None, self.scrape_article_text, r["url"])
            for r in results
        ]
        scraped_texts = await asyncio.gather(*scrape_tasks)

        # Process results (CPU-bound: cleaning & chunking) sequentially
        chunks = []
        for r, scraped_text in zip(results, scraped_texts):
            # Fallback to snippet if scraping fails or is too short
            if not scraped_text or len(scraped_text.split()) < 30:
                scraped_text = r.get("snippet", "")
            if not scraped_text:
                continue
            
            # Clean text
            cleaned_text = clean_text_transformer(scraped_text)
            if not cleaned_text:
                continue

            # Limit to first 15 chunks (~15,000 characters) per article to prevent noise/comments blowing up embedding time
            article_chunks = self.chunk_text(cleaned_text)[:15]
            for chunk in article_chunks:
                chunks.append({
                    "chunk_text": chunk,
                    "title": r["title"],
                    "url": r["url"],
                    "source": "trusted_internet"
                })
        logger.warning("RAG scraped chunks=%d", len(chunks))
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
        logger.warning("RAG selected chunks=%d", len(top_indices))
        
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
        logger.warning("Selected evidence count=%d", len(evidence))
        try:
            for idx, e in enumerate(evidence, 1):
                logger.warning("RAG Evidence #%d: url=%s score=%.4f title=%s", idx, e.get("url"), e.get("score"), e.get("title"))
                logger.warning("RAG Evidence #%d Chunk: %s", idx, e.get("chunk_text")[:200] + "..." if len(e.get("chunk_text", "")) > 200 else e.get("chunk_text"))
        except Exception:
            pass
            
        return evidence

# Global singleton instance
retrieval_service = RetrievalService()
