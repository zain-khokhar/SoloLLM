"""
Web Scraping Ingestion for SoloLLM — Phase 4.

Fetches web pages, extracts clean text content, and feeds it into
the RAG pipeline for indexing and knowledge graph extraction.
"""

import re
import logging
import hashlib
import uuid
from urllib.parse import urlparse, urljoin
from dataclasses import dataclass, field

import httpx
from bs4 import BeautifulSoup

from rag.ingest import ParsedDocument, DocumentSection

logger = logging.getLogger(__name__)

# Timeout for HTTP requests
REQUEST_TIMEOUT = 30
MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB max page size


@dataclass
class ScrapedPage:
    """Result of scraping a single web page."""
    url: str
    title: str
    content: str
    links: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


class WebScraper:
    """
    Fetches and parses web pages into clean text for RAG ingestion.

    Features:
    - Cleans HTML to meaningful text
    - Extracts headings as sections
    - Discovers outbound links
    - Respects content size limits
    """

    REMOVE_TAGS = {"script", "style", "nav", "footer", "header", "aside", "noscript", "iframe"}

    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=REQUEST_TIMEOUT,
                follow_redirects=True,
                headers={
                    "User-Agent": "SoloLLM/1.0 (Knowledge Assistant; +http://localhost)",
                },
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def scrape_url(self, url: str) -> ScrapedPage:
        """Scrape a single URL and return cleaned content."""
        parsed_url = urlparse(url)
        if parsed_url.scheme not in ("http", "https"):
            return ScrapedPage(url=url, title="", content="",
                               errors=["Only HTTP/HTTPS URLs are supported"])

        client = await self._get_client()
        try:
            response = await client.get(url)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            return ScrapedPage(url=url, title="", content="",
                               errors=[f"HTTP {e.response.status_code}"])
        except httpx.RequestError as e:
            return ScrapedPage(url=url, title="", content="",
                               errors=[f"Request failed: {str(e)}"])

        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type and "text/plain" not in content_type:
            return ScrapedPage(url=url, title="", content="",
                               errors=[f"Unsupported content type: {content_type}"])

        if len(response.content) > MAX_CONTENT_LENGTH:
            return ScrapedPage(url=url, title="", content="",
                               errors=["Page too large (>5MB)"])

        if "text/plain" in content_type:
            return ScrapedPage(
                url=url,
                title=parsed_url.netloc + parsed_url.path,
                content=response.text[:100000],
                metadata={"source_url": url, "content_type": "text/plain"},
            )

        return self._parse_html(url, response.text)

    def _parse_html(self, url: str, html: str) -> ScrapedPage:
        """Parse HTML and extract clean text, sections, and links."""
        soup = BeautifulSoup(html, "html.parser")

        # Remove unwanted elements
        for tag in soup.find_all(self.REMOVE_TAGS):
            tag.decompose()

        # Extract title
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        if not title:
            h1 = soup.find("h1")
            if h1:
                title = h1.get_text(strip=True)

        # Extract links
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            abs_url = urljoin(url, href)
            parsed = urlparse(abs_url)
            if parsed.scheme in ("http", "https"):
                links.append(abs_url)

        # Extract text content
        body = soup.find("body") or soup
        content = self._extract_text(body)

        # Extract metadata
        metadata = {"source_url": url, "content_type": "text/html"}
        desc_tag = soup.find("meta", attrs={"name": "description"})
        if desc_tag and desc_tag.get("content"):
            metadata["description"] = desc_tag["content"]

        return ScrapedPage(
            url=url,
            title=title or urlparse(url).netloc,
            content=content,
            links=list(set(links))[:100],
            metadata=metadata,
        )

    def _extract_text(self, element) -> str:
        """Extract clean text from a BeautifulSoup element."""
        texts = []
        for child in element.descendants:
            if child.name in self.REMOVE_TAGS:
                continue
            if isinstance(child, str):
                text = child.strip()
                if text:
                    texts.append(text)
            elif child.name in ("p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"):
                texts.append("\n")

        content = " ".join(texts)
        # Collapse whitespace
        content = re.sub(r'\n\s*\n', '\n\n', content)
        content = re.sub(r' +', ' ', content)
        return content.strip()[:100000]

    def to_parsed_document(self, page: ScrapedPage) -> ParsedDocument:
        """Convert a ScrapedPage to a ParsedDocument for the RAG pipeline."""
        # Build sections from content blocks
        sections = []
        paragraphs = page.content.split("\n\n")
        for i, para in enumerate(paragraphs):
            if para.strip():
                sections.append(DocumentSection(
                    title=f"Section {i + 1}",
                    content=para.strip(),
                    level=0,
                    section_index=i,
                ))

        return ParsedDocument(
            filename=page.url,
            file_type="html",
            title=page.title,
            content=page.content,
            sections=sections,
            metadata=page.metadata,
            page_count=1,
            errors=page.errors,
        )

    async def scrape_and_ingest(self, url: str, workspace_id: str = "default") -> dict:
        """
        Scrape a URL and ingest it into the RAG pipeline.
        Also extracts entities for the knowledge graph.
        """
        from rag.pipeline import rag_pipeline
        from memory.knowledge_graph import knowledge_graph

        page = await self.scrape_url(url)
        if page.errors:
            return {"success": False, "errors": page.errors, "url": url}

        if not page.content.strip():
            return {"success": False, "errors": ["No content extracted"], "url": url}

        parsed = self.to_parsed_document(page)

        # Generate document ID
        content_hash = hashlib.sha256(page.content.encode()).hexdigest()[:16]
        document_id = f"web_{uuid.uuid4().hex[:12]}"

        from rag.chunking import chunking_engine
        from rag.embeddings import embedding_engine
        from rag.vectorstore import vector_store
        from rag.keyword_index import keyword_index

        # Chunk
        chunks = chunking_engine.chunk_document(
            content=parsed.content,
            sections=parsed.sections if parsed.sections else None,
            document_id=document_id,
            document_title=parsed.title,
        )

        if not chunks:
            return {"success": False, "errors": ["No chunks created"], "url": url}

        # Embed
        chunk_texts = [c.content for c in chunks]
        embeddings = embedding_engine.embed_documents(chunk_texts)

        # Prepare and store
        chunk_dicts = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_dicts.append({
                "id": chunk.id,
                "document_id": document_id,
                "content": chunk.content,
                "embedding": embedding,
                "document_title": chunk.document_title,
                "section_title": chunk.section_title,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "parent_chunk_id": chunk.parent_chunk_id,
                "metadata": chunk.metadata,
            })

        await vector_store.add_document(
            document_id=document_id,
            filename=page.url,
            file_type="html",
            title=parsed.title,
            content_hash=content_hash,
            chunk_count=len(chunks),
            page_count=1,
            workspace_id=workspace_id,
            metadata=parsed.metadata,
        )
        await vector_store.add_chunks(chunk_dicts, workspace_id)
        await keyword_index.add_chunks(chunk_dicts, workspace_id)

        # Extract entities for knowledge graph
        graph_result = await knowledge_graph.extract_and_store(
            page.content, document_id
        )

        result = {
            "success": True,
            "url": url,
            "document_id": document_id,
            "title": parsed.title,
            "chunk_count": len(chunks),
            "content_length": len(page.content),
            "link_count": len(page.links),
            "workspace_id": workspace_id,
            "entities_extracted": graph_result.get("entities", 0),
            "relationships_extracted": graph_result.get("relationships", 0),
        }

        logger.info(
            f"Scraped & ingested '{page.title}' from {url}: "
            f"{len(chunks)} chunks, {graph_result.get('entities', 0)} entities"
        )

        return result


# Singleton
web_scraper = WebScraper()
