"""
Knowledge Graph & Web Scraping API for SoloLLM — Phase 4.

Provides endpoints for:
- Graph queries, entity search, and visualization data
- NetworkX graph analysis (PageRank, centrality, communities)
- Memory inspector (entity timeline, management)
- Web scraping ingestion
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from memory.knowledge_graph import knowledge_graph

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/graph", tags=["Knowledge Graph"])


class EntitySearchRequest(BaseModel):
    query: str
    limit: int = 20


class ScrapeRequest(BaseModel):
    url: str
    workspace_id: str = "default"


@router.get("/stats")
async def get_graph_stats():
    """Get knowledge graph statistics."""
    return await knowledge_graph.get_graph_stats()


@router.post("/search")
async def search_entities(request: EntitySearchRequest):
    """Search entities in the knowledge graph."""
    entities = await knowledge_graph.search_entities(request.query, request.limit)
    return {"entities": entities, "count": len(entities)}


@router.get("/entity/{entity_id}")
async def get_entity_neighbors(entity_id: str):
    """Get an entity and its neighbors."""
    return await knowledge_graph.get_neighbors(entity_id)


@router.delete("/entity/{entity_id}")
async def delete_entity(entity_id: str):
    """Delete an entity and its relationships."""
    await knowledge_graph.delete_entity(entity_id)
    return {"success": True, "entity_id": entity_id}


@router.get("/visualization")
async def get_graph_visualization(limit: int = 200):
    """Get graph data for D3.js / force-directed visualization."""
    return await knowledge_graph.get_graph_data(limit)


@router.get("/analysis")
async def get_graph_analysis():
    """Run NetworkX graph analysis: PageRank, centrality, communities."""
    return await knowledge_graph.get_graph_analysis()


@router.get("/timeline")
async def get_entity_timeline(limit: int = 50):
    """Get recently added/updated entities for the memory inspector."""
    entities = await knowledge_graph.get_entity_timeline(limit)
    return {"entities": entities, "count": len(entities)}


@router.delete("/clear")
async def clear_graph():
    """Clear all entities and relationships from the knowledge graph."""
    await knowledge_graph.clear_graph()
    return {"success": True}


@router.post("/scrape")
async def scrape_url(request: ScrapeRequest):
    """Scrape a URL and ingest content into the RAG pipeline + knowledge graph."""
    from core.config import settings
    if not settings.web_scraping_enabled:
        raise HTTPException(status_code=403, detail="Web scraping is disabled")

    from rag.scraper import web_scraper
    result = await web_scraper.scrape_and_ingest(
        url=request.url,
        workspace_id=request.workspace_id,
    )

    if not result.get("success"):
        raise HTTPException(
            status_code=422,
            detail=result.get("errors", ["Scraping failed"])[0],
        )

    return result


@router.post("/scrape/preview")
async def scrape_preview(request: ScrapeRequest):
    """Preview scraped content without ingesting (for inspection)."""
    from core.config import settings
    if not settings.web_scraping_enabled:
        raise HTTPException(status_code=403, detail="Web scraping is disabled")

    from rag.scraper import web_scraper
    page = await web_scraper.scrape_url(request.url)
    if page.errors:
        raise HTTPException(status_code=422, detail=page.errors[0])

    return {
        "url": page.url,
        "title": page.title,
        "content_length": len(page.content),
        "content_preview": page.content[:2000],
        "link_count": len(page.links),
        "links": page.links[:20],
        "metadata": page.metadata,
    }
