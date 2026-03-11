"""
Knowledge Graph Engine for SoloLLM — Phase 4.

Builds and queries a local knowledge graph from extracted entities.
Uses SQLite for persistence + NetworkX for in-memory graph algorithms.

Includes:
- Entity extraction (regex + heuristic NER)
- Relationship detection (co-occurrence + verb patterns)
- Graph storage (SQLite-backed)
- NetworkX graph analysis (PageRank, centrality, community detection)
- Graph-augmented retrieval
- Graph traversal and querying
"""

import re
import json
import logging
import hashlib
import aiosqlite
import networkx as nx
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict

from core.config import settings

logger = logging.getLogger(__name__)

GRAPH_DB_PATH = str(settings.data_dir / "db" / "knowledge_graph.db")

GRAPH_SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    entity_type TEXT DEFAULT 'concept',
    description TEXT DEFAULT '',
    source_document_id TEXT DEFAULT '',
    mention_count INTEGER DEFAULT 1,
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);

CREATE TABLE IF NOT EXISTS relationships (
    id TEXT PRIMARY KEY,
    source_entity_id TEXT NOT NULL,
    target_entity_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    description TEXT DEFAULT '',
    source_document_id TEXT DEFAULT '',
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY (source_entity_id) REFERENCES entities(id),
    FOREIGN KEY (target_entity_id) REFERENCES entities(id)
);

CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_entity_id);
CREATE INDEX IF NOT EXISTS idx_rel_type ON relationships(relation_type);
"""


@dataclass
class Entity:
    """A node in the knowledge graph."""
    id: str
    name: str
    entity_type: str = "concept"
    description: str = ""
    mention_count: int = 1
    metadata: dict = field(default_factory=dict)


@dataclass
class Relationship:
    """An edge in the knowledge graph."""
    id: str
    source_id: str
    target_id: str
    relation_type: str
    weight: float = 1.0
    description: str = ""


# ── Entity Extraction ──────────────────────────────────────

class EntityExtractor:
    """
    Extracts entities from text using regex + heuristic NER.

    Detects:
    - Proper nouns (capitalized multi-word phrases)
    - Technical terms (domain-specific patterns)
    - Dates, numbers, codes
    - Quoted terms
    """

    # Common entity type patterns
    PATTERNS = {
        "person": re.compile(
            r'\b(Dr\.?|Mr\.?|Mrs\.?|Ms\.?|Prof\.?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        ),
        "organization": re.compile(
            r'\b([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)+(?:\s+(?:Inc|Corp|Ltd|LLC|Co|University|Institute|Foundation|Association)\.?))\b'
        ),
        "date": re.compile(
            r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\w+ \d{1,2},? \d{4})\b'
        ),
        "code_ref": re.compile(
            r'`([^`]+)`'
        ),
        "quoted": re.compile(
            r'"([^"]{3,50})"'
        ),
        "url": re.compile(
            r'(https?://[^\s<>"]+)'
        ),
        "email": re.compile(
            r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b'
        ),
        "technology": re.compile(
            r'\b(Python|JavaScript|TypeScript|Java|C\+\+|Rust|Go|React|Vue|Angular|'
            r'Docker|Kubernetes|AWS|Azure|GCP|PostgreSQL|MongoDB|Redis|Kafka|'
            r'TensorFlow|PyTorch|FastAPI|Django|Flask|Node\.js|Next\.js)\b',
            re.IGNORECASE,
        ),
    }

    # Verb patterns for relationship extraction
    RELATION_PATTERNS = [
        (re.compile(r'(\b\w+\b)\s+(?:is|are|was|were)\s+(?:a|an|the)\s+(\b\w+\b)', re.I), "is_a"),
        (re.compile(r'(\b\w+\b)\s+(?:uses?|utilizes?|employs?)\s+(\b\w+\b)', re.I), "uses"),
        (re.compile(r'(\b\w+\b)\s+(?:contains?|includes?|has)\s+(\b\w+\b)', re.I), "contains"),
        (re.compile(r'(\b\w+\b)\s+(?:depends?\s+on|requires?)\s+(\b\w+\b)', re.I), "depends_on"),
        (re.compile(r'(\b\w+\b)\s+(?:creates?|generates?|produces?)\s+(\b\w+\b)', re.I), "creates"),
        (re.compile(r'(\b\w+\b)\s+(?:extends?|inherits?\s+from)\s+(\b\w+\b)', re.I), "extends"),
    ]

    def extract(self, text: str, document_id: str = "") -> list[Entity]:
        """Extract entities from text."""
        entities = []
        seen_names: set[str] = set()

        # Pattern-based extraction
        for entity_type, pattern in self.PATTERNS.items():
            for match in pattern.finditer(text):
                name = match.group(0).strip(' "\'`')
                if entity_type == "person" and len(match.groups()) > 1:
                    name = match.group(2)

                if len(name) < 2 or name.lower() in seen_names:
                    continue

                seen_names.add(name.lower())
                entity_id = hashlib.md5(name.lower().encode()).hexdigest()[:12]
                entities.append(Entity(
                    id=f"ent_{entity_id}",
                    name=name,
                    entity_type=entity_type,
                ))

        # Capitalized phrases (proper nouns, 2-4 words)
        proper_nouns = re.findall(
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b', text
        )
        for pn in proper_nouns:
            if pn.lower() not in seen_names and len(pn) > 3:
                seen_names.add(pn.lower())
                eid = hashlib.md5(pn.lower().encode()).hexdigest()[:12]
                entities.append(Entity(
                    id=f"ent_{eid}",
                    name=pn,
                    entity_type="concept",
                ))

        return entities[:100]  # Cap at 100 per document

    def extract_relationships(
        self, text: str, entities: list[Entity]
    ) -> list[Relationship]:
        """
        Extract relationships between entities using:
        1. Co-occurrence in the same sentence
        2. Verb pattern matching for typed relationships
        """
        relationships = []
        entity_names = {e.name.lower(): e for e in entities}

        sentences = re.split(r'(?<=[.!?])\s+', text)
        rel_id = 0

        for sentence in sentences:
            s_lower = sentence.lower()
            present = [e for name, e in entity_names.items() if name in s_lower]

            # Verb-pattern relationships
            for pattern, rel_type in self.RELATION_PATTERNS:
                for match in pattern.finditer(sentence):
                    subj, obj = match.group(1).lower(), match.group(2).lower()
                    subj_ent = entity_names.get(subj)
                    obj_ent = entity_names.get(obj)
                    if subj_ent and obj_ent and subj_ent.id != obj_ent.id:
                        rel_id += 1
                        relationships.append(Relationship(
                            id=f"rel_{rel_id:06d}",
                            source_id=subj_ent.id,
                            target_id=obj_ent.id,
                            relation_type=rel_type,
                            weight=1.5,
                        ))

            # Co-occurrence relationships (lower weight)
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    rel_id += 1
                    relationships.append(Relationship(
                        id=f"rel_{rel_id:06d}",
                        source_id=present[i].id,
                        target_id=present[j].id,
                        relation_type="co_occurs",
                        weight=1.0,
                    ))

        return relationships


# ── Knowledge Graph Store ──────────────────────────────────

class KnowledgeGraph:
    """
    SQLite-backed knowledge graph with entity and relationship management.
    """

    def __init__(self, db_path: str = GRAPH_DB_PATH):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.extractor = EntityExtractor()

    async def _get_db(self) -> aiosqlite.Connection:
        db = await aiosqlite.connect(self.db_path)
        db.row_factory = aiosqlite.Row
        return db

    async def init(self):
        """Initialize graph tables."""
        db = await self._get_db()
        try:
            await db.executescript(GRAPH_SCHEMA)
            await db.commit()
            logger.info("Knowledge graph initialized")
        finally:
            await db.close()

    async def add_entities(self, entities: list[Entity], document_id: str = ""):
        """Add entities to the graph, merging duplicates."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        db = await self._get_db()
        try:
            for entity in entities:
                existing = await db.execute(
                    "SELECT id, mention_count FROM entities WHERE name = ?",
                    (entity.name,),
                )
                row = await existing.fetchone()
                if row:
                    await db.execute(
                        "UPDATE entities SET mention_count = mention_count + 1, updated_at = ? WHERE id = ?",
                        (now, row["id"]),
                    )
                else:
                    await db.execute(
                        """INSERT INTO entities (id, name, entity_type, description,
                           source_document_id, mention_count, metadata, created_at, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (entity.id, entity.name, entity.entity_type,
                         entity.description, document_id, 1,
                         json.dumps(entity.metadata), now, now),
                    )
            await db.commit()
        finally:
            await db.close()

    async def add_relationships(self, relationships: list[Relationship], document_id: str = ""):
        """Add relationships to the graph."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        db = await self._get_db()
        try:
            for rel in relationships:
                await db.execute(
                    """INSERT OR IGNORE INTO relationships
                       (id, source_entity_id, target_entity_id, relation_type,
                        weight, description, source_document_id, metadata, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, '{}', ?)""",
                    (rel.id, rel.source_id, rel.target_id, rel.relation_type,
                     rel.weight, rel.description, document_id, now),
                )
            await db.commit()
        finally:
            await db.close()

    async def extract_and_store(self, text: str, document_id: str = ""):
        """Extract entities and relationships from text and store them."""
        entities = self.extractor.extract(text, document_id)
        relationships = self.extractor.extract_relationships(text, entities)

        if entities:
            await self.add_entities(entities, document_id)
        if relationships:
            await self.add_relationships(relationships, document_id)

        logger.info(
            f"Knowledge graph: extracted {len(entities)} entities, "
            f"{len(relationships)} relationships from doc {document_id}"
        )
        return {"entities": len(entities), "relationships": len(relationships)}

    async def search_entities(self, query: str, limit: int = 20) -> list[dict]:
        """Search entities by name."""
        db = await self._get_db()
        try:
            cursor = await db.execute(
                "SELECT * FROM entities WHERE name LIKE ? ORDER BY mention_count DESC LIMIT ?",
                (f"%{query}%", limit),
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]
        finally:
            await db.close()

    async def get_neighbors(self, entity_id: str, max_depth: int = 1) -> dict:
        """Get entity and its immediate neighbors."""
        db = await self._get_db()
        try:
            # Get entity
            cursor = await db.execute("SELECT * FROM entities WHERE id = ?", (entity_id,))
            entity = await cursor.fetchone()
            if not entity:
                return {"entity": None, "neighbors": []}

            # Get relationships
            cursor = await db.execute(
                """SELECT r.*, e.name as target_name, e.entity_type as target_type
                   FROM relationships r
                   JOIN entities e ON r.target_entity_id = e.id
                   WHERE r.source_entity_id = ?
                   UNION
                   SELECT r.*, e.name as target_name, e.entity_type as target_type
                   FROM relationships r
                   JOIN entities e ON r.source_entity_id = e.id
                   WHERE r.target_entity_id = ?
                   LIMIT 50""",
                (entity_id, entity_id),
            )
            neighbors = await cursor.fetchall()

            return {
                "entity": dict(entity),
                "neighbors": [dict(n) for n in neighbors],
            }
        finally:
            await db.close()

    async def get_graph_stats(self) -> dict:
        """Get knowledge graph statistics."""
        db = await self._get_db()
        try:
            entities_cursor = await db.execute("SELECT COUNT(*) as cnt FROM entities")
            rels_cursor = await db.execute("SELECT COUNT(*) as cnt FROM relationships")
            types_cursor = await db.execute(
                "SELECT entity_type, COUNT(*) as cnt FROM entities GROUP BY entity_type"
            )
            entities = await entities_cursor.fetchone()
            rels = await rels_cursor.fetchone()
            types = await types_cursor.fetchall()
            return {
                "entity_count": entities["cnt"] if entities else 0,
                "relationship_count": rels["cnt"] if rels else 0,
                "entity_types": {r["entity_type"]: r["cnt"] for r in types},
            }
        finally:
            await db.close()

    async def get_graph_data(self, limit: int = 200) -> dict:
        """Get graph data for visualization (nodes + edges)."""
        db = await self._get_db()
        try:
            nodes_cursor = await db.execute(
                "SELECT id, name, entity_type, mention_count FROM entities ORDER BY mention_count DESC LIMIT ?",
                (limit,),
            )
            edges_cursor = await db.execute(
                """SELECT source_entity_id, target_entity_id, relation_type, weight
                   FROM relationships LIMIT ?""",
                (limit * 3,),
            )
            nodes = [dict(r) for r in await nodes_cursor.fetchall()]
            edges = [dict(r) for r in await edges_cursor.fetchall()]

            # Filter edges to only include visible nodes
            node_ids = set(n["id"] for n in nodes)
            edges = [e for e in edges if e["source_entity_id"] in node_ids and e["target_entity_id"] in node_ids]

            return {"nodes": nodes, "edges": edges}
        finally:
            await db.close()

    # ── NetworkX Graph Analysis ────────────────────────────────

    async def build_networkx_graph(self) -> nx.Graph:
        """Build a NetworkX graph from the SQLite data for analysis."""
        G = nx.Graph()
        db = await self._get_db()
        try:
            nodes_cursor = await db.execute(
                "SELECT id, name, entity_type, mention_count FROM entities"
            )
            for row in await nodes_cursor.fetchall():
                r = dict(row)
                G.add_node(r["id"], name=r["name"], entity_type=r["entity_type"],
                           mention_count=r["mention_count"])

            edges_cursor = await db.execute(
                "SELECT source_entity_id, target_entity_id, relation_type, weight FROM relationships"
            )
            for row in await edges_cursor.fetchall():
                r = dict(row)
                G.add_edge(r["source_entity_id"], r["target_entity_id"],
                           relation_type=r["relation_type"], weight=r["weight"])
            return G
        finally:
            await db.close()

    async def get_graph_analysis(self) -> dict:
        """Run graph algorithms: PageRank, centrality, connected components."""
        G = await self.build_networkx_graph()
        if G.number_of_nodes() == 0:
            return {"pagerank": {}, "centrality": {}, "communities": [], "hub_entities": []}

        # PageRank
        try:
            pagerank = nx.pagerank(G, weight="weight")
        except Exception:
            pagerank = {}

        # Betweenness centrality
        try:
            centrality = nx.betweenness_centrality(G, weight="weight")
        except Exception:
            centrality = {}

        # Connected components as communities
        communities = []
        for i, component in enumerate(nx.connected_components(G)):
            members = []
            for node_id in component:
                data = G.nodes[node_id]
                members.append({"id": node_id, "name": data.get("name", "")})
            communities.append({"id": i, "size": len(component), "members": members[:20]})
        communities.sort(key=lambda c: c["size"], reverse=True)

        # Hub entities (top by PageRank)
        hub_entities = []
        if pagerank:
            top = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:20]
            for node_id, score in top:
                data = G.nodes[node_id]
                hub_entities.append({
                    "id": node_id,
                    "name": data.get("name", ""),
                    "entity_type": data.get("entity_type", ""),
                    "pagerank": round(score, 6),
                    "centrality": round(centrality.get(node_id, 0), 6),
                    "degree": G.degree(node_id),
                })

        return {
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "communities": communities[:20],
            "hub_entities": hub_entities,
            "density": round(nx.density(G), 6) if G.number_of_nodes() > 1 else 0,
        }

    async def get_related_context(self, query: str, max_entities: int = 10) -> str:
        """
        Graph-augmented retrieval: find entities related to a query
        and return their context as additional information for RAG.
        """
        # Search for entities matching query terms
        query_words = [w.lower() for w in query.split() if len(w) > 2]
        matched_entities = []

        db = await self._get_db()
        try:
            for word in query_words[:10]:
                cursor = await db.execute(
                    "SELECT id, name, entity_type, description, mention_count FROM entities WHERE name LIKE ? LIMIT 5",
                    (f"%{word}%",),
                )
                for row in await cursor.fetchall():
                    matched_entities.append(dict(row))

            if not matched_entities:
                return ""

            # Deduplicate
            seen_ids = set()
            unique = []
            for e in matched_entities:
                if e["id"] not in seen_ids:
                    seen_ids.add(e["id"])
                    unique.append(e)
            matched_entities = unique[:max_entities]

            # Get neighbors for matched entities
            all_entity_ids = set(e["id"] for e in matched_entities)
            neighbor_info = []
            for entity in matched_entities:
                cursor = await db.execute(
                    """SELECT e.name, r.relation_type
                       FROM relationships r
                       JOIN entities e ON (r.target_entity_id = e.id AND r.source_entity_id = ?)
                       UNION
                       SELECT e.name, r.relation_type
                       FROM relationships r
                       JOIN entities e ON (r.source_entity_id = e.id AND r.target_entity_id = ?)
                       LIMIT 10""",
                    (entity["id"], entity["id"]),
                )
                neighbors = await cursor.fetchall()
                if neighbors:
                    rel_strs = [f"{n['name']} ({n['relation_type']})" for n in neighbors]
                    neighbor_info.append(
                        f"- {entity['name']} [{entity['entity_type']}]: related to {', '.join(rel_strs)}"
                    )
                else:
                    neighbor_info.append(
                        f"- {entity['name']} [{entity['entity_type']}] (mentioned {entity['mention_count']}x)"
                    )

            if not neighbor_info:
                return ""

            context = "Knowledge Graph Context:\n" + "\n".join(neighbor_info[:max_entities])
            return context
        finally:
            await db.close()

    async def get_entity_timeline(self, limit: int = 50) -> list[dict]:
        """Get recently added/updated entities for the memory inspector."""
        db = await self._get_db()
        try:
            cursor = await db.execute(
                "SELECT id, name, entity_type, mention_count, source_document_id, created_at, updated_at "
                "FROM entities ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            )
            return [dict(r) for r in await cursor.fetchall()]
        finally:
            await db.close()

    async def delete_entity(self, entity_id: str):
        """Delete an entity and its relationships."""
        db = await self._get_db()
        try:
            await db.execute("DELETE FROM relationships WHERE source_entity_id = ? OR target_entity_id = ?",
                             (entity_id, entity_id))
            await db.execute("DELETE FROM entities WHERE id = ?", (entity_id,))
            await db.commit()
        finally:
            await db.close()

    async def clear_graph(self):
        """Clear all entities and relationships."""
        db = await self._get_db()
        try:
            await db.execute("DELETE FROM relationships")
            await db.execute("DELETE FROM entities")
            await db.commit()
        finally:
            await db.close()


# Singleton
knowledge_graph = KnowledgeGraph()
