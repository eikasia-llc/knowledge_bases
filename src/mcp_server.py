import asyncio
import json
import logging
import sqlite3
import argparse
from typing import List, Optional
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

# Local imports
try:
    from .semantic_matcher import SemanticMatcher
    from .index_builder import IndexBuilder
    from .dependency_manager import DependencyManager
except ImportError:
    from semantic_matcher import SemanticMatcher
    from index_builder import IndexBuilder
    from dependency_manager import DependencyManager

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("meta_mcp_server")

# Initialize FastMCP Server
mcp = FastMCP("meta-knowledge-mcp")

# Globals (lazy loaded)
matcher: Optional[SemanticMatcher] = None
builder: Optional[IndexBuilder] = None
dep_manager: Optional[DependencyManager] = None

def get_matcher():
    global matcher
    if matcher is None:
        matcher = SemanticMatcher()
    return matcher

def get_builder():
    global builder
    if builder is None:
        builder = IndexBuilder()
    return builder

def get_dep_manager():
    global dep_manager
    if dep_manager is None:
        dep_manager = DependencyManager()
    return dep_manager

# --- Tools ---

@mcp.tool()
async def discover_context(
    task_description: str = Field(..., description="Natural language description of the task"),
    task_type: str = Field("implement", description="Type of task: implement, debug, refactor, etc."),
    max_results: int = Field(5, description="Maximum number of recommendations to return")
) -> dict:
    """Analyzes a task and returns ranked recommendations of relevant knowledge bases."""
    logger.info(f"Discovering context for: {task_description[:50]}...")
    
    match_engine = get_matcher()
    results = match_engine.search(task_description, task_type=task_type, limit=max_results)
    
    return {
        "recommendations": [
            {
                "id": r.id,
                "path": r.path,
                "title": r.title,
                "type": r.type,
                "relevance_score": r.relevance_score,
                "reason": r.reason,
                "preview": r.content_preview
            }
            for r in results
        ],
        "total_found": len(results)
    }

@mcp.tool()
async def retrieve_knowledge(
    ids: List[str] = Field(..., description="List of node IDs to retrieve"),
    include_children: bool = Field(False, description="Whether to include child nodes")
) -> dict:
    """Fetches the full content of one or more knowledge base nodes."""
    logger.info(f"Retrieving nodes: {ids}")
    
    # Simple retrieval via SQLite for now
    # Ideally should use DependencyManager or direct FS read if freshness is key
    # But DB is faster for random access
    
    nodes = []
    with sqlite3.connect(".meta_mcp_index.db") as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        for node_id in ids:
            cursor.execute("SELECT id, path, title, type, content, metadata FROM knowledge_nodes WHERE id = ?", (node_id,))
            row = cursor.fetchone()
            if row:
                node = {
                    "id": row['id'],
                    "path": row['path'],
                    "title": row['title'],
                    "type": row['type'],
                    "content": row['content'],
                    "metadata": json.loads(row['metadata']) if row['metadata'] else {}
                }
                
                if include_children:
                    # Fetch immediate children
                    cursor.execute("SELECT id, title, type FROM knowledge_nodes WHERE parent_id = ?", (node_id,))
                    children = cursor.fetchall()
                    node["children"] = [{"id": c['id'], "title": c['title'], "type": c['type']} for c in children]
                    
                nodes.append(node)
                
    return {
        "nodes": nodes,
        "count": len(nodes)
    }

@mcp.tool()
async def search_knowledge(
    query: str = Field(..., description="Search query"),
    max_results: int = 10
) -> dict:
    """Performs semantic search across knowledge bases."""
    match_engine = get_matcher()
    results = match_engine.search(query, limit=max_results)
    
    return {
        "results": [
            {
                "id": r.id,
                "title": r.title,
                "path": r.path,
                "score": r.relevance_score
            }
            for r in results
        ]
    }

@mcp.tool()
async def list_knowledge_bases(
    filter_type: Optional[str] = None,
    filter_status: Optional[str] = None
) -> dict:
    """Lists available knowledge base nodes with filtering."""
    query = "SELECT id, title, type, status, path FROM knowledge_nodes"
    params = []
    conditions = []
    
    if filter_type:
        conditions.append("type = ?")
        params.append(filter_type)
    if filter_status:
        conditions.append("status = ?")
        params.append(filter_status)
        
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
        
    query += " LIMIT 50" # Safety cap
    
    items = []
    with sqlite3.connect(".meta_mcp_index.db") as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        
        for r in rows:
            items.append({
                "id": r['id'],
                "title": r['title'],
                "type": r['type'],
                "status": r['status'],
                "path": r['path']
            })
            
    return {"knowledge_bases": items, "count": len(items)}

@mcp.tool()
async def rebuild_index(
    full_rebuild: bool = False
) -> str:
    """Triggers a rebuild of the knowledge base index."""
    idx = get_builder()
    # Assuming the builder handles finding the root
    # For this implementation we'll assume current dir or hardcoded root
    # Ideally should be config driven
    idx.build_index(".") 
    return "Index rebuild complete."

# --- Main Entry ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", action="store_true", help="Run indexer on startup")
    args = parser.parse_args()
    
    if args.index:
        logger.info("Building index...")
        get_builder().build_index(".")
        
    # Run server
    mcp.run()
