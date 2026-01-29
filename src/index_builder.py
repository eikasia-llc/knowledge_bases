import os
import sqlite3
import json
import hashlib
from typing import List, Dict, Optional, Generator
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from .md_parser import MarkdownParser, Node
except ImportError:
    from md_parser import MarkdownParser, Node

# Database Schema
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS knowledge_nodes (
    id TEXT PRIMARY KEY,
    path TEXT,
    title TEXT,
    type TEXT,
    status TEXT,
    content TEXT,
    content_hash TEXT,
    metadata TEXT,
    parent_id TEXT,
    embedding BLOB
);

CREATE TABLE IF NOT EXISTS keywords (
    keyword TEXT,
    node_id TEXT,
    PRIMARY KEY (keyword, node_id),
    FOREIGN KEY(node_id) REFERENCES knowledge_nodes(id)
);
"""

@dataclass
class IndexEntry:
    id: str
    path: str
    title: str
    type: str
    status: str
    content: str
    content_hash: str
    metadata: Dict
    parent_id: Optional[str]
    embedding: Optional[List[float]] = None

class IndexBuilder:
    def __init__(self, db_path: str = ".meta_mcp_index.db", embedding_model: str = "all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.model_name = embedding_model
        self.parser = MarkdownParser()
        self._model = None  # Lazy load

    @property
    def model(self):
        if self._model is None:
            print(f"Loading embedding model: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def initialize_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(SCHEMA_SQL)

    def scan_directory(self, root_path: str) -> List[Path]:
        md_files = []
        for root, dirs, files in os.walk(root_path):
            # Skip hidden folders
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for file in files:
                if file.endswith('.md'):
                    md_files.append(Path(root) / file)
        return md_files

    def process_file(self, file_path: Path) -> List[IndexEntry]:
        try:
            root_node = self.parser.parse_file(str(file_path))
            entries = []
            
            # Helper to recursively flatten structure
            def traverse(node: Node, parent_id: Optional[str] = None):
                # Determine ID
                # If metadata has 'id', use it. Otherwise generate one or use title slug
                node_id = node.metadata.get('id')
                if node_id and not isinstance(node_id, str):
                    node_id = str(node_id)
                
                # Sanitize other fields
                node_type = node.metadata.get('type', 'unknown')
                if not isinstance(node_type, str):
                    node_type = str(node_type)
                    
                node_status = node.metadata.get('status', 'unknown')
                if not isinstance(node_status, str):
                    node_status = str(node_status)
                
                # If no explicit ID, we skip usage of this node as a distinct entity 
                # UNLESS it's the root or has substantial content.
                # But per current conventions, valuable nodes should have IDs.
                # Fallback: file_path + anchor
                if not node_id:
                     # Simple slugification for fallback
                    slug = node.title.lower().replace(' ', '-').replace('/', '-')
                    if parent_id:
                        node_id = f"{parent_id}.{slug}"
                    else:
                        node_id = file_path.stem # Root node defaults to filename stem

                # Generate Hash
                content_str = node.content + json.dumps(node.metadata, sort_keys=True)
                content_hash = hashlib.md5(content_str.encode()).hexdigest()

                entry = IndexEntry(
                    id=node_id,
                    path=str(file_path), # We might want to add anchor here later
                    title=node.title,
                    type=node_type,
                    status=node_status,
                    content=node.content,
                    content_hash=content_hash,
                    metadata=node.metadata,
                    parent_id=parent_id
                )
                entries.append(entry)

                for child in node.children:
                    traverse(child, node_id)

            traverse(root_node)
            return entries

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return []

    def build_index(self, root_path: str):
        self.initialize_db()
        files = self.scan_directory(root_path)
        print(f"Found {len(files)} Markdown files.")

        all_entries = []
        for f in files:
            all_entries.extend(self.process_file(f))
        
        print(f"Extracted {len(all_entries)} nodes. Generating embeddings...")
        
        # Batch embedding generation
        texts = [f"{e.title}\n{e.content}" for e in all_entries]
        if texts:
            embeddings = self.model.encode(texts, show_progress_bar=True)
            for i, entry in enumerate(all_entries):
                entry.embedding = embeddings[i].tolist()

        self._save_to_db(all_entries)
        print("Index build complete.")

    def _save_to_db(self, entries: List[IndexEntry]):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Upsert Nodes
            for e in entries:
                emb_bytes = np.array(e.embedding, dtype=np.float32).tobytes() if e.embedding else None
                cursor.execute("""
                    INSERT OR REPLACE INTO knowledge_nodes 
                    (id, path, title, type, status, content, content_hash, metadata, parent_id, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    e.id, 
                    e.path, 
                    e.title, 
                    e.type, 
                    e.status, 
                    e.content, 
                    e.content_hash, 
                    json.dumps(e.metadata), 
                    e.parent_id, 
                    emb_bytes
                ))

            conn.commit()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("root_path", help="Root directory to scan")
    args = parser.parse_args()
    
    builder = IndexBuilder()
    builder.build_index(args.root_path)
