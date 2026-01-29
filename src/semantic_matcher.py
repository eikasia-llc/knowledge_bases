import sqlite3
import json
import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

@dataclass
class SearchResult:
    id: str
    path: str
    title: str
    type: str
    relevance_score: float
    reason: str
    content_preview: str
    metadata: Dict

class SemanticMatcher:
    def __init__(self, db_path: str = ".meta_mcp_index.db", embedding_model: str = "all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.model_name = embedding_model
        self._model = None
        
        # Boost configuration (Task Type -> Node Type -> Multiplier)
        self.boost_matrix = {
            "implement": {"guideline": 1.5, "protocol": 1.2, "context": 1.3, "agent_skill": 1.0},
            "debug":     {"guideline": 1.0, "protocol": 1.3, "context": 1.5, "agent_skill": 0.8},
            "refactor":  {"guideline": 1.3, "protocol": 1.1, "context": 1.4, "agent_skill": 0.9},
            "document":  {"guideline": 1.2, "protocol": 1.0, "context": 1.5, "agent_skill": 0.7},
            "review":    {"guideline": 1.4, "protocol": 1.5, "context": 1.2, "agent_skill": 0.8},
            "design":    {"guideline": 1.3, "protocol": 1.4, "context": 1.5, "agent_skill": 1.1},
            "test":      {"guideline": 1.2, "protocol": 1.3, "context": 1.2, "agent_skill": 1.0}
        }

    @property
    def model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def search(self, query: str, task_type: Optional[str] = None, limit: int = 5) -> List[SearchResult]:
        query_emb = self.model.encode(query)
        
        results = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Fetch all embeddings
            # In production with large datasets, use FAISS or SQLite-vss. 
            # For <10k nodes, brute force numpy is fine (and faster due to overhead).
            cursor.execute("SELECT id, path, title, type, status, content, metadata, embedding FROM knowledge_nodes WHERE embedding IS NOT NULL")
            rows = cursor.fetchall()
            
            if not rows:
                return []

            # Vectorized Cosine Similarity
            ids = []
            row_data = []
            embeddings = []
            
            for r in rows:
                emb_blob = r['embedding']
                if emb_blob:
                    ids.append(r['id'])
                    row_data.append(r)
                    embeddings.append(np.frombuffer(emb_blob, dtype=np.float32))
            
            if not embeddings:
                return []
                
            emb_matrix = np.vstack(embeddings)
            
            # Normalize query and matrix (if not already)
            norm_query = query_emb / np.linalg.norm(query_emb)
            norm_matrix = emb_matrix / np.linalg.norm(emb_matrix, axis=1)[:, np.newaxis]
            
            # Cosine similarity
            scores = np.dot(norm_matrix, norm_query)
            
            # Apply Boosting
            final_scores = []
            for i, score in enumerate(scores):
                r = row_data[i]
                node_type = r['type']
                
                boost = 1.0
                reason = "Semantic match"
                
                # Active status boost
                if r['status'] == 'active':
                    boost *= 1.1
                
                # Task Type Boost
                if task_type and task_type in self.boost_matrix:
                    type_boost = self.boost_matrix[task_type].get(node_type, 1.0)
                    if type_boost > 1.0:
                        boost *= type_boost
                        reason += f" + {task_type}/{node_type} boost"
                
                final_score = score * boost
                final_scores.append((final_score, reason, r))
            
            # Sort and Trim
            final_scores.sort(key=lambda x: x[0], reverse=True)
            top_results = final_scores[:limit]
            
            for score, reason, r in top_results:
                results.append(SearchResult(
                    id=r['id'],
                    path=r['path'],
                    title=r['title'],
                    type=r['type'],
                    relevance_score=float(score),
                    reason=reason,
                    content_preview=r['content'][:200] + "...",
                    metadata=json.loads(r['metadata']) if r['metadata'] else {}
                ))
                
        return results
