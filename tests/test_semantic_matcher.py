import unittest
import sqlite3
import shutil
import tempfile
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from semantic_matcher import SemanticMatcher

class TestSemanticMatcher(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = str(Path(self.test_dir) / "test_index.db")
        
        # Setup Mock DB with some logical data
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE knowledge_nodes (
                    id TEXT, path TEXT, title TEXT, type TEXT, status TEXT, content TEXT, 
                    content_hash TEXT, metadata TEXT, parent_id TEXT, embedding BLOB
                )
            """)
            
            # Helper to insert
            def insert(id, type, status, embedding):
                emb_bytes = np.array(embedding, dtype=np.float32).tobytes()
                conn.execute(
                    "INSERT INTO knowledge_nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (id, f"path/{id}", f"Title {id}", type, status, "Content", "hash", "{}", None, emb_bytes)
                )

            # Insert Scenarios
            # Query will be [1, 0, 0]
            # 1. Guideline (Active) - Ideal match
            insert("guideline_active", "guideline", "active", [1.0, 0.0, 0.0])
            
            # 2. Protocol (Draft) - Ideal match but lower status
            insert("protocol_draft", "protocol", "draft", [1.0, 0.0, 0.0])
            
            # 3. Context (Active) - Poor semantic match
            insert("context_poor", "context", "active", [0.0, 1.0, 0.0])

        self.patcher = patch("semantic_matcher.SentenceTransformer")
        self.MockModel = self.patcher.start()
        self.mock_model_instance = MagicMock()
        self.MockModel.return_value = self.mock_model_instance
        
        # Determine behavior based on input
        def mock_encode(query, *args, **kwargs):
             return np.array([1.0, 0.0, 0.0], dtype=np.float32)

        self.mock_model_instance.encode.side_effect = mock_encode

        self.matcher = SemanticMatcher(db_path=self.db_path)

    def tearDown(self):
        self.patcher.stop()
        shutil.rmtree(self.test_dir)

    def test_search_basic(self):
        results = self.matcher.search("query")
        # All 3 might be returned as limit defaults to 5
        self.assertEqual(len(results), 3) 
        
        # Check top result
        self.assertEqual(results[0].id, "guideline_active")
        self.assertAlmostEqual(results[0].relevance_score, 1.1, places=1) # 1.0 * 1.1 active boost
        
        # Check last result (should be near 0)
        self.assertEqual(results[2].id, "context_poor")
        self.assertLess(results[2].relevance_score, 0.1)
        
    def test_boosting_implement(self):
        # 'implement' heavily boosts 'guideline' (1.5) over 'protocol' (1.2)
        # guideline_active base score = 1.0 * 1.1 (status) = 1.1
        # protocol_draft base score = 1.0 * 1.0 (status) = 1.0
        
        # With boost:
        # guideline: 1.1 * 1.5 = 1.65
        # protocol: 1.0 * 1.2 = 1.2
        
        results = self.matcher.search("query", task_type="implement")
        self.assertEqual(results[0].id, "guideline_active")
        self.assertTrue(results[0].relevance_score > 1.5)

    def test_boosting_debug(self):
        # 'debug' boosts 'protocol' (1.3) and 'context' (1.5)
        # guideline (1.0)
        
        # guideline_active: 1.1 * 1.0 = 1.1
        # protocol_draft: 1.0 * 1.3 = 1.3
        
        # So protocol_draft should win despite being draft, IF the boosting outweighs status
        results = self.matcher.search("query", task_type="debug")
        self.assertEqual(results[0].id, "protocol_draft")

if __name__ == '__main__':
    unittest.main()
