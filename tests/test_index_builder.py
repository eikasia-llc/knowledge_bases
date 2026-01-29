import unittest
import sqlite3
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

# Adjust path to import src
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from index_builder import IndexBuilder, IndexEntry
from md_parser import Node

class TestIndexBuilder(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = str(Path(self.test_dir) / "test_index.db")
        
        # Mocking the embedding model class
        self.patcher = patch("index_builder.SentenceTransformer")
        self.MockModel = self.patcher.start()
        self.mock_model_instance = MagicMock()
        self.MockModel.return_value = self.mock_model_instance
        
        # Mock encode to return dummy embeddings
        self.mock_model_instance.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        self.builder = IndexBuilder(db_path=self.db_path)

    def tearDown(self):
        self.patcher.stop()
        shutil.rmtree(self.test_dir)

    def test_initialize_db(self):
        self.builder.initialize_db()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            self.assertIn("knowledge_nodes", tables)

    def test_process_file(self):
        # Create a dummy MD file
        md_path = Path(self.test_dir) / "test.md"
        with open(md_path, "w") as f:
            f.write("# Test Title\n- id: test.node\n- type: guideline\n<!-- content -->\nThis is test content.")
        
        entries = self.builder.process_file(md_path)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].id, "test.node")
        self.assertEqual(entries[0].title, "Test Title")
        self.assertEqual(entries[0].content, "This is test content.")

    def test_build_index_and_save(self):
        # 1. Setup file
        md_path = Path(self.test_dir) / "test.md"
        with open(md_path, "w") as f:
            f.write("# Test Title\n- id: test.node\n<!-- content -->\nContent")
            
        # 2. Build Index (mocks scan_directory to just return our file)
        with patch.object(self.builder, 'scan_directory', return_value=[md_path]):
            self.builder.build_index(self.test_dir)
            
        # 3. Verify DB content
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, title, embedding FROM knowledge_nodes")
            row = cursor.fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row[0], "test.node")
            self.assertEqual(row[1], "Test Title")
            
            # Check embedding was stored
            emb_blob = row[2]
            emb_arr = np.frombuffer(emb_blob, dtype=np.float32)
            self.assertTrue(np.allclose(emb_arr, [0.1, 0.2, 0.3]))

if __name__ == '__main__':
    unittest.main()
