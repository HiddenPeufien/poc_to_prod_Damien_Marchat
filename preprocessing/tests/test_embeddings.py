import unittest
import sys
sys.path.insert(0, '/path/to/poc-to-prod-capstone/poc-to-prod-capstone/preprocessing/preprocessing')
import utils


class EmbeddingsTest(unittest.TestCase):
    def test_embed(self):
        embeddings = embed(['hello world'])
        self.assertEqual(embeddings.shape, (1, 768))
