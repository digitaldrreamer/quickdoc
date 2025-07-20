import unittest
from sentence_transformers import SentenceTransformer

class TestModelLoading(unittest.TestCase):
    def test_load_all_mini_lm_l6_v2(self):
        try:
            SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            self.fail(f"Failed to load all-MiniLM-L6-v2 model: {e}")

if __name__ == '__main__':
    unittest.main()