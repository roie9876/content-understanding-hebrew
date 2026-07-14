import tempfile
import unittest
from pathlib import Path

import crop_agent_benchmark


class CropAgentBenchmarkTests(unittest.TestCase):
    def test_sealed_guard_rejects_golden_and_prior_results(self):
        for forbidden in ("golden/answer.json", "handwriting_result.json", "merged.md", "result.json"):
            with self.subTest(forbidden=forbidden), self.assertRaisesRegex(ValueError, "forbidden"):
                crop_agent_benchmark.assert_sealed({"path": forbidden})

    def test_sealed_guard_accepts_crop_views(self):
        crop_agent_benchmark.assert_sealed({
            "regions": [{"views": [{"path": "/tmp/01_original_crop.png"}]}]
        })

    def test_core_views_include_only_crop_transformations(self):
        self.assertEqual(4, len(crop_agent_benchmark.CORE_VIEWS))
        self.assertNotIn("08_context_highlight.png", crop_agent_benchmark.CORE_VIEWS)
        self.assertNotIn("09_page_highlight.png", crop_agent_benchmark.CORE_VIEWS)


if __name__ == "__main__":
    unittest.main()