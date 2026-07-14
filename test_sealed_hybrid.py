import unittest
from unittest.mock import patch
import tempfile
import json
from pathlib import Path

import sealed_hybrid_benchmark


class SealedHybridTests(unittest.TestCase):
    def test_candidate_profiles_are_openai_only(self):
        self.assertEqual(
            ("gpt-5.4", "gpt-5.6-sol"),
            sealed_hybrid_benchmark.CANDIDATE_PROFILES,
        )

    def test_sol_environment_uses_low_reasoning_without_endpoint_pool(self):
        with patch.dict(sealed_hybrid_benchmark.os.environ, {
            "GPT_SOL_ENDPOINT": "https://sol.example/openai/v1/",
            "GPT_SOL_DEPLOYMENT": "gpt-5.6-sol",
            "GPT_TARGETS": "private-pool",
        }, clear=False):
            environment = sealed_hybrid_benchmark._model_environment("gpt-5.6-sol")

        self.assertEqual("https://sol.example/openai/v1/", environment["GPT_ENDPOINT"])
        self.assertEqual("low", environment["OCR_REASONING"])
        self.assertEqual("", environment["GPT_TARGETS"])

    def test_sealed_request_rejects_evaluation_leakage(self):
        with self.assertRaisesRegex(RuntimeError, "forbidden evaluation terms"):
            sealed_hybrid_benchmark._assert_sealed({"expected": "secret answer"})

    def test_sealed_request_accepts_model_candidates(self):
        sealed_hybrid_benchmark._assert_sealed({
            "regions": [{
                "neutral_label": "handwriting region 1",
                "primary": {"transcription": "abc"},
                "secondary": {"transcription": "abd"},
            }]
        })

    def test_nearest_regions_matches_reordered_boxes(self):
        anchors = [{"box": [0, 0, 10, 10]}, {"box": [100, 0, 10, 10]}]
        others = [
            {"box": [100, 0, 10, 10], "value": "right"},
            {"box": [0, 0, 10, 10], "value": "left"},
        ]

        matches = sealed_hybrid_benchmark._nearest_regions(anchors, others)

        self.assertEqual(["left", "right"], [region["value"] for region in matches])

    def test_evaluation_report_ranks_exact_accuracy_first(self):
        def score(exact, similarity):
            return {
                "strict": {
                    "text_exact": exact,
                    "text_total": 20,
                    "text_accuracy": exact / 20,
                    "classification_accuracy": 1.0,
                },
                "avg_ratio": similarity,
            }

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            sealed_hybrid_benchmark._write_evaluation_report(run_dir, {
                "slower": score(9, 0.9),
                "more exact": score(11, 0.8),
            })
            summary = json.loads((run_dir / "evaluation_summary.json").read_text())

        self.assertEqual(["more exact", "slower"], summary["ranking"])


if __name__ == "__main__":
    unittest.main()