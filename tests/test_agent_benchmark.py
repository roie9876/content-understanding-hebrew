import json
import tempfile
import unittest
from pathlib import Path

import agent_benchmark


class AgentBenchmarkTests(unittest.TestCase):
    def test_profile_matrix_contains_sol_reasoning_levels_and_gpt54(self):
        matrix = agent_benchmark.profile_matrix()

        self.assertEqual("low", matrix["sol-low"]["reasoning"])
        self.assertEqual("medium", matrix["sol-medium"]["reasoning"])
        self.assertEqual("high", matrix["sol-high"]["reasoning"])
        self.assertEqual("gpt-5.4", matrix["gpt-5.4-medium"]["deployment"])

    def test_summary_ranks_accuracy_before_speed(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            manifest = {"profiles": ["fast", "accurate"]}
            for profile, accuracy, similarity, elapsed in (
                ("fast", 0.5, 0.8, 10),
                ("accurate", 0.75, 0.7, 100),
            ):
                profile_dir = run_dir / profile
                profile_dir.mkdir()
                (profile_dir / "handwriting_result.json").write_text(json.dumps({
                    "regions": [{"box": [0, 0, 1, 1]}],
                    "elapsed_seconds": elapsed,
                }))
                (profile_dir / "score.json").write_text(json.dumps({
                    "strict": {
                        "text_exact": int(accuracy * 4),
                        "text_total": 4,
                        "text_accuracy": accuracy,
                        "classification_accuracy": 1.0,
                    },
                    "avg_ratio": similarity,
                }))

            summary = agent_benchmark.build_summary(run_dir, manifest)

        self.assertEqual(["accurate", "fast"], summary["ranking"])


if __name__ == "__main__":
    unittest.main()