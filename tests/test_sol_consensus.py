import unittest

import sol_consensus


def region(x, text, confidence, classification="handwritten_text"):
    return {
        "box": [x, 0, 10, 10],
        "final_transcription": text,
        "classification": classification,
        "confidence": confidence,
    }


class SolConsensusTests(unittest.TestCase):
    def test_majority_wins_over_higher_confidence_outlier(self):
        result = sol_consensus.build_consensus([
            {"regions": [region(0, "5252 731", 0.7)]},
            {"regions": [region(0, "5252731", 0.8)]},
            {"regions": [region(0, "6252731", 0.99)]},
        ])

        selected = result["regions"][0]
        self.assertEqual("5252731", selected["final_transcription"])
        self.assertEqual("majority", selected["consensus"]["method"])

    def test_all_disagree_uses_highest_confidence(self):
        result = sol_consensus.build_consensus([
            {"regions": [region(0, "one", 0.5)]},
            {"regions": [region(0, "two", 0.9)]},
            {"regions": [region(0, "three", 0.7)]},
        ])

        selected = result["regions"][0]
        self.assertEqual("two", selected["final_transcription"])
        self.assertEqual("confidence_fallback", selected["consensus"]["method"])

    def test_spatial_alignment_handles_reordered_regions(self):
        result = sol_consensus.build_consensus([
            {"regions": [region(0, "left", 0.8), region(100, "right", 0.8)]},
            {"regions": [region(100, "right", 0.8), region(0, "left", 0.8)]},
            {"regions": [region(0, "left", 0.8), region(100, "right", 0.8)]},
        ])

        self.assertEqual(["left", "right"], [item["final_transcription"] for item in result["regions"]])


if __name__ == "__main__":
    unittest.main()