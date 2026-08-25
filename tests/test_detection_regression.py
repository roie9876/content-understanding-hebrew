import unittest

from detection_regression import score_detection


class DetectionRegressionTests(unittest.TestCase):
    def test_score_accepts_detection_only_targets(self):
        golden = {"regions": [{
            "label": "field",
            "box": [100, 100, 100, 50],
            "detection_target": True,
        }]}

        report = score_detection(
            golden,
            [{"label": "field", "box": [100, 100, 100, 50]}],
            (1000, 1000),
        )

        self.assertTrue(report["pass"])
        self.assertEqual(1, report["target_count"])

    def test_score_passes_complete_local_detection(self):
        golden = {"regions": [
            {"label": "first", "box": [100, 100, 100, 50], "final_transcription": "א"},
            {"label": "second", "box": [500, 500, 100, 50], "final_transcription": "ב"},
            {"label": "signature", "box": [700, 700, 100, 50],
             "final_transcription": "", "human_classification": "signature_non_text"},
        ]}
        detected = [
            {"label": "one", "box": [105, 105, 100, 50]},
            {"label": "two", "box": [495, 495, 100, 50]},
            {"label": "signature", "box": [700, 700, 100, 50]},
        ]

        report = score_detection(golden, detected, (1000, 1000))

        self.assertTrue(report["pass"])
        self.assertEqual(2, report["matched_count"])

    def test_score_fails_distant_or_missing_detection(self):
        golden = {"regions": [
            {"label": "first", "box": [100, 100, 100, 50], "final_transcription": "א"},
            {"label": "second", "box": [500, 500, 100, 50], "final_transcription": "ב"},
        ]}
        detected = [{"label": "wrong", "box": [850, 850, 100, 50]}]

        report = score_detection(golden, detected, (1000, 1000))

        self.assertFalse(report["pass"])
        self.assertLess(report["matched_count"], report["target_count"])

    def test_score_allows_tight_crop_for_overlapping_golden_targets(self):
        golden = {"regions": [
            {"label": "combined", "box": [100, 100, 500, 200],
             "final_transcription": "א ב"},
            {"label": "specific", "box": [250, 120, 200, 160],
             "final_transcription": "ב"},
        ]}
        detected = [{"label": "tight", "box": [270, 140, 140, 100]}]

        report = score_detection(golden, detected, (1000, 1000))

        self.assertTrue(report["pass"])
        self.assertEqual(2, report["matched_count"])
        self.assertEqual(0, report["extra_count"])

    def test_score_allows_offset_tight_crop_inside_broad_target(self):
        golden = {"regions": [{
            "label": "broad handwritten margin note",
            "box": [309, 0, 466, 93],
            "final_transcription": "בסד",
        }]}
        detected = [{
            "label": "tight handwritten margin note",
            "box": [672, 18, 117, 58],
        }]

        report = score_detection(golden, detected, (1008, 943))

        self.assertTrue(report["pass"])
        self.assertEqual(1, report["matched_count"])
        self.assertGreater(report["rows"][0]["center_distance"], 0.12)


if __name__ == "__main__":
    unittest.main()