import json
import tempfile
import unittest
from pathlib import Path

import evaluate_golden as evaluation


class GoldenEvaluationTests(unittest.TestCase):
    def test_edit_metrics_count_substitutions_insertions_and_deletions(self):
        metrics = evaluation.text_metrics("abc 123", "axc 124 extra")

        self.assertFalse(metrics["exact"])
        self.assertEqual(7, metrics["character_errors"])
        self.assertEqual(3, metrics["word_errors"])
        self.assertEqual(6, metrics["reference_characters"])

    def test_markup_and_comments_are_not_transcription_evidence(self):
        text = evaluation.plain_text(
            '<div>Reference <span style="color:red">12345</span></div>'
            '<!-- hidden answer 98765 -->',
        )

        self.assertIn("12345", text)
        self.assertNotIn("color", text)
        self.assertNotIn("98765", text)
        self.assertTrue(evaluation.value_present("12345", text))
        self.assertFalse(evaluation.value_present("98765", text))

    def test_numeric_presence_respects_boundaries_and_grouped_digits(self):
        self.assertFalse(evaluation.value_present("20", "2026 and 2000"))
        self.assertTrue(evaluation.value_present("20", "20/534/12345"))
        self.assertTrue(evaluation.value_present("12345", "12 345"))
        self.assertFalse(evaluation.value_present("12345", "123456"))
        with self.assertRaises(ValueError):
            evaluation.value_present("", "anything")

    def test_document_date_checks_do_not_hide_a_wrong_year_behind_formatting(self):
        self.assertTrue(evaluation.date_value_present("22/9/26", "Date: 22.09.26"))
        self.assertFalse(evaluation.date_value_present("22/9/96", "Date: 22/09/26"))
        self.assertFalse(evaluation.date_value_present("22/9/26", "Date: 22/9/2026"))
        self.assertIsNone(evaluation.date_value_present("22/9/?", "Date: 22/9/26"))

    def test_internal_relationship_indexes_are_not_treated_as_visible_numbers(self):
        result = evaluation.score_document({
            "literal_transcription": "Account 12345",
            "expected": {"transfers": [{"source_account_index": 0, "account_number": "12345"}]},
            "critical_fields": [
                "transfers[0].source_account_index", "transfers[0].account_number",
            ],
        }, "Account 12345")

        self.assertEqual(1, result["critical_values_present"])
        self.assertEqual(1, result["critical_values_total"])
        self.assertEqual(1, len(result["relationship_expectations_not_scored"]))
    def test_discovery_supports_both_reference_schemas(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pdf"
            source.touch()
            for stem, name, metadata in (
                ("regions", "handwriting_result.json", {"source": str(source), "page": 2}),
                ("document", "golden.json", {"document": {"file": str(source)}}),
            ):
                folder = root / stem
                folder.mkdir()
                (folder / name).write_text(json.dumps(metadata))

            samples = evaluation.discover_samples(root)

        self.assertEqual(["document", "regions"], [sample.stem for sample in samples])
        self.assertEqual([1, 2], [sample.page for sample in samples])

    def test_distant_prediction_does_not_count_as_a_matched_field(self):
        golden = {"regions": [{"label": "field", "box": [0, 0, 10, 10], "final_transcription": "123"}]}
        current = {"regions": [{"box": [90, 90, 10, 10], "final_transcription": "123"}]}

        result = evaluation.score_regions(golden, current, (100, 100))

        self.assertEqual(0, result["exact_regions"])
        self.assertEqual(1, result["missing_text_regions"])

    def test_signature_omission_is_distinguished_from_text_hallucination(self):
        golden = {"regions": [{
            "label": "signature", "box": [10, 10, 20, 10],
            "final_transcription": "", "human_classification": "signature_non_text",
        }]}
        omitted = evaluation.score_regions(golden, {"regions": []}, (100, 100))
        transcribed = evaluation.score_regions(golden, {"regions": [{
            "box": [10, 10, 20, 10], "final_transcription": "invented name",
            "classification": "handwritten_text",
        }]}, (100, 100))

        self.assertEqual(1, omitted["non_text_safely_excluded"])
        self.assertEqual(0, transcribed["non_text_safely_excluded"])

    def test_word_inventory_does_not_punish_a_combined_paragraph_for_segmentation(self):
        golden = {"regions": [
            {"box": [0, 0, 30, 20], "final_transcription": "alpha"},
            {"box": [0, 30, 30, 20], "final_transcription": "beta"},
        ]}
        result = evaluation.score_regions(golden, {"regions": [{
            "box": [0, 0, 30, 50], "final_transcription": "alpha beta",
        }]}, (100, 100))

        self.assertEqual(1.0, result["word_inventory"]["precision"])
        self.assertEqual(1.0, result["word_inventory"]["recall"])

    def test_numeric_groups_distinguish_values_from_order_and_segmentation(self):
        golden = {"regions": [{
            "box": [0, 0, 80, 20], "final_transcription": "123 456",
        }]}
        reordered = evaluation.score_regions(golden, {"regions": [{
            "box": [0, 0, 80, 20], "final_transcription": "456 123",
        }]}, (100, 100))
        regrouped = evaluation.score_regions(golden, {"regions": [{
            "box": [0, 0, 80, 20], "final_transcription": "12 3456",
        }]}, (100, 100))

        self.assertEqual(1, reordered["numeric_groups_same_values"])
        self.assertEqual(0, reordered["numeric_groups_same_order"])
        self.assertEqual(0, regrouped["numeric_groups_same_values"])
        self.assertTrue(regrouped["rows"][0]["exact"])

    def test_output_cannot_overwrite_golden_references(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            golden = root / "golden"
            golden.mkdir()
            for output in (root, golden, golden / "sample"):
                with self.subTest(output=output), self.assertRaises(ValueError):
                    evaluation.validate_output_location(output, golden)
            evaluation.validate_output_location(root / "results", golden)

    def test_failed_model_execution_is_not_scored_as_a_quality_regression(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.json"
            reference.write_text(json.dumps({"regions": []}))
            sample = evaluation.Sample("sample", root / "source.pdf", reference, 1)
            output = root / "output"
            output.mkdir()
            for calls, log in (
                (0, ""),
                (3, "Tenant provided in token does not match resource token"),
            ):
                with self.subTest(calls=calls):
                    (output / "handwriting_result.json").write_text(json.dumps({
                        "regions": [], "usage": {"calls": calls},
                    }))
                    (output / "pipeline.log").write_text(log)
                    with self.assertRaises(ValueError):
                        evaluation.score_sample(sample, output)

    def test_documented_successful_merge_recovery_keeps_error_history_and_partial_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.json"
            reference.write_text(json.dumps({
                "literal_transcription": "value", "expected": {}, "critical_fields": [],
            }))
            sample = evaluation.Sample("sample", root / "source.pdf", reference, 1)
            output = root / "output"
            output.mkdir()
            (output / "merged.md").write_text("value")
            (output / "pipeline.log").write_text("ModelAccessError: earlier merge failed")
            (output / "handwriting_result.json").write_text(json.dumps({
                "regions": [], "usage": {"calls": 3, "complete": False},
                "recovery": {
                    "stage": "final_markdown_merge", "merge_succeeded": True,
                    "recognition_reused": True, "fresh_successful_model_calls": 1,
                },
            }))

            result = evaluation.score_sample(sample, output)

        self.assertTrue(result["text"]["exact"])
        self.assertIsNone(result["calls"])
        self.assertFalse(result["usage_complete"])
        self.assertEqual(3, result["successful_calls_lower_bound"])


if __name__ == "__main__":
    unittest.main()
