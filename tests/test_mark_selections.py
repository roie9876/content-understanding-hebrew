import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import handwriting_pipeline as pipeline


class MarkSelectionTests(unittest.TestCase):
    def setUp(self):
        self.final = {
            "classification": "mark_or_checkbox",
            "final_transcription": "",
            "marks": ["circle"],
            "notes": [],
            "selection": {
                "kind": "circle", "text": "Option A",
                "visual_evidence": "A blue pen circle encloses this printed option.",
            },
        }
        self.usable = [{
            "source": "core4_multi",
            "ocr": {"full_transcription": "Option A"},
        }]

    def test_circled_printed_value_is_not_discarded_as_body_print(self):
        result = pipeline.finalize_region_semantics(self.final, self.usable)

        self.assertEqual("Option A", result["final_transcription"])
        self.assertEqual("mark_or_checkbox", result["classification"])
        self.assertEqual("selected_mark_value", result["decision_status"])
        self.assertFalse(result["needs_review"])

    def test_unconfirmed_selection_is_explicitly_marked_for_review(self):
        result = pipeline.finalize_mark_selection(self.final, [])

        self.assertTrue(result["needs_review"])
        self.assertEqual("selected_mark_review", result["decision_status"])
        self.assertEqual("Option A", result["final_transcription"])
        self.assertEqual([], self.final["notes"])

    def test_selection_requires_explicit_supported_kind_and_pen_evidence(self):
        for change in (
            {"kind": "printed border"},
            {"visual_evidence": ""},
            {"text": []},
        ):
            with self.subTest(change=change):
                candidate = {
                    **self.final, "selection": {**self.final["selection"], **change},
                }

                result = pipeline.finalize_mark_selection(candidate, self.usable)

                self.assertTrue(result["needs_review"])
                self.assertEqual("", result["final_transcription"])

    def test_printed_value_exception_does_not_transcribe_a_signature(self):
        result = pipeline.finalize_region_semantics(
            {**self.final, "classification": "signature_non_text"},
            [], signature_flourish=True,
        )

        self.assertEqual("", result["final_transcription"])
        self.assertEqual("signature_non_text", result["classification"])

    def test_selected_label_is_not_replaced_by_generic_x_and_digit_description(self):
        result = pipeline.finalize_region_semantics({
            **self.final, "marks": ["circle", "x"], "digits": ["1"],
        }, self.usable)

        self.assertEqual("Option A", result["final_transcription"])
        self.assertEqual(["1"], result["digits"])

    def test_ocr_prompt_distinguishes_selections_from_unmarked_print(self):
        prompt = pipeline.build_ocr_prompt()

        self.assertIn("hand-circled or hand-checked PRINTED option", prompt)
        self.assertIn("Do not infer a selection", prompt)
        self.assertIn('"classification":', prompt)
        self.assertIn('"selection": null', prompt)

    def test_selection_does_not_discard_handwriting_in_the_same_crop(self):
        candidate = {
            **self.final, "classification": "handwritten_text",
            "final_transcription": "Written name",
            "handwritten_transcription": "Written name",
        }
        evidence = [{
            "source": "core4_multi",
            "ocr": {
                "classification": "handwritten_text",
                "full_transcription": "Written name",
                "selection": self.final["selection"],
            },
        }]

        result = pipeline.finalize_region_semantics(candidate, evidence)

        self.assertEqual("Written name", result["final_transcription"])
        self.assertEqual("Option A", result["selection"]["text"])
        self.assertEqual("handwritten_text", result["classification"])

    def test_mixed_legacy_transcription_is_preserved_instead_of_replaced_by_selection(self):
        result = pipeline.finalize_mark_selection({
            **self.final, "final_transcription": "Written name and Option A",
        }, self.usable)

        self.assertEqual("Written name and Option A", result["final_transcription"])
        self.assertTrue(result["needs_review"])

    def test_multi_digit_amount_with_x_is_not_replaced_by_a_mark_description(self):
        result = pipeline.finalize_region_semantics({
            "classification": "mark_or_checkbox",
            "final_transcription": "12345678",
            "handwritten_transcription": "12345678",
            "marks": ["x"], "digits": list("12345678"), "notes": [],
        }, [])

        self.assertEqual("12345678", result["final_transcription"])
        self.assertEqual("handwritten_text", result["classification"])
        self.assertEqual(["x"], result["marks"])

    def test_unreadable_handwriting_is_not_hidden_by_a_readable_selected_label(self):
        result = pipeline.finalize_mark_selection({
            **self.final, "handwritten_transcription": "[?]",
        }, self.usable)

        self.assertEqual("[?]", result["final_transcription"])
        self.assertEqual("Option A", result["selection"]["text"])
        self.assertTrue(result["needs_review"])

    def test_selected_value_and_mark_metadata_survive_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "marked.png"
            output = root / "output"
            Image.new("RGB", (100, 60), "white").save(source)
            reading = {
                "full_transcription": "Option A",
                "classification": "mark_or_checkbox",
                "marks": ["circle"],
                "selection": self.final["selection"],
            }
            with patch.object(pipeline, "OCR_STRATEGY", "standard"), \
                 patch.object(pipeline, "VOTE_ENABLED", False), \
                 patch.object(pipeline, "ocr_image_set", return_value=reading), \
                 patch.object(pipeline, "visual_reconcile_ocr_results", return_value=self.final), \
                 patch.object(pipeline, "should_contextual_recheck", return_value=False), \
                 patch.object(pipeline, "build_merged_jpg", return_value=None):
                code = pipeline.run_pipeline(
                    str(source), page=1, dpi=400, crops=[(0, 0, 100, 60)],
                    output_dir=str(output), chunk_width=900, overlap=150, ocr_mode="full",
                )
            result = json.loads((output / "handwriting_result.json").read_text())
            region = result["regions"][0]
            details = json.loads((output / region["folder"] / "result.json").read_text())

        self.assertEqual(0, code)
        for exported in (region, details):
            self.assertEqual("Option A", exported["selection"]["text"])
            self.assertEqual(["circle"], exported["marks"])
            self.assertEqual([], exported["digits"])


if __name__ == "__main__":
    unittest.main()
