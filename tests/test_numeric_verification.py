import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import handwriting_pipeline as pipeline


class NumericVerificationTests(unittest.TestCase):
    @staticmethod
    def reading(text, confidence=0.95):
        return {
            "classification": "handwritten_text", "final_transcription": text,
            "confidence": confidence, "words": [], "notes": [],
        }

    def test_numeric_correction_needs_two_clear_independent_rereads(self):
        current = self.reading("12345")
        recheck = self.reading("12346")

        self.assertTrue(pipeline.numeric_correction_confirmed(
            current, recheck, self.reading("12346"),
        ))
        for confirmation in (
            self.reading("12347"), self.reading("12346[?]"),
            self.reading("12346", 0.79), self.reading("12346", True),
            {**self.reading("12346"), "classification": "no_handwriting"},
        ):
            with self.subTest(confirmation=confirmation):
                self.assertFalse(pipeline.numeric_correction_confirmed(
                    current, recheck, confirmation,
                ))

    def test_unconfirmed_numeric_changes_do_not_replace_existing_values(self):
        current = self.reading("12345")
        recheck = self.reading("12346")

        self.assertIs(current, pipeline.accept_contextual_recheck(current, recheck))
        accepted = pipeline.accept_contextual_recheck(current, recheck, confirmed_numeric=True)
        self.assertEqual("12346", accepted["final_transcription"])

    def test_confirmed_date_correction_can_fix_a_wrong_day_without_completing_year(self):
        current = self.reading("10/7/26")
        recheck = self.reading("12/7/26")
        self.assertIs(current, pipeline.accept_contextual_recheck(current, recheck))
        self.assertTrue(pipeline.numeric_correction_confirmed(
            current, recheck, self.reading("12/7/26"),
        ))
        accepted = pipeline.accept_contextual_recheck(current, recheck, confirmed_numeric=True)
        self.assertEqual("12/7/26", accepted["final_transcription"])

    def test_numeric_signs_are_not_silently_dropped(self):
        current = self.reading("-12345")
        self.assertIs(current, pipeline.accept_contextual_recheck(
            current, self.reading("12345"),
        ))

    def test_cross_region_conflict_is_not_overruled_by_another_regions_cu_value(self):
        current = {
            **self.reading("123457[?]"), "field_type": "numeric",
            "cross_region_conflict": {"alternative_transcription": "123456"},
        }
        peer = {**self.reading("123456"), "field_type": "numeric"}

        results = pipeline.reconcile_uncertain_repeated_numbers(
            [current, peer], [{"label": "account"}, {"label": "account"}], "123456",
        )

        self.assertEqual("123457[?]", results[0]["final_transcription"])
        without_conflict = {key: value for key, value in current.items() if key != "cross_region_conflict"}
        existing_repair = pipeline.reconcile_uncertain_repeated_numbers(
            [without_conflict, peer], [{"label": "account"}, {"label": "account"}], "123456",
        )
        self.assertEqual("123456", existing_repair[0]["final_transcription"])

    @patch.object(pipeline, "_responses_call")
    def test_context_reread_gets_original_crop_and_no_numeric_candidate(self, call):
        call.return_value = self.reading("12345")
        page = Image.new("RGB", (400, 300), "white")
        box = [100, 100, 80, 40]

        pipeline.contextual_region_recheck(page, {"box": box}, self.reading("987654321"))

        content = call.call_args.args[0]
        self.assertNotIn("987654321", content[0]["text"])
        self.assertIn("Never copy a word or number from another field", content[0]["text"])
        self.assertEqual(
            pipeline._to_data_uri(page.crop((100, 100, 180, 140))),
            content[2]["image_url"],
        )
        self.assertEqual(3, sum(item["type"] == "input_image" for item in content))

    @patch.object(pipeline, "_responses_call")
    def test_confirmation_is_candidate_blind_and_uses_only_faithful_views(self, call):
        call.return_value = self.reading("12345")
        images = [
            (name, Image.new("RGB", (40, 20), "white"))
            for name in (*pipeline.CORE_OCR_VIEWS, "03_adaptive_threshold.png")
        ]

        pipeline.numeric_region_confirmation(images)

        content = call.call_args.args[0]
        self.assertEqual(3, sum(item["type"] == "input_image" for item in content))
        self.assertIn("no OCR candidates", content[0]["text"])
        self.assertEqual("medium", call.call_args.kwargs["reasoning"])
        self.assertEqual(1, call.call_args.kwargs["endpoint_limit"])

    def test_pipeline_surfaces_disagreement_instead_of_overwriting_a_numeric_field(self):
        for confirmed in (False, True):
            with self.subTest(confirmed=confirmed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "numeric.png"
                output = root / "output"
                Image.new("RGB", (120, 60), "white").save(source)
                with patch.object(pipeline, "OCR_STRATEGY", "standard"), \
                     patch.object(pipeline, "VOTE_ENABLED", False), \
                     patch.object(pipeline, "ocr_image_set", return_value={
                         "classification": "handwritten_text",
                         "full_transcription": "12345", "words": [],
                     }), \
                     patch.object(
                         pipeline, "visual_reconcile_ocr_results", return_value=self.reading("12345"),
                     ), \
                     patch.object(pipeline, "contextual_region_recheck", return_value=(
                         self.reading("12346"), (Image.new("RGB", (120, 60)),) * 2,
                     )), \
                     patch.object(pipeline, "numeric_region_confirmation", return_value=self.reading(
                         "12346" if confirmed else "12347",
                     )), \
                     patch.object(pipeline, "build_merged_jpg", return_value=None):
                    code = pipeline.run_pipeline(
                        str(source), page=1, dpi=400, crops=[(0, 0, 120, 60)],
                        output_dir=str(output), chunk_width=900, overlap=150, ocr_mode="full",
                    )
                result = json.loads((output / "handwriting_result.json").read_text())["regions"][0]
                self.assertEqual(0, code)
                self.assertEqual("12346" if confirmed else "12345[?]", result["final_transcription"])
                self.assertEqual(confirmed, result["numeric_confirmation"]["confirmed"])
                if not confirmed:
                    self.assertTrue(result["needs_review"])
                    self.assertIn("12346", result["alternative_transcriptions"])


if __name__ == "__main__":
    unittest.main()
