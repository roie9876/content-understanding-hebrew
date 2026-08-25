import json
import tempfile
import unittest
from pathlib import Path

import direct_region_ocr


class DirectRegionOcrTests(unittest.TestCase):
    def test_discovers_numbered_crops_without_reading_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for region_id in (2, 1, 3):
                crop_dir = root / f"region_{region_id:02d}_label"
                crop_dir.mkdir()
                (crop_dir / "01_original_crop.png").write_bytes(b"png")
            (root / "handwriting_result.json").write_text("not valid JSON")

            discovered = direct_region_ocr.discover_region_images(root)

        self.assertEqual([1, 2, 3], [region_id for region_id, _path in discovered])

    def test_payload_contains_only_prompt_and_image_evidence(self):
        payload = direct_region_ocr.build_payload(b"image", "gpt-5.6-sol", "high", 32000)
        serialized = json.dumps(payload)

        self.assertEqual({"effort": "high"}, payload["reasoning"])
        self.assertNotIn("golden", serialized.lower())
        self.assertNotIn("final_transcription", serialized)
        self.assertNotIn("region_", serialized)
        self.assertIn("data:image/png;base64,", serialized)

    def test_stroke_grounded_payload_forbids_unsupported_entity_completion(self):
        payload = direct_region_ocr.build_payload(
            b"image", "gpt-5.6-sol", "high", 32000,
            prompt=direct_region_ocr.STROKE_GROUNDED_PROMPT,
        )
        prompt = payload["input"][0]["content"][0]["text"]

        self.assertIn("minimal spelling repair", prompt)
        self.assertIn("changes at most one character per token", prompt)
        self.assertIn("Never introduce a new proper noun", prompt)
        self.assertIn("Prefer incomplete text over invention", prompt)

    def test_parses_structured_sections_and_multiline_text(self):
        parsed = direct_region_ocr.parse_sections("""CLASSIFICATION
handwritten_text
TRANSCRIPTION
בנק הפועלים
סניף פארק הירקון
CONFIDENCE
medium
ALTERNATIVES
- עומר
- אומר
""")

        self.assertEqual("handwritten_text", parsed["classification"])
        self.assertEqual("בנק הפועלים\nסניף פארק הירקון", parsed["transcription"])
        self.assertEqual("medium", parsed["confidence"])
        self.assertEqual(["עומר", "אומר"], parsed["alternatives"])

    def test_parses_stroke_grounded_evidence_and_final_text(self):
        parsed = direct_region_ocr.parse_stroke_grounded_sections("""CLASSIFICATION
handwritten_text
VISUAL_TRANSCRIPTION
שלו[?]
GLYPH_EVIDENCE
שלו[?]: final glyph uncertain
CORRECTIONS
שלו[?] -> שלום: final glyph looks like mem
FINAL_TRANSCRIPTION
שלום
CONFIDENCE
medium
""")

        self.assertEqual("שלו[?]", parsed["visual_transcription"])
        self.assertIn("final glyph uncertain", parsed["glyph_evidence"])
        self.assertIn("-> שלום", parsed["corrections"])
        self.assertEqual("שלום", parsed["transcription"])
        self.assertEqual("medium", parsed["confidence"])


if __name__ == "__main__":
    unittest.main()