import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

import segmented_line_benchmark as benchmark
from segmented_ocr import build_line_view_set


class SegmentedLineBenchmarkTests(unittest.TestCase):
    def test_detects_and_isolates_three_blue_handwritten_lines(self):
        image = Image.new("RGB", (240, 150), "white")
        draw = ImageDraw.Draw(image)
        for baseline in (40, 85, 130):
            draw.line((0, baseline, 239, baseline), fill="black", width=2)
        draw.line((165, 12, 220, 37), fill=(35, 75, 210), width=6)
        draw.line((85, 55, 150, 82), fill=(35, 75, 210), width=6)
        draw.line((15, 102, 70, 127), fill=(35, 75, 210), width=6)

        lines, metadata, _mask = benchmark.segment_handwritten_lines(image, expected_lines=3)

        self.assertEqual(3, len(lines))
        for expected, actual in zip((40, 85, 130), (item["baseline_y"] for item in metadata)):
            self.assertLessEqual(abs(expected - actual), 1)
        self.assertGreater(metadata[0]["box"][0], metadata[1]["box"][0])
        self.assertGreater(metadata[1]["box"][0], metadata[2]["box"][0])
        self.assertTrue(all(int(benchmark.blue_pen_mask(line).sum()) > 100 for line in lines))

    def test_character_alignment_marks_only_disagreement(self):
        consensus, similarity = benchmark.align_readings("אבגד", "אבחד")

        self.assertEqual("אב[?]ד", consensus)
        self.assertEqual(0.75, similarity)

    def test_configured_model_requires_one_deployment(self):
        self.assertEqual(
            "gpt-5.6-sol",
            benchmark.configured_model_name([
                {"deployment": "gpt-5.6-sol"},
                {"deployment": "gpt-5.6-sol"},
            ]),
        )
        with self.assertRaisesRegex(RuntimeError, "exactly one configured model"):
            benchmark.configured_model_name([
                {"deployment": "gpt-5.4"},
                {"deployment": "gpt-5.6-sol"},
            ])

    @patch.object(benchmark.pipeline, "_responses_call")
    def test_model_reads_use_configured_output_cap(self, responses_call):
        responses_call.return_value = {"transcription": "אב"}

        benchmark.run_model_reads(
            [Image.new("RGB", (20, 10), "white")],
            "low",
            workers=1,
            max_output_tokens=2048,
        )

        self.assertEqual(2, responses_call.call_count)
        self.assertTrue(all(call.args[3] == 2048 for call in responses_call.call_args_list))

    def test_segments_word_groups_in_right_to_left_order(self):
        line = Image.new("RGB", (240, 70), "white")
        draw = ImageDraw.Draw(line)
        draw.line((185, 15, 225, 55), fill=(35, 75, 210), width=6)
        draw.line((100, 15, 145, 55), fill=(35, 75, 210), width=6)
        draw.line((15, 15, 65, 55), fill=(35, 75, 210), width=6)

        words, metadata = benchmark.segment_word_groups(line)

        self.assertEqual(3, len(words))
        left_edges = [item["box"][0] for item in metadata]
        self.assertEqual(sorted(left_edges, reverse=True), left_edges)
        self.assertTrue(all(int(benchmark.blue_pen_mask(word).sum()) > 100 for word in words))

    def test_narrow_join_width_preserves_close_visual_units(self):
        line = Image.new("RGB", (220, 70), "white")
        draw = ImageDraw.Draw(line)
        for left, right in ((170, 205), (105, 145), (82, 92), (15, 55)):
            draw.line((left, 15, right, 55), fill=(35, 75, 210), width=6)

        words, _metadata = benchmark.segment_word_groups(line, join_width=7)

        self.assertEqual(4, len(words))

    def test_multi_view_content_contains_only_one_units_views(self):
        unit = [
            ("original_with_form_line", Image.new("RGB", (20, 10), "white")),
            ("blue_only", Image.new("RGB", (20, 10), "white")),
            ("blue_only_enlarged", Image.new("RGB", (60, 30), "white")),
        ]

        content = benchmark._unit_content(benchmark.PROMPT_A, unit)

        self.assertEqual(3, sum(item["type"] == "input_image" for item in content))
        prompt = content[0]["text"]
        self.assertIn("SAME isolated handwriting unit", prompt)
        self.assertIn("no neighboring word", prompt)

    def test_line_view_set_has_three_faithful_views(self):
        source = Image.new("RGB", (120, 70), "white")
        draw = ImageDraw.Draw(source)
        draw.line((0, 45, 119, 45), fill="black", width=2)
        draw.line((20, 15, 100, 42), fill=(35, 75, 210), width=6)
        line = source.crop((10, 5, 111, 56))

        views = build_line_view_set(
            source,
            line,
            {"box": [10, 5, 101, 51]},
        )

        self.assertEqual(
            ["original_with_form_line", "blue_only", "blue_only_enlarged"],
            [name for name, _image in views],
        )
        self.assertEqual((303, 153), views[-1][1].size)

    def test_prompts_are_candidate_blind_and_forbid_entity_completion(self):
        for prompt in (benchmark.PROMPT_A, benchmark.PROMPT_B):
            normalized = " ".join(prompt.lower().split())
            self.assertIn("[?]", prompt)
            self.assertNotIn("פתח תקווה", prompt)
            self.assertNotIn("פארק הירקון", prompt)
            self.assertRegex(normalized, r"no (document context|previous reading)")


if __name__ == "__main__":
    unittest.main()