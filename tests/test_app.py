import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image, ImageDraw

import app as web_app
import compare_golden
import handwriting_pipeline as pipeline
import quality_replay


class UploadFormatTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.upload_dir = Path(self.temp_dir.name) / "uploads"
        self.upload_ids = []

    def tearDown(self):
        for upload_id in self.upload_ids:
            web_app._uploads.pop(upload_id, None)
        self.temp_dir.cleanup()

    def _upload(self, data: bytes, file_name: str):
        with patch.object(web_app, "UPLOAD_DIR", self.upload_dir):
            return web_app.app.test_client().post(
                "/api/uploads",
                data={"document": (io.BytesIO(data), file_name)},
                content_type="multipart/form-data",
            )

    def test_webp_upload_is_normalized_and_previewable(self):
        source = io.BytesIO()
        Image.new("RGB", (24, 16), "white").save(source, format="WEBP")

        response = self._upload(source.getvalue(), "scan.webp")
        payload = response.get_json()

        self.assertEqual(201, response.status_code)
        self.assertEqual(1, payload["page_count"])
        self.upload_ids.append(payload["id"])
        upload = web_app._uploads[payload["id"]]
        self.assertEqual(".png", Path(upload["input_path"]).suffix)
        preview = web_app.app.test_client().get(payload["pages"][0]["preview_url"])
        self.assertEqual(200, preview.status_code)
        preview.close()

    def test_heic_upload_is_normalized_and_previewable(self):
        source = io.BytesIO()
        Image.new("RGB", (24, 16), "white").save(source, format="HEIF")

        response = self._upload(source.getvalue(), "scan.heic")
        payload = response.get_json()

        self.assertEqual(201, response.status_code)
        self.assertEqual(1, payload["page_count"])
        self.upload_ids.append(payload["id"])
        upload = web_app._uploads[payload["id"]]
        self.assertEqual(".png", Path(upload["input_path"]).suffix)
        preview = web_app.app.test_client().get(payload["pages"][0]["preview_url"])
        self.assertEqual(200, preview.status_code)
        preview.close()

    def test_multiframe_tiff_upload_preserves_pages(self):
        source = io.BytesIO()
        first = Image.new("RGB", (24, 16), "white")
        second = Image.new("RGB", (24, 16), "black")
        first.save(source, format="TIFF", save_all=True, append_images=[second])

        response = self._upload(source.getvalue(), "scan.tiff")
        payload = response.get_json()

        self.assertEqual(201, response.status_code)
        self.assertEqual(2, payload["page_count"])
        self.upload_ids.append(payload["id"])
        upload = web_app._uploads[payload["id"]]
        self.assertEqual(".pdf", Path(upload["input_path"]).suffix)
        for page in payload["pages"]:
            preview = web_app.app.test_client().get(page["preview_url"])
            self.assertEqual(200, preview.status_code)
            preview.close()


class ConcurrentJobTests(unittest.TestCase):
    def test_ui_only_offers_sol61_with_standard_core4(self):
        self.assertEqual("gpt-6.1-sol", web_app.DEFAULT_MODEL_PROFILE)
        self.assertEqual("standard", web_app.DEFAULT_OCR_STRATEGY)

        response = web_app.app.test_client().get("/")
        html = response.get_data(as_text=True)
        response.close()

        self.assertEqual(200, response.status_code)
        models = html.split('<select id="modelSelect" name="model_profile">', 1)[1].split(
            "</select>", 1,
        )[0]
        self.assertEqual(1, models.count("<option"))
        self.assertIn('<option value="gpt-6.1-sol" selected>', models)
        self.assertIn('<option value="standard" selected>', html)
        self.assertNotIn('<option value="isolated-multiview">', html)
        self.assertNotIn('data-view="ivrit"', html)
        self.assertNotIn('data-view="kraken"', html)
        self.assertNotIn('standard-kraken', html)

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)
        self.job_id = "test-job"
        web_app._jobs[self.job_id] = {
            "file_name": "test.pdf",
            "model_profile": "gpt-5.4",
            "input_path": str(self.output_dir / "test.pdf"),
            "output_dir": str(self.output_dir),
            "page_count": 4,
            "page_states": {
                page: {"stage": "waiting", "message": "Waiting"}
                for page in range(1, 5)
            },
            "current_page": 1,
            "status": "queued",
            "stage": "queued",
            "message": "Waiting to start",
            "elapsed_seconds": None,
            "error": None,
            "log": [],
        }

    def tearDown(self):
        web_app._jobs.pop(self.job_id, None)
        web_app._processes.pop(self.job_id, None)
        self.temp_dir.cleanup()

    def test_two_pages_run_concurrently(self):
        def fake_page(job_id, page_number, workers):
            with web_app._lock:
                state = web_app._jobs[job_id]["page_states"][page_number]
                state.update(stage="preparing", started_at=time.time())
            time.sleep(0.05)
            with web_app._lock:
                state.update(stage="complete", elapsed_seconds=0.05)
            return 0

        started = time.monotonic()
        with patch.object(web_app, "PAGE_CONCURRENCY", 2), \
             patch.object(web_app, "_run_page", side_effect=fake_page):
            web_app._run_job(self.job_id)
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.18)
        self.assertEqual("complete", web_app._jobs[self.job_id]["status"])
        self.assertTrue(all(
            state["stage"] == "complete"
            for state in web_app._jobs[self.job_id]["page_states"].values()
        ))

    def test_page_range_parser_supports_lists_and_ranges(self):
        self.assertEqual([1, 3, 4, 5, 8], web_app._parse_page_spec("1,3-5,8", 8))

        with self.assertRaisesRegex(ValueError, "outside this 8-page document"):
            web_app._parse_page_spec("1,9", 8)
        with self.assertRaisesRegex(ValueError, "ascending"):
            web_app._parse_page_spec("5-3", 8)
        with self.assertRaisesRegex(ValueError, "Select at least one"):
            web_app._parse_page_spec("", 8)

    def test_golden_match_uses_global_assignment_for_overlapping_boxes(self):
        golden = [
            {"box": [0, 0, 200, 100]},
            {"box": [80, 0, 100, 100]},
        ]
        current = [
            {"box": [0, 0, 100, 100], "value": "first"},
            {"box": [100, 0, 100, 100], "value": "second"},
        ]

        pairs, extras = compare_golden._match(golden, current)

        self.assertEqual(["first", "second"], [pair[1]["value"] for pair in pairs])
        self.assertEqual([], extras)

    def test_golden_match_uses_spatial_assignment_above_fifteen_regions(self):
        golden = [
            {"box": [index * 100, 0, 10, 10], "final_transcription": str(index)}
            for index in range(20)
        ]
        current = [
            {"box": [index * 100, 0, 10, 10], "final_transcription": str(index)}
            for index in reversed(range(20))
        ]

        pairs, extras = compare_golden._match(golden, current)

        self.assertEqual([], extras)
        self.assertEqual(
            [region["final_transcription"] for region in golden],
            [region["final_transcription"] for _golden, region in pairs],
        )

    def test_exact_golden_normalization_ignores_layout_not_characters(self):
        self.assertEqual(
            compare_golden._exact_norm("012123\n142 05686399"),
            compare_golden._exact_norm("012123 142 05686399"),
        )
        self.assertNotEqual(
            compare_golden._exact_norm("5252731"),
            compare_golden._exact_norm("731 5252"),
        )

    def test_quality_replay_similarity_ignores_punctuation_and_layout(self):
        self.assertEqual(
            1.0,
            quality_replay._quality_ratio(
                "012123\n142 05686399", "012-123 / 142.05686399"
            ),
        )
        self.assertLess(
            quality_replay._quality_ratio("5252731", "731 5252"),
            1.0,
        )

    def test_quality_replay_aggregate_weights_text_regions(self):
        totals = quality_replay.aggregate([
            {
                "text_total": 2,
                "before_exact": 1,
                "after_exact": 2,
                "before_similarity_sum": 1.0,
                "after_similarity_sum": 2.0,
                "classification_total": 1,
                "before_classification_correct": 0,
                "after_classification_correct": 1,
                "golden_regions": 3,
                "saved_regions": 3,
                "processed_regions": 3,
                "before_extras": 0,
                "after_extras": 0,
            },
            {
                "text_total": 1,
                "before_exact": 0,
                "after_exact": 0,
                "before_similarity_sum": 0.5,
                "after_similarity_sum": 0.5,
                "classification_total": 0,
                "before_classification_correct": 0,
                "after_classification_correct": 0,
                "golden_regions": 1,
                "saved_regions": 1,
                "processed_regions": 1,
                "before_extras": 0,
                "after_extras": 0,
            },
        ])

        self.assertEqual(3, totals["text_total"])
        self.assertEqual(2, totals["after_exact"])
        self.assertEqual(2.5 / 3, totals["after_similarity"])
        self.assertEqual(1, totals["after_classification_correct"])

    def test_strict_golden_scores_text_and_signature_classification(self):
        golden = {
            "regions": [
                {"box": [0, 0, 10, 10], "final_transcription": "12/7/26"},
                {"box": [20, 0, 10, 10], "final_transcription": "",
                 "human_classification": "signature_non_text"},
            ]
        }
        current = {
            "regions": [
                {"box": [0, 0, 10, 10], "final_transcription": "12.7.26"},
                {"box": [20, 0, 10, 10], "final_transcription": "",
                 "classification": "signature_non_text"},
            ]
        }
        with tempfile.TemporaryDirectory() as temp_dir, \
             patch.object(compare_golden, "GOLDEN_DIR", Path(temp_dir) / "golden"):
            stem = "sample"
            golden_dir = compare_golden.GOLDEN_DIR / stem
            golden_dir.mkdir(parents=True)
            (golden_dir / "handwriting_result.json").write_text(
                json.dumps(golden), encoding="utf-8"
            )
            current_path = Path(temp_dir) / "current.json"
            current_path.write_text(json.dumps(current), encoding="utf-8")

            result = compare_golden.compare_sample(stem, current_path)

        self.assertEqual(1, result["strict"]["text_exact"])
        self.assertEqual(1, result["strict"]["classification_correct"])
        self.assertTrue(result["strict"]["pass"])

    def test_golden_comparison_fails_extra_region_count(self):
        golden = {"regions": [{
            "box": [0, 0, 10, 10],
            "final_transcription": "12/7/26",
        }]}
        current = {"regions": [
            {
                "box": [0, 0, 10, 10],
                "final_transcription": "12/7/26",
            },
            {
                "box": [20, 0, 10, 10],
                "final_transcription": "extra",
            },
        ]}
        with tempfile.TemporaryDirectory() as temp_dir, \
             patch.object(compare_golden, "GOLDEN_DIR", Path(temp_dir) / "golden"):
            stem = "sample"
            golden_dir = compare_golden.GOLDEN_DIR / stem
            golden_dir.mkdir(parents=True)
            (golden_dir / "handwriting_result.json").write_text(
                json.dumps(golden), encoding="utf-8"
            )
            current_path = Path(temp_dir) / "current.json"
            current_path.write_text(json.dumps(current), encoding="utf-8")

            result = compare_golden.compare_sample(stem, current_path)

        self.assertTrue(result["count_fail"])
        self.assertEqual("FAIL", result["verdict"])

    def test_merged_png_download_renders_merged_document(self):
        page_dir = web_app._page_dir(web_app._jobs[self.job_id], 1)
        page_dir.mkdir(parents=True)
        Image.new("RGB", (12, 8), "white").save(page_dir / "merged.jpg")
        (page_dir / "merged.md").write_text(
            '# מסמך ממוזג\n\n<span style="color:#d00">כתב יד</span>',
            encoding="utf-8",
        )

        response = web_app.app.test_client().get(
            f"/api/jobs/{self.job_id}/pages/1/merged.png"
        )

        self.assertEqual(200, response.status_code)
        self.assertEqual("image/png", response.mimetype)
        self.assertIn("attachment", response.headers["Content-Disposition"])
        self.assertEqual(b"\x89PNG\r\n\x1a\n", response.data[:8])
        with Image.open(io.BytesIO(response.data)) as image:
            self.assertEqual((1800, 2546), image.size)

    def test_golden_comparison_route_returns_similarity_and_rows(self):
        result = {
            "regions": [{
                "box": [0, 0, 10, 10],
                "label": "date",
                "final_transcription": "12/7/26",
                "classification": "handwritten_text",
            }]
        }
        golden = {
            "page": 1,
            "regions": [{
                "box": [0, 0, 10, 10],
                "label": "date",
                "final_transcription": "12/7/26",
            }]
        }
        page_dir = web_app._page_dir(web_app._jobs[self.job_id], 1)
        page_dir.mkdir(parents=True)
        (page_dir / "handwriting_result.json").write_text(
            json.dumps(result), encoding="utf-8"
        )
        golden_root = self.output_dir / "golden"
        golden_dir = golden_root / "test"
        golden_dir.mkdir(parents=True)
        (golden_dir / "handwriting_result.json").write_text(
            json.dumps(golden), encoding="utf-8"
        )

        with patch.object(compare_golden, "GOLDEN_DIR", golden_root):
            response = web_app.app.test_client().get(
                f"/api/jobs/{self.job_id}/pages/1/golden-comparison"
            )

        payload = response.get_json()
        self.assertEqual(200, response.status_code)
        self.assertTrue(payload["available"])
        self.assertEqual(100.0, payload["success_percent"])
        self.assertEqual("12/7/26", payload["rows"][0]["golden"])

    def test_only_selected_pages_are_scheduled_and_exposed(self):
        selected = [2, 4]
        web_app._jobs[self.job_id]["selected_pages"] = selected
        web_app._jobs[self.job_id]["page_states"] = {
            page: {"stage": "waiting", "message": "Waiting"} for page in selected
        }
        calls = []

        def fake_page(job_id, page_number, workers):
            calls.append(page_number)
            with web_app._lock:
                state = web_app._jobs[job_id]["page_states"][page_number]
                state.update(stage="complete", elapsed_seconds=0.01)
            return 0

        with patch.object(web_app, "_run_page", side_effect=fake_page):
            web_app._run_job(self.job_id)

        snapshot = web_app._job_snapshot(self.job_id)
        self.assertEqual(selected, sorted(calls))
        self.assertEqual(selected, snapshot["selected_pages"])
        self.assertEqual(selected, [page["page"] for page in snapshot["pages"]])
        self.assertEqual(2, snapshot["selected_page_count"])

    def test_page_snapshot_uses_live_progress_usage_before_final_result(self):
        page_dir = self.output_dir / "page_0001"
        page_dir.mkdir()
        (page_dir / "ocr_progress.json").write_text(json.dumps({
            "completed": 2,
            "failed": 0,
            "total": 10,
            "usage": {
                "input_tokens": 120,
                "output_tokens": 80,
                "total_tokens": 200,
                "calls": 2,
            },
        }))

        snapshot = web_app._page_snapshot(self.job_id, web_app._jobs[self.job_id], 1)

        self.assertEqual(2, snapshot["calls"])
        self.assertEqual(200, snapshot["tokens"])
        self.assertEqual(120, snapshot["input_tokens"])
        self.assertEqual(80, snapshot["output_tokens"])

    def test_pipeline_log_is_tagged_and_persistent(self):
        web_app._jobs[self.job_id]["status"] = "running"
        web_app._jobs[self.job_id]["page_states"][2]["stage"] = "preparing"

        web_app._handle_output(self.job_id, 2, "[detect] locating handwriting regions")

        log_text = (self.output_dir / "pipeline.log").read_text(encoding="utf-8")
        self.assertIn("[page 2] [detect] locating", log_text)
        self.assertEqual(
            "detecting_handwriting",
            web_app._jobs[self.job_id]["page_states"][2]["stage"],
        )

    def test_tiled_detection_progress_is_visible(self):
        web_app._jobs[self.job_id]["status"] = "running"
        web_app._jobs[self.job_id]["page_states"][2]["stage"] = "detecting_handwriting"

        web_app._handle_output(
            self.job_id, 2,
            "[detect] scanning coverage segment 2/5 (24%-48%) ...",
        )

        page_state = web_app._jobs[self.job_id]["page_states"][2]
        self.assertEqual("detecting_handwriting", page_state["stage"])
        self.assertEqual("Scanning handwriting coverage 2/5", page_state["message"])

    def test_luna_profile_sets_page_subprocess_environment(self):
        environment = web_app._model_environment("gpt-5.6-luna")

        self.assertEqual(
            os.environ.get("GPT_LUNA_ENDPOINT", "https://example.invalid/openai/v1/"),
            environment["GPT_ENDPOINT"],
        )
        self.assertEqual("gpt-5.6-luna", environment["GPT_DEPLOYMENT"])
        self.assertEqual("", environment["GPT_TARGETS"])

    def test_gpt_54_profile_sets_two_endpoint_pool(self):
        environment = web_app._model_environment("gpt-5.4")
        targets = json.loads(environment["GPT_TARGETS"])

        self.assertEqual(
            [target["name"] for target in json.loads(os.environ.get("GPT_TARGETS", "[]"))],
            [target["name"] for target in targets],
        )
        self.assertEqual(os.environ.get("GPT_DEPLOYMENT", "gpt-5.4"), targets[0]["deployment"])

    def test_gpt_54_pro_profile_sets_deployment_reasoning_and_timeout(self):
        environment = web_app._model_environment("gpt-5.4-pro")

        self.assertEqual(
            os.environ.get("GPT_PRO_ENDPOINT", os.environ.get("GPT_ENDPOINT")),
            environment["GPT_ENDPOINT"],
        )
        self.assertEqual("gpt-5.4-pro", environment["GPT_DEPLOYMENT"])
        self.assertEqual("medium", environment["GPT_REASONING"])
        self.assertEqual("medium", environment["OCR_REASONING"])
        self.assertEqual("180", environment["GPT_READ_TIMEOUT"])

    def test_sol_profile_sets_page_subprocess_environment(self):
        environment = web_app._model_environment("gpt-5.6-sol")

        self.assertEqual(
            os.environ.get("GPT_SOL_ENDPOINT", "https://example.invalid/openai/v1/"),
            environment["GPT_ENDPOINT"],
        )
        self.assertEqual("gpt-5.6-sol", environment["GPT_DEPLOYMENT"])
        self.assertEqual("low", environment["GPT_REASONING"])
        self.assertEqual("low", environment["OCR_REASONING"])
        self.assertEqual("", environment["GPT_TARGETS"])

    def test_sol61_profile_is_selectable_and_sets_page_environment(self):
        response = web_app.app.test_client().get("/")
        html = response.get_data(as_text=True)
        response.close()
        self.assertIn('<option value="gpt-6.1-sol" selected>', html)

        with patch.dict(os.environ, {"GPT_TARGETS": '[{"deployment":"gpt-5.4"}]'}):
            environment = web_app._page_environment("gpt-6.1-sol", "standard")

        self.assertEqual(
            os.environ.get("GPT_61_SOL_ENDPOINT", os.environ.get(
                "GPT_SOL_ENDPOINT", web_app._PRIMARY_ENDPOINT,
            )),
            environment["GPT_ENDPOINT"],
        )
        self.assertEqual(
            os.environ.get("GPT_61_SOL_DEPLOYMENT", "gpt-6.1-sol"),
            environment["GPT_DEPLOYMENT"],
        )
        self.assertEqual("low", environment["GPT_REASONING"])
        self.assertEqual("low", environment["OCR_REASONING"])
        self.assertEqual("", environment["GPT_TARGETS"])
        self.assertEqual("standard", environment["OCR_STRATEGY"])

    def test_isolated_multiview_reaches_page_subprocess_environment(self):
        environment = web_app._page_environment(
            "gpt-5.6-sol", "isolated-multiview",
        )

        self.assertEqual("gpt-5.6-sol", environment["GPT_DEPLOYMENT"])
        self.assertEqual("low", environment["OCR_REASONING"])
        self.assertEqual("isolated-multiview", environment["OCR_STRATEGY"])
        self.assertEqual("240", environment["GPT_READ_TIMEOUT"])
        self.assertEqual("2", environment["ISOLATED_MULTIVIEW_WORKERS"])
        self.assertEqual("1800", environment["ISOLATED_MULTIVIEW_MAX_TOKENS"])

    def test_gpt54_isolated_multiview_uses_medium_profile_timeout(self):
        environment = web_app._page_environment(
            "gpt-5.4", "isolated-multiview",
        )

        self.assertEqual("gpt-5.4", environment["GPT_DEPLOYMENT"])
        self.assertEqual("isolated-multiview", environment["OCR_STRATEGY"])
        self.assertEqual("120", environment["GPT_READ_TIMEOUT"])
        self.assertEqual("2", environment["ISOLATED_MULTIVIEW_WORKERS"])
        self.assertEqual("6000", environment["ISOLATED_MULTIVIEW_MAX_TOKENS"])

    def test_gpt54_standard_uses_fast_ocr_profile(self):
        environment = web_app._page_environment("gpt-5.4", "standard")

        self.assertEqual("low", environment["GPT_REASONING"])
        self.assertEqual("low", environment["OCR_REASONING"])
        self.assertEqual("4000", environment["OCR_MAX_TOKENS"])
        self.assertEqual("0", environment["OCR_FINAL_ORIGINAL_RESCUE"])
        self.assertEqual("medium", environment["DETECTION_REASONING"])
        self.assertEqual("6000", environment["DETECTION_MAX_TOKENS"])
        self.assertEqual("90", environment["GPT_READ_TIMEOUT"])

    def test_sol_none_profile_sets_fast_page_subprocess_environment(self):
        environment = web_app._model_environment("gpt-5.6-sol-none")

        self.assertEqual(
            os.environ.get("GPT_SOL_ENDPOINT", "https://example.invalid/openai/v1/"),
            environment["GPT_ENDPOINT"],
        )
        self.assertEqual("gpt-5.6-sol", environment["GPT_DEPLOYMENT"])
        self.assertEqual("none", environment["GPT_REASONING"])
        self.assertEqual("none", environment["OCR_REASONING"])
        self.assertEqual("", environment["GPT_TARGETS"])

    def test_copilot_opus_48_profile_uses_configured_gh_identity(self):
        token_result = Mock(stdout="account-specific-token\n")
        with patch.dict(os.environ, {"COPILOT_GITHUB_USER": "vision-user"}), \
             patch.object(web_app.subprocess, "run", return_value=token_result) as run:
            environment = web_app._model_environment("copilot-claude-opus-4.8")

        self.assertEqual("github-copilot-sdk", environment["MODEL_BACKEND"])
        self.assertEqual("claude-opus-4.8", environment["GPT_DEPLOYMENT"])
        self.assertEqual("1", environment["OCR_MULTI_IMAGE"])
        self.assertEqual("account-specific-token", environment["COPILOT_GITHUB_TOKEN"])
        self.assertEqual("1", environment["COPILOT_DISABLE_KEYTAR"])
        self.assertIn("vision-user", run.call_args.args[0])

    def test_copilot_opus_48_standard_page_uses_core4(self):
        token_result = Mock(stdout="account-specific-token\n")
        with patch.object(web_app.subprocess, "run", return_value=token_result):
            environment = web_app._page_environment(
                "copilot-claude-opus-4.8", "standard"
            )

        self.assertEqual("standard", environment["OCR_STRATEGY"])
        self.assertEqual("1", environment["OCR_MULTI_IMAGE"])
        self.assertEqual("github-copilot-sdk", environment["MODEL_BACKEND"])

    def test_copilot_gemini_35_flash_profile_uses_sdk_backend(self):
        token_result = Mock(stdout="account-specific-token\n")
        with patch.dict(os.environ, {"COPILOT_GITHUB_USER": "vision-user"}), \
             patch.object(web_app.subprocess, "run", return_value=token_result):
            environment = web_app._model_environment("copilot-gemini-3.5-flash")

        self.assertEqual("github-copilot-sdk", environment["MODEL_BACKEND"])
        self.assertEqual("gemini-3.5-flash", environment["GPT_DEPLOYMENT"])
        self.assertEqual("low", environment["GPT_REASONING"])
        self.assertEqual("low", environment["OCR_REASONING"])
        self.assertEqual("120", environment["GPT_READ_TIMEOUT"])
        self.assertEqual("account-specific-token", environment["COPILOT_GITHUB_TOKEN"])

    def test_sol_medium_profile_sets_higher_reasoning_and_timeout(self):
        environment = web_app._model_environment("gpt-5.6-sol-medium")

        self.assertEqual("gpt-5.6-sol", environment["GPT_DEPLOYMENT"])
        self.assertEqual("medium", environment["GPT_REASONING"])
        self.assertEqual("medium", environment["OCR_REASONING"])
        self.assertEqual("180", environment["GPT_READ_TIMEOUT"])

    def test_job_snapshot_exposes_selected_model(self):
        web_app._jobs[self.job_id]["model_profile"] = "gpt-5.6-luna"

        snapshot = web_app._job_snapshot(self.job_id)

        self.assertEqual("gpt-5.6-luna", snapshot["model_profile"])
        self.assertEqual("GPT-5.6 Luna (experimental)", snapshot["model_label"])

    def test_merged_document_renders_printed_black_and_handwriting_red(self):
        page_dir = self.output_dir / "page_0001"
        page_dir.mkdir()
        (page_dir / "merged.md").write_text(
            '# מסמך\n\nטקסט מודפס <span style="color:#d00">כתב יד</span>',
            encoding="utf-8",
        )

        response = web_app.app.test_client().get(
            f"/api/jobs/{self.job_id}/pages/1/merged-document"
        )

        self.assertEqual(200, response.status_code)
        html = response.get_data(as_text=True)
        self.assertIn("טקסט מודפס", html)
        self.assertIn("כתב יד", html)
        self.assertIn("color:#d00", html)
        self.assertIn('dir="rtl"', html)

    def test_create_job_rejects_unknown_model_profile(self):
        response = web_app.app.test_client().post(
            "/api/jobs",
            data={
                "document": (io.BytesIO(b"not-used"), "document.png"),
                "model_profile": "unknown-model",
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(400, response.status_code)
        self.assertEqual("Unsupported model profile.", response.get_json()["error"])

    def test_create_job_rejects_unknown_ocr_strategy(self):
        response = web_app.app.test_client().post(
            "/api/jobs",
            data={
                "document": (io.BytesIO(b"not-used"), "document.png"),
                "model_profile": "gpt-5.6-sol",
                "ocr_strategy": "unknown-strategy",
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(400, response.status_code)
        self.assertEqual("Unsupported OCR strategy.", response.get_json()["error"])

    def test_isolated_multiview_strategy_rejects_other_models(self):
        response = web_app.app.test_client().post(
            "/api/jobs",
            data={
                "document": (io.BytesIO(b"not-used"), "document.png"),
                "model_profile": "gpt-5.6-luna",
                "ocr_strategy": "isolated-multiview",
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(400, response.status_code)
        self.assertIn("requires GPT-5.4 or GPT-5.6 Sol low", response.get_json()["error"])

    def test_isolated_multiview_strategy_accepts_gpt54(self):
        upload_id = "gpt54-isolated-upload"
        web_app._uploads[upload_id] = {
            "id": upload_id,
            "file_name": "document.png",
            "input_path": str(self.output_dir / "document.png"),
            "page_count": 1,
            "created_at": time.time(),
        }
        try:
            with patch.object(web_app.threading.Thread, "start"), \
                 patch.object(web_app, "RESULTS_DIR", self.output_dir / "saved-runs"):
                response = web_app.app.test_client().post(
                    "/api/jobs",
                    data={
                        "upload_id": upload_id,
                        "model_profile": "gpt-5.4",
                        "ocr_strategy": "isolated-multiview",
                        "pages": "1",
                    },
                )
            payload = response.get_json()

            self.assertEqual(202, response.status_code)
            self.assertEqual("gpt-5.4", payload["model_profile"])
            self.assertEqual("isolated-multiview", payload["ocr_strategy"])
        finally:
            if "payload" in locals() and payload.get("id"):
                web_app._jobs.pop(payload["id"], None)
            web_app._uploads.pop(upload_id, None)

    def test_create_job_uses_selected_pages_from_prepared_upload(self):
        upload_id = "prepared-upload"
        web_app._uploads[upload_id] = {
            "id": upload_id,
            "file_name": "document.pdf",
            "input_path": str(self.output_dir / "document.pdf"),
            "page_count": 6,
            "created_at": time.time(),
        }
        try:
            with patch.object(web_app.threading.Thread, "start"):
                with patch.object(web_app, "RESULTS_DIR", self.output_dir / "saved-runs"):
                    response = web_app.app.test_client().post(
                        "/api/jobs",
                        data={
                            "upload_id": upload_id,
                            "ocr_strategy": "standard",
                            "pages": "2,4-5",
                        },
                    )
            payload = response.get_json()

            self.assertEqual(202, response.status_code)
            self.assertEqual("gpt-6.1-sol", payload["model_profile"])
            self.assertEqual([2, 4, 5], payload["selected_pages"])
            self.assertEqual("standard", payload["ocr_strategy"])
            self.assertEqual([2, 4, 5], [page["page"] for page in payload["pages"]])
            self.assertEqual(3, payload["selected_page_count"])
        finally:
            if "payload" in locals():
                web_app._jobs.pop(payload["id"], None)
            web_app._uploads.pop(upload_id, None)

    def test_run_name_keeps_original_name_and_timestamp(self):
        with patch.object(web_app, "RESULTS_DIR", self.output_dir):
            run_name = web_app._run_name("אישור מסירה 8.pdf", 1783830000)

        self.assertTrue(run_name.startswith("אישור מסירה 8.pdf__"))
        self.assertRegex(run_name, r"__\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")

    def test_saved_manifest_restores_completed_job(self):
        run_dir = self.output_dir / "document.pdf__2026-07-12_10-30-00"
        page_dir = run_dir / "page_0002"
        page_dir.mkdir(parents=True)
        (page_dir / "handwriting_result.json").write_text(
            json.dumps({"page": 2, "regions": [], "usage": {}}), encoding="utf-8"
        )
        manifest = {
            "id": "saved-job",
            "run_name": run_dir.name,
            "file_name": "document.pdf",
            "model_profile": "gpt-5.4",
            "page_count": 3,
            "selected_pages": [2],
            "created_at": 1783830000,
            "elapsed_seconds": 12.5,
        }
        (run_dir / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
        try:
            with patch.object(web_app, "RESULTS_DIR", self.output_dir):
                web_app._load_saved_jobs()
                snapshot = web_app._job_snapshot("saved-job")

            self.assertEqual("complete", snapshot["status"])
            self.assertEqual([2], snapshot["selected_pages"])
            self.assertEqual(run_dir.name, snapshot["run_name"])
        finally:
            web_app._jobs.pop("saved-job", None)

    def test_legacy_manifest_without_model_keeps_original_gpt54_profile(self):
        job = web_app._job_from_manifest(
            {"file_name": "legacy.pdf", "page_count": 1}, self.output_dir,
        )

        self.assertEqual("gpt-5.4", job["model_profile"])

    def test_atomic_json_diagnostic_is_immediately_readable(self):
        diagnostic = self.output_dir / "ocr_progress.json"
        value = {
            "completed": 2,
            "failed": 0,
            "total": 9,
            "regions": [{"region": 1, "candidates": [{"text": "שלום"}]}],
        }

        pipeline._write_json_atomic(diagnostic, value)

        self.assertEqual(value, web_app.json.loads(diagnostic.read_text(encoding="utf-8")))

    def test_atomic_text_artifact_is_immediately_readable(self):
        artifact = self.output_dir / "content_understanding.md"

        pipeline._write_text_atomic(artifact, "תוכן מודפס")

        self.assertEqual("תוכן מודפס", artifact.read_text(encoding="utf-8"))

    def test_result_artifact_reconciles_stale_running_job(self):
        web_app._jobs[self.job_id]["page_count"] = 1
        web_app._jobs[self.job_id]["page_states"] = {
            1: {"stage": "handwriting_ocr", "message": "Running", "started_at": time.time()}
        }
        web_app._jobs[self.job_id].update(status="running", started_at=time.time())
        pipeline._write_json_atomic(
            self.output_dir / "handwriting_result.json",
            {"page": 1, "regions": [], "usage": {}, "elapsed_seconds": 1.0},
        )

        snapshot = web_app._job_snapshot(self.job_id)

        self.assertEqual("complete", snapshot["status"])
        self.assertEqual(1, snapshot["processed_pages"])

    def test_diagnostics_endpoint_merges_live_and_final_evidence(self):
        web_app._jobs[self.job_id]["page_count"] = 1
        web_app._jobs[self.job_id]["page_states"] = {1: {"stage": "complete"}}
        pipeline._write_json_atomic(
            self.output_dir / "detected_regions.json",
            {"count": 1, "regions": [{"region": 1, "label": "note", "box": [1, 2, 3, 4]}]},
        )
        pipeline._write_json_atomic(
            self.output_dir / "ocr_progress.json",
            {"completed": 3, "total": 3, "regions": [{"region": 1, "candidates": [{"text": "שלום"}]}]},
        )
        region_dir = self.output_dir / "region_01_note"
        region_dir.mkdir()
        pipeline._write_json_atomic(
            region_dir / "result.json",
            {"label": "note", "final_transcription": "שלום", "vote": {"candidate_count": 3}},
        )
        pipeline._write_json_atomic(
            region_dir / "contextual_recheck.json",
            {"final_transcription": "שלם[?]", "confidence": 0.4},
        )

        response = web_app.app.test_client().get(f"/api/jobs/{self.job_id}/pages/1/diagnostics")
        payload = response.get_json()

        self.assertEqual(200, response.status_code)
        self.assertEqual(1, payload["detected"]["count"])
        self.assertEqual("שלום", payload["progress"]["regions"][0]["candidates"][0]["text"])
        self.assertEqual("שלום", payload["final_regions"][0]["final_transcription"])
        self.assertEqual(
            "שלם[?]", payload["final_regions"][0]["contextual_evidence"]["final_transcription"]
        )

    def test_full_mode_uses_four_candidates_and_saves_all_enhancements(self):
        page = Image.new("RGB", (800, 1000), "white")
        with tempfile.TemporaryDirectory() as output_dir:
            tasks, images = pipeline.prepare_region_images(
                page,
                {"label": "test", "box": [200, 300, 200, 100]},
                output_dir,
                "full",
                900,
                150,
            )

        self.assertEqual(list(pipeline.CORE_OCR_VIEWS), [name for name, _ in tasks])
        self.assertEqual(8, len(images))

    def test_handwriting_context_keeps_compact_field_local(self):
        page = Image.new("RGB", (3400, 4400), "white")

        expanded = pipeline.add_handwriting_context(page, [1579, 559, 180, 308])

        self.assertGreater(expanded[2], 180)
        self.assertLess(expanded[2], 260)
        self.assertEqual(559, expanded[1])
        self.assertEqual(867, expanded[1] + expanded[3])

    def test_refiner_does_not_chain_compact_field_across_form(self):
        page_path = Path(
            "analysis_results/אישור מסירה -דירה 8.pdf__2026-07-12_09-52-52/"
            "page_1_render.png"
        )
        if not page_path.exists():
            self.skipTest("saved form probe is unavailable")
        page = Image.open(page_path)

        refined = pipeline.refine_box_to_ink(page, [1371, 702, 732, 350])

        self.assertLessEqual(refined[2], 1212)

    def test_color_refiner_follows_blue_phone_not_black_redaction(self):
        page_path = Path("sample/חוזר מכר - דף ראשון.png")
        page = Image.open(page_path)

        refined = pipeline.refine_box_to_ink(
            page, [289, 339, 281, 70], color_hint="handwritten phone number"
        )

        self.assertGreater(refined[0], 280)
        self.assertGreater(refined[1] + refined[3], 405)
        self.assertLess(refined[2], 260)

    def test_color_refiner_does_not_treat_yellow_paper_as_pen_ink(self):
        page = Image.new("RGB", (600, 600), (235, 228, 150))
        draw = ImageDraw.Draw(page)
        draw.line((180, 320, 205, 260), fill="black", width=7)
        draw.line((250, 320, 275, 270), fill="black", width=7)

        refined = pipeline.refine_box_to_colored_ink(
            page, [150, 240, 160, 100], 200, 380,
        )

        self.assertIsNone(refined)

    def test_mixed_hebrew_numeric_line_is_not_a_compact_number_field(self):
        label = "handwritten number and Hebrew line"

        self.assertFalse(pipeline.is_compact_handwriting_label(label))
        self.assertIsNone(pipeline.compact_handwriting_kind(label))
        penalty, issues = pipeline.detection_geometry_penalty(
            [{"label": label, "box": [100, 400, 800, 150]}], 1000, 1000,
        )
        self.assertEqual(0, penalty)
        self.assertEqual([], issues)

    @patch.object(pipeline, "_large_black_handwriting_components")
    def test_stroke_edge_preservation_does_not_chain_into_neighbors(self, components):
        components.return_value = [
            (170, 100, 210, 160),
            (205, 100, 220, 160),
            (100, 200, 130, 260),
        ]
        page = Image.new("RGB", (400, 400), "white")
        original = [{"label": "handwritten line", "box": [100, 100, 100, 60]}]

        regions = pipeline.preserve_black_ink_edges(page, original)

        self.assertEqual([100, 100, 110, 60], regions[0]["box"])
        self.assertEqual([100, 100, 100, 60], original[0]["box"])

    @patch("handwriting_pipeline._bearer", return_value="token")
    @patch("handwriting_pipeline.requests.Session.post")
    def test_responses_retries_unsupported_parameter_with_zero_retries(
        self, post, _bearer
    ):
        rejected = Mock(status_code=400, text="Unsupported parameter: seed")
        accepted = Mock(status_code=200)
        accepted.json.return_value = {
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": "ok"}],
            }]
        }
        post.side_effect = [rejected, accepted]
        pipeline._unsupported_params.discard("seed")

        result = pipeline._responses_text(
            [{"type": "input_text", "text": "test"}], retries=0
        )

        self.assertEqual(result, "ok")
        self.assertEqual(post.call_count, 2)

    def test_structured_vote_requires_strict_majority(self):
        result = pipeline.aggregate_votes([
            "053329[?]", "053329[?]", "058328", "0583229[?]", "0583228"
        ])

        self.assertIn("[?]", result["final_transcription"])
        self.assertNotEqual(result["final_transcription"], "053329")

    def test_detection_geometry_penalizes_page_wide_number_field(self):
        bad = [{"label": "handwritten number", "box": [261, 626, 2137, 226]}]
        good = [{"label": "handwritten number", "box": [1371, 702, 732, 350]}]

        bad_penalty, bad_issues = pipeline.detection_geometry_penalty(bad, 3400, 4400)
        good_penalty, good_issues = pipeline.detection_geometry_penalty(good, 3400, 4400)

        self.assertGreater(bad_penalty, good_penalty)
        self.assertTrue(bad_issues)
        self.assertFalse(good_issues)

    def test_detection_retry_repairs_only_invalid_region(self):
        original = [
            {"label": "handwritten number", "box": [261, 626, 2137, 226]},
            {"label": "handwritten note line", "box": [500, 1300, 2300, 400]},
        ]
        retry = [
            {"label": "handwritten number", "box": [1371, 702, 732, 350]},
            {"label": "handwritten note line", "box": [900, 1800, 1500, 400]},
        ]

        repaired = pipeline.repair_invalid_detection_regions(original, retry, 3400, 4400)

        self.assertEqual(retry[0], repaired[0])
        self.assertEqual(original[1], repaired[1])

    @patch.object(pipeline, "detect_colored_pen_regions")
    @patch.object(pipeline, "add_handwriting_context", side_effect=lambda _image, box: box)
    @patch.object(
        pipeline, "refine_box_to_ink",
        side_effect=lambda _image, box, color_hint=None: box,
    )
    @patch.object(pipeline, "_responses_call")
    def test_detection_keeps_initial_regions_when_geometry_retry_fails(
        self, responses_call, _refine, _context, colored_regions,
    ):
        responses_call.side_effect = [
            {"regions": [{
                "label": "handwritten number",
                "box": [100, 500, 800, 100],
            }]},
            RuntimeError("[GPT] all endpoints failed: secondary empty output"),
        ]
        colored_regions.side_effect = lambda _image, regions: regions

        regions = pipeline.detect_handwriting_regions(
            Image.new("RGB", (1000, 1000), "white")
        )

        self.assertEqual(1, len(regions))
        self.assertEqual("handwritten number", regions[0]["label"])
        self.assertEqual(2, responses_call.call_count)

    @patch.object(pipeline, "detect_colored_pen_regions")
    @patch.object(pipeline, "add_handwriting_context", side_effect=lambda _image, box: box)
    @patch.object(
        pipeline, "refine_box_to_ink",
        side_effect=lambda _image, box, color_hint=None: box,
    )
    @patch.object(pipeline, "_responses_call")
    def test_detection_keeps_initial_regions_when_lower_pass_fails(
        self, responses_call, _refine, _context, colored_regions,
    ):
        responses_call.side_effect = [
            {"regions": [{
                "label": "handwritten note",
                "box": [100, 100, 200, 50],
            }]},
            RuntimeError("[GPT] all endpoints failed: primary read timeout"),
            RuntimeError("[GPT] all endpoints failed: secondary empty output"),
            RuntimeError("[GPT] all endpoints failed: primary empty output"),
            RuntimeError("[GPT] all endpoints failed: secondary read timeout"),
            RuntimeError("[GPT] all endpoints failed: primary read timeout"),
        ]
        colored_regions.side_effect = lambda _image, regions: regions

        regions = pipeline.detect_handwriting_regions(
            Image.new("RGB", (1000, 1000), "white")
        )

        self.assertEqual(1, len(regions))
        self.assertEqual("handwritten note", regions[0]["label"])
        self.assertEqual(6, responses_call.call_count)

    @patch.object(pipeline, "detect_colored_pen_regions")
    @patch.object(pipeline, "add_handwriting_context", side_effect=lambda _image, box: box)
    @patch.object(
        pipeline, "refine_box_to_ink",
        side_effect=lambda _image, box, color_hint=None: box,
    )
    @patch.object(pipeline, "_responses_call")
    def test_detection_maps_tiled_coverage_to_full_page(
        self, responses_call, _refine, _context, colored_regions,
    ):
        responses_call.side_effect = [
            {"regions": [{
                "label": "initial handwriting",
                "box": [100, 100, 200, 50],
            }]},
            {"regions": [{
                "label": "top handwriting",
                "box": [100, 100, 200, 50],
            }]},
            {"regions": [{
                "label": "middle handwriting",
                "box": [100, 100, 200, 50],
            }]},
            {"regions": [{
                "label": "bottom handwriting",
                "box": [100, 100, 200, 50],
            }]},
            {"regions": [{
                "label": "lower edge handwriting",
                "box": [100, 100, 200, 50],
            }]},
            {"regions": [{
                "label": "bottom edge handwriting",
                "box": [100, 100, 200, 50],
            }]},
        ]
        colored_regions.side_effect = lambda _image, regions: regions

        regions = pipeline.detect_handwriting_regions(
            Image.new("RGB", (1000, 1000), "white"), det_max_side=1000,
        )

        self.assertEqual(6, responses_call.call_count)
        for call in responses_call.call_args_list:
            evidence = call.args[0]
            self.assertEqual(
                2,
                sum(item.get("type") == "input_image" for item in evidence),
            )
        self.assertEqual(
            [94, 334, 514, 694, 854],
            [region["box"][1] for region in regions],
        )

    @patch.object(pipeline, "detect_colored_pen_regions")
    @patch.object(pipeline, "add_handwriting_context", side_effect=lambda _image, box: box)
    @patch.object(
        pipeline, "refine_box_to_ink",
        side_effect=lambda _image, box, color_hint=None: box,
    )
    @patch.object(pipeline, "_responses_call")
    def test_tiled_coverage_recovers_when_full_page_detection_fails(
        self, responses_call, _refine, _context, colored_regions,
    ):
        responses_call.side_effect = [
            RuntimeError("[GPT] all endpoints failed: primary read timeout"),
            {"regions": [{"label": "top note", "box": [100, 100, 200, 50]}]},
            {"regions": [{"label": "middle note", "box": [100, 100, 200, 50]}]},
            {"regions": [{"label": "lower note", "box": [100, 100, 200, 50]}]},
            {"regions": []},
            {"regions": []},
        ]
        colored_regions.side_effect = lambda _image, regions: regions

        regions = pipeline.detect_handwriting_regions(
            Image.new("RGB", (1000, 1000), "white"), det_max_side=1000,
        )

        self.assertEqual(6, responses_call.call_count)
        self.assertEqual(3, len(regions))
        self.assertEqual(
            [94, 334, 514],
            [region["box"][1] for region in regions],
        )

    @patch.object(pipeline, "refine_box_to_colored_ink", return_value=None)
    def test_compact_detection_refinement_uses_narrow_vertical_band(
        self, _colored_refinement,
    ):
        page = Image.new("RGB", (1000, 1000), "white")
        draw = ImageDraw.Draw(page)
        draw.rectangle((130, 420, 170, 455), fill="black")
        box = [100, 400, 200, 100]

        compact = pipeline.refine_box_to_ink(
            page, box, color_hint="handwritten date",
        )
        free_text = pipeline.refine_box_to_ink(
            page, box, color_hint="handwritten note",
        )

        self.assertEqual(388, compact[1])
        self.assertEqual(124, compact[3])
        self.assertEqual(365, free_text[1])
        self.assertEqual(170, free_text[3])

    @patch.object(pipeline, "detect_colored_pen_regions")
    @patch.object(pipeline, "refine_box_to_ink")
    @patch.object(pipeline, "_responses_call")
    def test_detection_keeps_compact_model_box_local(
        self, responses_call, refine, colored_regions,
    ):
        responses_call.return_value = {"regions": [{
            "label": "handwritten date",
            "box": [100, 400, 200, 50],
        }]}
        colored_regions.side_effect = lambda _image, regions: regions

        regions = pipeline.detect_handwriting_regions(
            Image.new("RGB", (1000, 1000), "white"), det_max_side=1000,
        )

        refine.assert_not_called()
        self.assertEqual([92, 394, 216, 62], regions[0]["box"])

    @patch.object(pipeline, "enhance_pen_stroke_isolation")
    def test_compact_black_pen_cleanup_drops_empty_candidate(self, isolation):
        page = Image.new("RGB", (700, 700), "white")
        ImageDraw.Draw(page).rectangle((100, 100, 300, 220), outline="black")
        isolation.return_value = Image.new("L", page.size, 255)

        cleaned = pipeline.clean_compact_black_pen_regions(page, [{
            "label": "handwritten guarantee details",
            "box": [100, 100, 200, 120],
        }])

        self.assertEqual([], cleaned)

    @patch.object(pipeline, "enhance_pen_stroke_isolation")
    def test_compact_black_pen_cleanup_keeps_less_clipped_duplicate(
        self, isolation,
    ):
        page = Image.new("RGB", (700, 700), "white")
        support = Image.new("L", page.size, 255)
        for image in (page, support):
            draw = ImageDraw.Draw(image)
            for left in (130, 175, 220, 265):
                draw.rectangle((left, 120, left + 20, 165), fill="black")
        isolation.return_value = support
        complete = {"label": "id number", "box": [100, 100, 220, 100]}
        clipped = {
            "label": "handwritten id number",
            "box": [100, 70, 220, 85],
        }

        cleaned = pipeline.clean_compact_black_pen_regions(
            page, [complete, clipped],
        )

        self.assertEqual([complete], cleaned)

    @patch.object(pipeline, "enhance_pen_stroke_isolation")
    def test_compact_black_pen_cleanup_keeps_distinct_generic_numbers(
        self, isolation,
    ):
        page = Image.new("RGB", (700, 700), "white")
        support = Image.new("L", page.size, 255)
        for image in (page, support):
            draw = ImageDraw.Draw(image)
            for left in (130, 175, 220, 265):
                draw.line((left, 180, left + 20, 120), fill="black", width=10)
        isolation.return_value = support
        regions = [
            {"label": "handwritten building number", "box": [100, 100, 220, 100]},
            {"label": "handwritten apartment number", "box": [110, 110, 210, 100]},
        ]

        cleaned = pipeline.clean_compact_black_pen_regions(page, regions)

        self.assertEqual(regions, cleaned)

    @patch.object(pipeline, "enhance_pen_stroke_isolation")
    def test_compact_black_pen_cleanup_relocates_to_glyph_band(self, isolation):
        page = Image.new("RGB", (800, 800), "white")
        support = Image.new("L", page.size, 255)
        page_draw = ImageDraw.Draw(page)
        support_draw = ImageDraw.Draw(support)
        page_draw.rectangle((400, 300, 600, 380), fill="black")
        for row in (320, 340, 360):
            support_draw.rectangle((410, row, 590, row + 1), fill="black")
        page_draw.line((360, 450, 640, 450), fill="black", width=5)
        support_draw.line((360, 450, 640, 450), fill="black", width=5)
        for left in (440, 485, 530):
            page_draw.rectangle((left, 500, left + 22, 550), fill="black")
            support_draw.rectangle((left, 500, left + 22, 550), fill="black")
        isolation.return_value = support

        cleaned = pipeline.clean_compact_black_pen_regions(page, [{
            "label": "handwritten date",
            "box": [400, 300, 200, 80],
        }])

        self.assertEqual(1, len(cleaned))
        self.assertGreaterEqual(cleaned[0]["box"][1], 490)
        self.assertLessEqual(cleaned[0]["box"][1] + cleaned[0]["box"][3], 560)

    def test_black_handwriting_peer_supplement_recovers_omitted_row_field(self):
        page = Image.new("RGB", (1000, 1500), "white")
        draw = ImageDraw.Draw(page)
        for left in (180, 230, 280):
            draw.line((left, 770, left + 28, 700), fill="black", width=12)
            draw.line((left + 10, 735, left + 36, 760), fill="black", width=10)
        for left in (650, 700, 750):
            draw.line((left, 770, left + 28, 700), fill="black", width=12)
            draw.line((left + 10, 735, left + 36, 760), fill="black", width=10)
        regions = [{
            "label": "handwritten name",
            "box": [150, 680, 220, 120],
        }]

        supplemented = pipeline.supplement_black_handwriting_peer_regions(
            page, regions,
        )

        self.assertEqual(2, len(supplemented))
        self.assertTrue(all(
            region["label"] == "handwritten declaration peer"
            for region in supplemented
        ))
        self.assertLess(supplemented[0]["box"][0], 200)
        self.assertGreater(supplemented[1]["box"][0], 600)

    def test_black_handwriting_peer_supplement_ignores_colored_page(self):
        page = Image.new("RGB", (1000, 1500), "white")
        draw = ImageDraw.Draw(page)
        for row in range(50, 1450, 20):
            draw.line((50, row, 950, row), fill=(20, 60, 180), width=3)
        regions = [{
            "label": "handwritten name",
            "box": [150, 680, 220, 120],
        }]

        supplemented = pipeline.supplement_black_handwriting_peer_regions(
            page, regions,
        )

        self.assertEqual(regions, supplemented)

    @patch.object(pipeline, "_large_black_handwriting_components")
    def test_uncovered_group_supplement_adds_text_but_not_signature(
        self, components,
    ):
        components.return_value = [
            (100, 300, 130, 370), (150, 305, 180, 365), (200, 310, 230, 360),
            (500, 500, 540, 610), (570, 510, 610, 600),
            (700, 520, 730, 590), (750, 525, 780, 585), (800, 530, 830, 580),
        ]
        regions = [
            {"label": "handwritten declaration peer", "box": [80, 280, 180, 110]},
            {"label": "handwritten declaration peer", "box": [300, 280, 180, 110]},
        ]

        supplemented = pipeline.supplement_uncovered_black_handwriting_groups(
            Image.new("RGB", (1000, 1000), "white"), regions,
        )

        self.assertEqual(3, len(supplemented))
        self.assertEqual("handwritten stroke group", supplemented[-1]["label"])
        self.assertGreater(supplemented[-1]["box"][0], 650)

    def test_detection_dedup_drops_oversized_containing_compact_field(self):
        regions = [
            {"label": "handwritten number in center upper field", "box": [261, 626, 2137, 226]},
            {"label": "handwritten number 8", "box": [1371, 702, 732, 350]},
            {"label": "handwritten note line", "box": [554, 1335, 2350, 500]},
        ]

        deduplicated = pipeline.deduplicate_detection_regions(regions, 3400, 4400)

        self.assertNotIn(regions[0], deduplicated)
        self.assertIn(regions[1], deduplicated)
        self.assertIn(regions[2], deduplicated)

    def test_detection_dedup_keeps_distinct_overlapping_note_lines(self):
        regions = [
            {"label": "warranty note upper line", "box": [554, 1335, 2350, 500]},
            {"label": "warranty note lower line", "box": [364, 1589, 2607, 673]},
        ]

        deduplicated = pipeline.deduplicate_detection_regions(regions, 3400, 4400)

        self.assertEqual(regions, deduplicated)

    def test_detection_dedup_merges_nested_form_field_crops(self):
        regions = [
            {"label": "handwritten number in upper left field", "box": [415, 669, 911, 615]},
            {"label": "handwritten number in left address field", "box": [599, 824, 519, 357]},
            {"label": "handwritten number in center top field", "box": [1285, 660, 842, 615]},
            {"label": "handwritten street address entry", "box": [1369, 785, 1045, 395]},
        ]

        deduplicated = pipeline.deduplicate_detection_regions(regions, 3400, 4400)

        self.assertEqual(2, len(deduplicated))
        self.assertEqual([599, 824, 519, 357], deduplicated[0]["box"])
        self.assertEqual([1369, 785, 1045, 395], deduplicated[1]["box"])

    def test_detection_dedup_merges_same_sized_duplicate_fields(self):
        regions = [
            {"label": "handwritten number in upper left field", "box": [431, 658, 895, 642]},
            {"label": "handwritten number in second upper left field", "box": [433, 745, 893, 657]},
        ]

        deduplicated = pipeline.deduplicate_detection_regions(regions, 3400, 4400)

        self.assertEqual(1, len(deduplicated))
        self.assertEqual([431, 658, 895, 642], deduplicated[0]["box"])

    def test_detection_dedup_drops_same_label_clipped_fragment(self):
        regions = [
            {
                "label": "small handwritten note at top right",
                "box": [688, 0, 68, 38],
            },
            {
                "label": "small handwritten note at top right",
                "box": [672, 18, 117, 58],
            },
        ]

        deduplicated = pipeline.deduplicate_detection_regions(regions, 1008, 943)

        self.assertEqual([regions[1]], deduplicated)

    def test_detection_dedup_overlap_chain_does_not_grow_crop(self):
        regions = [
            {"label": "field one", "box": [400, 700, 700, 300]},
            {"label": "field one duplicate", "box": [430, 720, 680, 290]},
            {"label": "field one retry", "box": [460, 740, 650, 280]},
        ]

        deduplicated = pipeline.deduplicate_detection_regions(regions, 3400, 4400)

        self.assertEqual(1, len(deduplicated))
        self.assertEqual([460, 740, 650, 280], deduplicated[0]["box"])

    def test_midpage_handwriting_line_is_not_stretched_across_page(self):
        regions = [
            {"label": "handwritten name entry in mid upper line", "box": [1411, 1352, 929, 336]},
            {"label": "small form field", "box": [1411, 900, 600, 400]},
        ]

        widened = pipeline.widen_midpage_handwriting_lines(regions, 3400, 4400)

        self.assertEqual(regions[0]["box"], widened[0]["box"])
        self.assertEqual(regions[1]["box"], widened[1]["box"])

    def test_lower_detection_pass_runs_when_regions_only_cover_header(self):
        header_regions = [
            {"label": "date", "box": [315, 279, 1013, 529]},
            {"label": "number", "box": [1296, 869, 1309, 368]},
        ]
        body_regions = header_regions + [
            {"label": "warranty note", "box": [408, 1676, 2652, 357]},
        ]

        self.assertTrue(pipeline.needs_lower_detection_pass(header_regions, 4400))
        self.assertFalse(pipeline.needs_lower_detection_pass(body_regions, 4400))

    def test_detector_default_keeps_more_detail(self):
        self.assertEqual(3000, pipeline.detect_handwriting_regions.__defaults__[0])

    def test_long_line_context_stays_local(self):
        page = Image.new("RGB", (3400, 4400), "white")

        expanded = pipeline.add_handwriting_context(page, [437, 1569, 2380, 395])

        self.assertGreater(expanded[2], 2380)
        self.assertLess(expanded[2], 2500)
        self.assertEqual(1569, expanded[1])
        self.assertEqual(1964, expanded[1] + expanded[3])

    def test_compact_field_context_uses_local_not_page_padding(self):
        page = Image.new("RGB", (3308, 4678), "white")

        expanded = pipeline.add_handwriting_context(page, [500, 600, 220, 120])

        self.assertEqual([483, 600, 254, 120], expanded)

    def test_colorless_false_positive_is_removed_on_colored_pen_page(self):
        page = Image.new("RGB", (1000, 1000), "white")
        draw = ImageDraw.Draw(page)
        draw.rectangle((100, 100, 400, 120), fill=(30, 80, 190))
        regions = [
            {"label": "blue handwritten note", "box": [80, 80, 350, 80]},
            {"label": "printed legal line", "box": [500, 500, 350, 80]},
        ]

        filtered = pipeline.filter_colorless_regions_on_colored_page(page, regions)

        self.assertEqual([regions[0]], filtered)

    def test_colored_refiner_does_not_absorb_adjacent_row(self):
        page = Image.new("RGB", (1000, 1000), "white")
        draw = ImageDraw.Draw(page)
        draw.rectangle((200, 300, 500, 320), fill=(25, 70, 200))
        draw.rectangle((200, 500, 500, 520), fill=(25, 70, 200))

        refined = pipeline.refine_box_to_ink(
            page, [180, 275, 350, 80], color_hint="handwritten date"
        )

        self.assertLess(refined[1] + refined[3], 400)
        self.assertGreaterEqual(refined[1], 260)

    def test_colored_page_does_not_refine_printed_box_to_black_ink(self):
        page = Image.new("RGB", (1000, 1000), "white")
        draw = ImageDraw.Draw(page)
        draw.rectangle((100, 100, 400, 120), fill=(25, 70, 200))
        draw.rectangle((500, 500, 900, 520), fill="black")
        original = [480, 470, 440, 100]

        refined = pipeline.refine_box_to_ink(page, original, color_hint="field")

        self.assertEqual(original, refined)

    def test_colored_geometry_splits_rows_and_drops_printed_region(self):
        page = Image.new("RGB", (1000, 1000), "white")
        draw = ImageDraw.Draw(page)
        draw.rectangle((200, 300, 500, 320), fill=(25, 70, 200))
        draw.rectangle((220, 500, 520, 520), fill=(25, 70, 200))
        draw.rectangle((100, 700, 900, 720), fill="black")
        regions = [
            {"label": "two handwritten rows", "box": [150, 250, 450, 350]},
            {"label": "printed clause", "box": [80, 650, 840, 120]},
        ]

        tightened = pipeline.tighten_regions_to_colored_ink(page, regions)

        self.assertEqual(2, len(tightened))
        self.assertLess(tightened[0]["box"][1] + tightened[0]["box"][3], 400)
        self.assertGreater(tightened[1]["box"][1], 400)

    def test_black_form_geometry_snaps_oversized_crop_to_handwritten_row(self):
        page = Image.new("RGB", (1000, 1000), "white")
        draw = ImageDraw.Draw(page)
        for text_y in (245, 265, 285):
            draw.text((140, text_y), "printed form text " * 8, fill="black")
        for rule_y in (320, 440, 560, 680):
            draw.line((100, rule_y, 900, rule_y), fill="black", width=2)
        for stroke_x, stroke_height in ((220, 66), (340, 78), (510, 62), (690, 72)):
            draw.line(
                (stroke_x, 535, stroke_x + 22, 535 - stroke_height),
                fill="black",
                width=7,
            )
        for stroke_x in (260, 470, 650):
            draw.line(
                (stroke_x, 655, stroke_x + 18, 600),
                fill="black",
                width=7,
            )
        regions = [{"label": "handwritten amount row", "box": [80, 250, 840, 390]}]

        tightened = pipeline.tighten_regions_to_black_ink_rows(page, regions)

        self.assertEqual(1, len(tightened))
        left, top, width, height = tightened[0]["box"]
        self.assertGreaterEqual(top, 430)
        self.assertLessEqual(top + height, 570)
        self.assertGreater(left, 150)
        self.assertLess(left + width, 800)

    def test_black_form_geometry_preserves_multiline_blocks(self):
        page = Image.new("RGB", (1000, 1000), "white")
        draw = ImageDraw.Draw(page)
        for rule_y in range(180, 901, 120):
            draw.line((80, rule_y, 920, rule_y), fill="black", width=2)
        for row_y in (275, 395, 515, 635, 755, 875):
            for stroke_x in (180, 290, 410, 540, 680, 790):
                draw.line(
                    (stroke_x, row_y, stroke_x + 24, row_y - 64),
                    fill="black", width=7,
                )
        regions = [
            {"label": "Upper handwritten block with date and recipient",
             "box": [100, 190, 800, 220]},
            {"label": "Middle handwritten note and numeric details",
             "box": [100, 430, 800, 220]},
            {"label": "Lower handwritten closing and contact details",
             "box": [100, 670, 800, 220]},
        ]

        tightened = pipeline.tighten_regions_to_black_ink_rows(page, regions)

        self.assertEqual(regions, tightened)
        self.assertFalse(pipeline.is_compact_handwriting_label(regions[0]["label"]))

    @patch.object(pipeline, "_responses_call")
    @patch.object(pipeline, "refine_box_to_ink")
    @patch.object(pipeline, "detect_black_ink_form_rows")
    @patch.object(pipeline, "detect_colored_pen_regions", return_value=[])
    def test_detection_preserves_block_extent_through_form_postprocessing(
        self, _colored, form_rows, refine, responses,
    ):
        page = Image.new("RGB", (1000, 1000), "white")
        model_box = [100, 200, 800, 600]
        responses.return_value = {"regions": [{
            "label": "Handwritten block with date and recipient",
            "box": model_box,
        }]}
        form_rows.return_value = [
            {"label": f"black handwritten form row {index}", "box": [200, top, 600, 80]}
            for index, top in enumerate((250, 450, 650), 1)
        ]
        refine.return_value = [200, 250, 600, 80]

        regions = pipeline.detect_handwriting_regions(page)

        self.assertTrue(any(
            region["box"][1] <= model_box[1]
            and region["box"][1] + region["box"][3] >= model_box[1] + model_box[3]
            for region in regions
        ))
        refine.assert_not_called()

    @patch.object(pipeline, "detect_colored_pen_regions")
    @patch.object(pipeline, "_responses_call")
    def test_paragraph_detection_is_not_replaced_by_blue_margin_fragments(
        self, responses, colored_regions,
    ):
        responses.return_value = {"regions": [
            {"label": "handwritten paragraph", "box": [100, 100, 800, 350]},
            {"label": "handwritten closing paragraph", "box": [100, 550, 800, 350]},
        ]}
        colored_regions.return_value = [
            {"label": "colored handwriting", "box": [950, 200, 30, 20]},
            {"label": "colored handwriting", "box": [950, 600, 30, 20]},
        ]

        regions = pipeline.detect_handwriting_regions(
            Image.new("RGB", (1000, 1000), "white"),
        )

        self.assertEqual(2, len(regions))
        self.assertTrue(all(region["box"][2] >= 800 for region in regions))
        colored_regions.assert_not_called()

    def test_black_form_geometry_assigns_overlapping_regions_to_distinct_rows(self):
        page = Image.new("RGB", (1000, 1000), "white")
        draw = ImageDraw.Draw(page)
        for rule_y in (300, 420, 540):
            draw.line((100, rule_y, 900, rule_y), fill="black", width=2)
        for row_y in (395, 515):
            for stroke_x in (220, 360, 520, 700):
                draw.line(
                    (stroke_x, row_y, stroke_x + 22, row_y - 62),
                    fill="black",
                    width=7,
                )
        regions = [
            {"label": "handwritten row 1", "box": [80, 280, 840, 230]},
            {"label": "handwritten row 2", "box": [80, 310, 840, 230]},
        ]

        tightened = pipeline.tighten_regions_to_black_ink_rows(page, regions)

        self.assertEqual(2, len(tightened))
        first_top = tightened[0]["box"][1]
        second_top = tightened[1]["box"][1]
        self.assertLess(first_top, second_top)
        self.assertGreaterEqual(second_top - first_top, 90)

    def test_black_form_geometry_keeps_model_anchor_and_cross_rule_strokes(self):
        page = Image.new("RGB", (1000, 1000), "white")
        draw = ImageDraw.Draw(page)
        for rule_y in (300, 420, 540):
            draw.line((100, rule_y, 900, rule_y), fill="black", width=2)
        for stroke_x in (220, 360, 520):
            draw.line((stroke_x, 410, stroke_x + 22, 280), fill="black", width=7)
        for stroke_x in range(200, 801, 60):
            draw.line((stroke_x, 530, stroke_x + 22, 455), fill="black", width=10)
        regions = [{
            "label": "handwritten Hebrew line",
            "box": [80, 260, 840, 290],
            "_model_box": [180, 275, 400, 140],
        }]

        tightened = pipeline.tighten_regions_to_black_ink_rows(page, regions)

        left, top, width, height = tightened[0]["box"]
        self.assertLessEqual(top, 275)
        self.assertGreaterEqual(top + height, 415)
        self.assertLess(top + height, 450)
        self.assertLessEqual(left, 180)
        self.assertGreaterEqual(left + width, 580)

    @patch.object(pipeline, "_responses_call")
    def test_black_note_detection_covers_all_strong_strokes(self, responses):
        source = Path("sample/bank discount samples/העברה-קוקי.pdf")
        if not source.exists():
            self.skipTest("local handwritten note fixture is unavailable")
        labels_and_boxes = [
            ("handwritten date", [738, 240, 394, 240]),
            ("short handwritten Hebrew entry", [2265, 576, 142, 105]),
            ("handwritten Hebrew line", [1585, 680, 752, 191]),
            ("handwritten Hebrew line", [1510, 1052, 719, 141]),
            ("handwritten Hebrew and numeric line", [648, 1232, 1777, 189]),
            ("handwritten Hebrew continuation line", [1420, 1390, 982, 164]),
            ("handwritten Hebrew line", [1135, 1600, 1320, 205]),
            ("handwritten number and Hebrew line", [849, 1758, 1484, 198]),
            ("short handwritten Hebrew heading", [1196, 2117, 444, 91]),
            ("handwritten Hebrew line", [977, 2190, 731, 166]),
            ("handwritten phone number and Hebrew text", [919, 2330, 788, 177]),
        ]
        responses.return_value = {
            "regions": [{"label": label, "box": box} for label, box in labels_and_boxes],
        }
        page = pipeline.render_pdf_page(str(source), 1, 400).convert("RGB")

        regions = pipeline.detect_handwriting_regions(page)

        self.assertEqual(11, len(regions))
        self.assertTrue(all("_model_box" not in region for region in regions))
        mask = Image.new("L", page.size, 0)
        draw = ImageDraw.Draw(mask)
        for region in regions:
            x, y, width, height = region["box"]
            draw.rectangle((x, y, x + width - 1, y + height - 1), fill=255)
        components = pipeline._large_black_handwriting_components(page)
        self.assertTrue(components)
        for left, top, right, bottom in components:
            with self.subTest(component=(left, top, right, bottom)):
                pixels = mask.crop((left, top, right, bottom))
                coverage = sum(pixels.getdata()) / (
                    255 * (right - left) * (bottom - top)
                )
                self.assertGreaterEqual(coverage, 0.95)

    def test_black_form_detector_recovers_rows_missing_from_model_detection(self):
        page = Image.new("RGB", (1000, 1000), "white")
        draw = ImageDraw.Draw(page)
        for rule_y in (180, 300, 420, 540, 660, 780, 900):
            draw.line((80, rule_y, 920, rule_y), fill="black", width=2)
        for row_y in (275, 515, 875):
            for stroke_x in (180, 290, 410, 540, 680, 790):
                draw.line(
                    (stroke_x, row_y, stroke_x + 24, row_y - 64),
                    fill="black",
                    width=7,
                )
        draw.text((150, 335), "printed header", fill="black")
        draw.text((150, 695), "printed header", fill="black")

        detected = pipeline.detect_black_ink_form_rows(page)

        self.assertEqual(3, len(detected))
        self.assertEqual([175, 415, 775], [region["box"][1] for region in detected])
        for region in detected:
            left, _top, width, height = region["box"]
            self.assertGreater(left, 100)
            self.assertLess(left + width, 900)
            self.assertLessEqual(height, 140)

    def test_colored_line_fragments_merge_without_merging_rows(self):
        regions = [
            {"label": "left fragment", "box": [100, 300, 200, 80]},
            {"label": "right fragment", "box": [325, 305, 240, 75]},
            {"label": "next row", "box": [100, 460, 465, 80]},
        ]

        merged = pipeline.merge_colored_line_fragments(regions)

        self.assertEqual(2, len(merged))
        self.assertEqual([100, 300, 465, 80], merged[0]["box"])
        self.assertEqual(regions[2]["box"], merged[1]["box"])

    def test_black_form_row_keeps_short_trailing_digits(self):
        page = Image.new("RGB", (1000, 1000), "white")
        draw = ImageDraw.Draw(page)
        for rule_y in (180, 300, 420, 540, 660, 780, 900):
            draw.line((80, rule_y, 920, rule_y), fill="black", width=2)
        for row_y in (275, 515, 875):
            for stroke_x in (180, 290, 410, 540, 680, 790):
                draw.line(
                    (stroke_x, row_y, stroke_x + 24, row_y - 64),
                    fill="black", width=7,
                )
            draw.rectangle((850, row_y - 20, 870, row_y), outline="black", width=4)

        detected = pipeline.detect_black_ink_form_rows(page)

        self.assertEqual(3, len(detected))
        for region in detected:
            self.assertGreaterEqual(region["box"][0] + region["box"][2], 871)

    def test_colored_page_finds_pen_region_model_missed(self):
        page = Image.new("RGB", (1000, 1000), "white")
        draw = ImageDraw.Draw(page)
        draw.rectangle((150, 250, 400, 280), fill=(25, 70, 200))
        draw.rectangle((600, 700, 850, 730), fill=(25, 70, 200))
        draw.rectangle((100, 500, 900, 520), fill="black")
        model_regions = [
            {"label": "model found first row", "box": [120, 220, 320, 100]},
        ]

        detected = pipeline.detect_colored_pen_regions(page, model_regions)

        self.assertEqual(2, len(detected))
        self.assertEqual("model found first row", detected[0]["label"])
        self.assertEqual("colored handwriting", detected[1]["label"])

    def test_overlapping_long_lines_are_split_vertically(self):
        regions = [
            {"label": "line 1", "box": [484, 1430, 2484, 522]},
            {"label": "line 2", "box": [366, 1632, 2522, 522]},
            {"label": "compact field", "box": [1528, 678, 749, 347]},
        ]

        separated = pipeline.separate_overlapping_line_regions(regions)

        upper_end = separated[0]["box"][1] + separated[0]["box"][3]
        lower_start = separated[1]["box"][1]
        self.assertGreaterEqual(separated[0]["box"][3], round(522 * 0.70))
        self.assertGreaterEqual(separated[1]["box"][3], round(522 * 0.70))
        self.assertLess(upper_end - lower_start, 0.35 * 522)
        self.assertEqual(regions[2]["box"], separated[2]["box"])

    def test_tall_overlapping_note_lines_are_also_split(self):
        regions = [
            {"label": "line 1", "box": [484, 1377, 2484, 531]},
            {"label": "line 2", "box": [364, 1591, 2607, 673]},
            {"label": "number", "box": [287, 721, 831, 444]},
        ]

        separated = pipeline.separate_overlapping_line_regions(regions)

        upper_end = separated[0]["box"][1] + separated[0]["box"][3]
        lower_start = separated[1]["box"][1]
        self.assertGreaterEqual(separated[0]["box"][3], round(531 * 0.70))
        self.assertGreaterEqual(separated[1]["box"][3], round(673 * 0.70))
        self.assertLess(upper_end - lower_start, 0.35 * 531)
        self.assertEqual(regions[2]["box"], separated[2]["box"])

    def test_dense_handwritten_page_is_grouped_into_paragraph_blocks(self):
        regions = [
            {
                "label": f"handwritten line {index + 1}",
                "box": [250, 600 + index * 180, 2900, 150],
            }
            for index in range(21)
        ]

        grouped = pipeline.group_dense_handwriting_lines(regions, 3541, 4911)

        self.assertEqual(3, len(grouped))
        self.assertEqual("handwritten paragraph block 1", grouped[0]["label"])
        self.assertLess(grouped[0]["box"][1], grouped[1]["box"][1])
        self.assertLess(grouped[1]["box"][1], grouped[2]["box"][1])

    def test_form_fields_are_not_grouped_as_dense_handwriting(self):
        regions = [
            {"label": f"field {index}", "box": [100 + index * 200, 500, 300, 300]}
            for index in range(8)
        ]

        grouped = pipeline.group_dense_handwriting_lines(regions, 3400, 4400)

        self.assertEqual(regions, grouped)

    def test_detected_paragraph_blocks_are_normalized_without_large_overlap(self):
        regions = [
            {"label": f"handwritten paragraph block {index + 1}", "box": box}
            for index, box in enumerate([
                [199, 592, 3114, 900],
                [223, 1450, 3089, 900],
                [149, 2300, 3149, 900],
                [379, 3150, 2778, 900],
            ])
        ]
        regions.append({"label": "small handwritten note at bottom", "box": [1027, 4100, 1385, 500]})

        normalized = pipeline.normalize_detected_paragraph_blocks(regions, 3541, 4911)

        self.assertEqual(3, len(normalized))
        for upper, lower in zip(normalized, normalized[1:]):
            upper_end = upper["box"][1] + upper["box"][3]
            self.assertLessEqual(upper_end - lower["box"][1], 40)

    @patch("handwriting_pipeline._responses_call")
    def test_ocr_image_leaves_retries_to_batch(self, responses_call):
        responses_call.return_value = {"full_transcription": ""}

        pipeline.ocr_image(Image.new("RGB", (10, 10)), "read", seed=7)

        self.assertEqual(0, responses_call.call_args.kwargs["retries"])
        self.assertEqual(1, responses_call.call_args.kwargs["endpoint_limit"])

    def test_script_conflict_recheck_is_candidate_blind(self):
        page = Image.new("RGB", (800, 1000), "white")
        draw = ImageDraw.Draw(page)
        draw.line((300, 400, 500, 400), fill="black", width=2)
        draw.line((330, 360, 470, 430), fill=(30, 70, 190), width=8)
        response = {
            "script": "hebrew",
            "classification": "handwritten_text",
            "field_type": "name",
            "final_transcription": "דנית ראובן",
            "confidence": 0.61,
        }

        with patch.object(
            pipeline, "_responses_call", return_value=response,
        ) as responses_call:
            result, images = pipeline.script_conflict_region_recheck(
                page,
                {"label": "handwritten acknowledgment entry", "box": [300, 340, 200, 100]},
            )

        prompt_text = "\n".join(
            item.get("text", "")
            for item in responses_call.call_args.args[0]
            if item.get("type") == "input_text"
        )
        self.assertEqual("exact", result["script_consensus"])
        self.assertEqual("דנית ראובן", result["final_transcription"])
        self.assertEqual(2, responses_call.call_count)
        self.assertIn("NO prior OCR strings", prompt_text)
        self.assertNotIn("1242 13", prompt_text)
        self.assertIn("context", images)
        self.assertIn("page", images)
        for call in responses_call.call_args_list:
            self.assertEqual("high", call.kwargs["reasoning"])
            self.assertEqual(1, call.kwargs["endpoint_limit"])

    def test_script_conflict_acceptance_requires_confident_hebrew_name(self):
        current = {
            "classification": "handwritten_text",
            "field_type": "numeric",
            "final_transcription": "1242 13",
        }
        accepted = pipeline.accept_hebrew_numeric_script_recheck(current, {
            "script": "hebrew",
            "script_consensus": "exact",
            "classification": "handwritten_text",
            "field_type": "name",
            "final_transcription": "דנית ראובן",
            "confidence": 0.61,
            "notes": [],
        })

        self.assertEqual("דנית ראובן", accepted["final_transcription"])
        self.assertEqual(
            "hebrew_numeric_script_conflict", accepted["decision_status"]
        )
        self.assertIn("1242 13", accepted["alternative_transcriptions"])
        self.assertTrue(accepted["needs_review"])
        for rejected in (
            {
                "script": "numeric",
                "script_consensus": "disagreed",
                "classification": "handwritten_text",
                "field_type": "numeric",
                "final_transcription": "1242 13",
                "confidence": 0.95,
            },
            {
                "script": "hebrew",
                "script_consensus": "exact",
                "classification": "handwritten_text",
                "field_type": "name",
                "final_transcription": "בן כהן",
                "confidence": 0.54,
            },
            {
                "script": "hebrew",
                "script_consensus": "exact",
                "classification": "handwritten_text",
                "field_type": "name",
                "final_transcription": "דנית 13",
                "confidence": 0.90,
            },
        ):
            self.assertIs(
                current,
                pipeline.accept_hebrew_numeric_script_recheck(current, rejected),
            )

    def test_low_confidence_hebrew_script_replaces_false_digits_for_review(self):
        current = {
            "classification": "handwritten_text",
            "field_type": "numeric",
            "final_transcription": "2142 13",
        }
        recheck = {
            "script": "hebrew",
            "script_consensus": "script_only",
            "classification": "handwritten_text",
            "field_type": "name",
            "final_transcription": "ב[?]לין",
            "confidence": 0.39,
            "notes": [],
        }

        unresolved = pipeline.preserve_uncertain_hebrew_script_conflict(
            current, recheck,
        )

        self.assertEqual("ב[?]לין", unresolved["final_transcription"])
        self.assertEqual("script_conflict_unresolved", unresolved["decision_status"])
        self.assertTrue(unresolved["needs_review"])
        self.assertIn("2142 13", unresolved["alternative_transcriptions"])
        self.assertIs(
            current,
            pipeline.preserve_uncertain_hebrew_script_conflict(
                current,
                {
                    **recheck,
                    "script_consensus": None,
                    "confidence": 0.34,
                },
            ),
        )

    def test_script_conflict_audits_emit_text_only_on_exact_agreement(self):
        first = {
            "script": "hebrew",
            "classification": "handwritten_text",
            "field_type": "name",
            "final_transcription": "ב[?]לין",
            "confidence": 0.39,
            "alternative_transcriptions": [],
        }
        second = {
            "script": "hebrew",
            "classification": "handwritten_text",
            "field_type": "name",
            "final_transcription": "יובל אלון",
            "confidence": 0.68,
            "alternative_transcriptions": [],
        }

        disagreed = pipeline.combine_script_conflict_audits([first, second])
        agreed = pipeline.combine_script_conflict_audits([
            {**second, "confidence": 0.72},
            {**second, "confidence": 0.61},
        ])
        partial = pipeline.combine_script_conflict_audits([
            second,
            {"script": "error", "_error": "read timeout"},
        ])
        numeric = {
            "script": "numeric",
            "classification": "handwritten_text",
            "field_type": "numeric",
            "final_transcription": "1242 13",
            "confidence": 0.81,
        }
        mixed = pipeline.combine_script_conflict_audits([second, numeric])
        numeric_partial = pipeline.combine_script_conflict_audits([
            numeric,
            {"script": "error", "_error": "read timeout"},
        ])
        unavailable = pipeline.combine_script_conflict_audits([
            {"script": "error", "_error": "primary timeout"},
            {"script": "error", "_error": "secondary timeout"},
        ])

        self.assertEqual("script_only", disagreed["script_consensus"])
        self.assertEqual("[?]", disagreed["final_transcription"])
        self.assertIn("ב[?]לין", disagreed["alternative_transcriptions"])
        self.assertIn("יובל אלון", disagreed["alternative_transcriptions"])
        self.assertEqual("exact", agreed["script_consensus"])
        self.assertEqual("יובל אלון", agreed["final_transcription"])
        self.assertEqual(0.61, agreed["confidence"])
        self.assertEqual("script_only", partial["script_consensus"])
        self.assertEqual("[?]", partial["final_transcription"])
        self.assertEqual("read timeout", partial["script_audits"][1]["_error"])
        self.assertEqual("disagreed", mixed["script_consensus"])
        self.assertEqual("[?]", mixed["final_transcription"])
        self.assertEqual("non_hebrew", numeric_partial["script_consensus"])
        self.assertEqual("1242 13", numeric_partial["final_transcription"])
        self.assertEqual("unavailable", unavailable["script_consensus"])
        self.assertEqual("[?]", unavailable["final_transcription"])
        suppressed = pipeline.preserve_uncertain_hebrew_script_conflict(
            numeric,
            unavailable,
        )
        self.assertEqual("[?]", suppressed["final_transcription"])
        self.assertEqual("script_conflict_unresolved", suppressed["decision_status"])

    @patch("handwriting_pipeline._responses_call")
    def test_reconciliation_uses_fallbacks_instead_of_hidden_retries(self, responses_call):
        responses_call.return_value = {"final_transcription": "טקסט", "words": []}
        results = [{"source": "01_original_crop.png", "ocr": {"full_transcription": "טקסט"}}]
        images = [("01_original_crop.png", Image.new("RGB", (10, 10)))]

        pipeline.visual_reconcile_ocr_results(images, results)
        pipeline.reconcile_ocr_results(results)

        self.assertEqual([0, 0], [call.kwargs["retries"] for call in responses_call.call_args_list])

    @patch.object(pipeline, "_responses_call")
    def test_visual_adjudication_includes_candidate_agreement(self, responses_call):
        responses_call.return_value = {"final_transcription": "8 עמינדב", "words": []}
        image = Image.new("RGB", (20, 20), "white")
        results = [
            {"source": "a", "ocr": {"full_transcription": "8 עמינדב"}},
            {"source": "b", "ocr": {"full_transcription": "8 עמינדב"}},
            {"source": "c", "ocr": {"full_transcription": "8 עמיאל"}},
        ]

        pipeline.visual_reconcile_ocr_results([("01_original_crop.png", image)], results)

        prompt = responses_call.call_args.args[0][0]["text"]
        self.assertIn("Deterministic agreement summary", prompt)
        self.assertIn("8 עמינדב", prompt)
        agreement_section = prompt.split(
            "Deterministic agreement summary across candidates", 1
        )[1].split("Return ONLY", 1)[0]
        self.assertIn("8 עמינדב", agreement_section)

    @patch.object(pipeline, "_responses_call")
    def test_visual_adjudication_omits_hash_warning_without_candidate(self, responses_call):
        responses_call.return_value = {"final_transcription": "שלום", "words": []}
        pipeline.visual_reconcile_ocr_results(
            [("01_original_crop.png", Image.new("RGB", (20, 20), "white"))],
            [{"source": "core4_multi", "ocr": {"full_transcription": "שלום"}}],
        )

        prompt = responses_call.call_args.args[0][0]["text"]
        self.assertNotIn("ivrit_hash_v1", prompt)
        self.assertNotIn("word error rate is high", prompt)

    def test_agreement_gate_rejects_single_thickened_name_candidate(self):
        readings = [
            ("01_original_crop.png", "שלום רב"),
            ("02_upscaled_contrast.png", "טיפול [?]"),
            ("06_optional_line_removed.png", "של דמות"),
            ("05_thickened_strokes.png", "אלון זמיר"),
            ("04_denoised.png", "שולם במזומן"),
            ("03_adaptive_threshold.png", "שלום רב"),
            ("07_pen_stroke_isolated.png", "אין דימום"),
        ]
        usable = [
            {"source": source, "ocr": {"full_transcription": text}}
            for source, text in readings
        ]

        guarded = pipeline.guard_visual_reconciliation(
            {"final_transcription": "אלון זמיר", "words": []}, usable
        )

        self.assertNotEqual("אלון זמיר", guarded["final_transcription"])
        self.assertEqual("שלום רב", guarded["final_transcription"])
        self.assertTrue(guarded["needs_review"])
        self.assertEqual(
            ["05_thickened_strokes.png"],
            guarded["agreement_gate"]["supporting_sources"],
        )

    def test_agreement_gate_keeps_reading_supported_by_faithful_view(self):
        usable = [
            {"source": source, "ocr": {"full_transcription": text}}
            for source, text in [
                ("01_original_crop.png", "גולדברג"),
                ("02_upscaled_contrast.png", "גולדברג"),
                ("04_denoised.png", "גולדברג"),
                ("03_adaptive_threshold.png", "גולברג"),
                ("05_thickened_strokes.png", "אלון זמיר"),
            ]
        ]

        guarded = pipeline.guard_visual_reconciliation(
            {"final_transcription": "גולדברג", "words": []}, usable
        )

        self.assertEqual("גולדברג", guarded["final_transcription"])

    def test_agreement_gate_keeps_visual_synthesis_without_exact_candidate(self):
        usable = [
            {"source": source, "ocr": {"full_transcription": text}}
            for source, text in [
                ("01_original_crop.png", "כירה (וולקנו), גז מחמם מים"),
                ("02_upscaled_contrast.png", "מכירה (מוקדמת), צו מטעם"),
                ("04_denoised.png", "שידה (פורמיקה), מיטה וחצי"),
                ("03_adaptive_threshold.png", "סניטריה (אור קרמיקה)"),
                ("05_thickened_strokes.png", "מוצרי חשמל"),
            ]
        ]
        final = {"final_transcription": "סניטריה (קרמיקה), גז, מחמם מים", "words": []}

        guarded = pipeline.guard_visual_reconciliation(final, usable)

        self.assertEqual(final, guarded)

    def test_numeric_order_guard_preserves_faithful_candidate_sequence(self):
        usable = [
            {"source": source, "ocr": {"full_transcription": text}}
            for source, text in [
                ("01_original_crop.png", "5252 731"),
                ("04_denoised.png", "5252 731"),
                ("02_upscaled_contrast.png", "731 5252"),
            ]
        ]

        final = pipeline.finalize_region_semantics(
            {"final_transcription": "731 5252", "words": [], "notes": []}, usable
        )

        self.assertEqual("5252 731", final["final_transcription"])
        self.assertEqual("numeric_order_guard", final["decision_status"])
        self.assertEqual("handwritten_text", final["classification"])

    def test_numeric_field_separates_standalone_x_mark(self):
        final = {
            "classification": "handwritten_text",
            "field_type": "numeric",
            "final_transcription": "X 0,2 00,3,0,0,0",
            "notes": [],
        }

        separated = pipeline.separate_numeric_field_marks(final)

        self.assertEqual("0,2 00,3,0,0,0", separated["final_transcription"])
        self.assertEqual(["x"], separated["marks"])
        self.assertEqual("structured_mark_separation", separated["decision_status"])

    def test_numeric_mark_separation_does_not_change_untyped_field(self):
        final = {
            "classification": "handwritten_text",
            "final_transcription": "012 056 x 6389",
        }

        self.assertIs(final, pipeline.separate_numeric_field_marks(final))

    def test_numeric_s_homoglyph_normalizes_numeric_only_result(self):
        final = {"final_transcription": "SSS6790", "notes": []}

        normalized = pipeline.normalize_numeric_s_homoglyph(final)

        self.assertEqual("5556790", normalized["final_transcription"])
        self.assertEqual(
            "numeric_homoglyph_normalization", normalized["decision_status"]
        )

    def test_numeric_s_homoglyph_rejects_semantic_or_hebrew_text(self):
        semantic = {"final_transcription": "853 O+ 1/8/26"}
        hebrew = {"final_transcription": "סניף S56790"}

        self.assertIs(semantic, pipeline.normalize_numeric_s_homoglyph(semantic))
        self.assertIs(hebrew, pipeline.normalize_numeric_s_homoglyph(hebrew))

    def test_candidate_protection_rejects_reorder_and_information_loss(self):
        usable = [{
            "source": "core4_multi",
            "ocr": {"full_transcription": "5252 731", "words": []},
        }]
        reordered = pipeline.protect_faithful_candidate(
            {"final_transcription": "731 5252", "words": []}, usable
        )
        shortened = pipeline.protect_faithful_candidate(
            {"final_transcription": "5252", "words": []}, usable
        )

        self.assertEqual("5252 731", reordered["final_transcription"])
        self.assertEqual("reordered", reordered["candidate_protection"]["reason"])
        self.assertEqual("5252 731", shortened["final_transcription"])
        self.assertEqual("lower_information", shortened["candidate_protection"]["reason"])

    def test_contextual_recheck_is_adaptive(self):
        easy = [{
            "source": "core4_multi",
            "ocr": {"full_transcription": "5298570", "words": [{"confidence": 0.95}]},
        }]
        difficult = [{
            "source": "core4_multi",
            "ocr": {"full_transcription": "שלום[?]", "words": [{"confidence": 0.4}]},
        }]

        self.assertTrue(pipeline.should_contextual_recheck(
            {"classification": "handwritten_text", "final_transcription": "5298570"}, easy
        ))
        self.assertTrue(pipeline.should_contextual_recheck(
            {"classification": "handwritten_text", "final_transcription": "שלום[?]"}, difficult
        ))
        self.assertTrue(pipeline.should_contextual_recheck(
            {"classification": "handwritten_text", "final_transcription": "134/26"}, easy
        ))

    def test_high_uncertainty_rescue_requires_explicit_marker_and_no_text(self):
        self.assertTrue(pipeline.should_high_uncertainty_rescue(
            {
                "classification": "handwritten_text",
                "final_transcription": "[?]",
            },
            signature_flourish=False,
        ))
        self.assertFalse(pipeline.should_high_uncertainty_rescue(
            {
                "classification": "handwritten_text",
                "final_transcription": "מילה[?]",
            },
            signature_flourish=False,
        ))
        self.assertFalse(pipeline.should_high_uncertainty_rescue(
            {
                "classification": "handwritten_text",
                "final_transcription": "",
            },
            signature_flourish=False,
        ))
        self.assertFalse(pipeline.should_high_uncertainty_rescue(
            {
                "classification": "handwritten_text",
                "final_transcription": "[?]",
            },
            signature_flourish=True,
        ))

    def test_high_uncertainty_rescue_accepts_only_confident_hebrew_text(self):
        current = {
            "classification": "handwritten_text",
            "final_transcription": "[?]",
            "confidence": 0.0,
        }
        accepted = pipeline.accept_high_uncertainty_rescue(current, {
            "classification": "handwritten_text",
            "final_transcription": "הפועלים",
            "confidence": 0.70,
        })

        self.assertEqual("הפועלים", accepted["final_transcription"])
        for rejected in (
            {
                "classification": "handwritten_text",
                "final_transcription": "encircling mark",
                "confidence": 0.95,
            },
            {
                "classification": "signature_non_text",
                "final_transcription": "הפועלים",
                "confidence": 0.95,
            },
            {
                "classification": "handwritten_text",
                "final_transcription": "הפועלים",
                "confidence": 0.69,
            },
        ):
            self.assertIs(
                current,
                pipeline.accept_high_uncertainty_rescue(current, rejected),
            )

    def test_empty_no_handwriting_is_excluded_only_without_blind_text(self):
        empty = {
            "classification": "no_handwriting",
            "final_transcription": "",
        }

        self.assertFalse(pipeline.should_emit_region(
            empty, [{"full_transcription": ""}]
        ))
        self.assertTrue(pipeline.should_emit_region(
            empty, [{"full_transcription": "כתב"}]
        ))
        self.assertTrue(pipeline.should_emit_region(
            {
                "classification": "no_handwriting",
                "final_transcription": "[?] כתב",
            },
            [{"full_transcription": ""}],
        ))
        self.assertTrue(pipeline.should_emit_region(
            {
                "classification": "signature_non_text",
                "final_transcription": "",
            },
            [{"full_transcription": ""}],
        ))

    def test_contextual_images_include_larger_crop_and_full_page(self):
        page = Image.new("RGB", (1000, 800), "white")
        context, highlighted = pipeline.build_contextual_region_images(
            page, [400, 300, 100, 60]
        )

        self.assertGreater(context.width, 100)
        self.assertGreater(context.height, 60)
        self.assertEqual((1000, 800), highlighted.size)

    def test_nearby_printed_context_tracks_vertical_region_position(self):
        markdown = "\n".join(f"line {index}" for index in range(20))

        top = pipeline.nearby_printed_context(markdown, [0, 0, 100, 50], 1000, window=1)
        bottom = pipeline.nearby_printed_context(
            markdown, [0, 900, 100, 50], 1000, window=1
        )

        self.assertIn("line 0", top)
        self.assertNotIn("line 19", top)
        self.assertIn("line 19", bottom)
        self.assertNotIn("line 0", bottom)

    def test_contextual_acceptance_rejects_lower_information(self):
        current = {"final_transcription": "בנק הפועלים", "words": []}
        rejected = pipeline.accept_contextual_recheck(current, {
            "final_transcription": "בנק", "confidence": 0.9,
        })
        accepted = pipeline.accept_contextual_recheck(current, {
            "classification": "handwritten_text",
            "final_transcription": "בנק הפועלים", "confidence": 0.8,
        })

        self.assertIs(current, rejected)
        self.assertEqual("contextual_recheck", accepted["decision_status"])

    def test_contextual_acceptance_uses_configured_confidence_floor(self):
        current = {"final_transcription": "גבעתיים", "words": []}
        below_floor = pipeline.accept_contextual_recheck(current, {
            "classification": "handwritten_text",
            "final_transcription": "בארותים",
            "confidence": pipeline.CONTEXTUAL_RECHECK_MIN_CONFIDENCE - 0.01,
        })
        at_floor = pipeline.accept_contextual_recheck(current, {
            "classification": "handwritten_text",
            "final_transcription": "בארותיים",
            "confidence": pipeline.CONTEXTUAL_RECHECK_MIN_CONFIDENCE,
        })

        self.assertIs(current, below_floor)
        self.assertEqual("בארותיים", at_floor["final_transcription"])
        self.assertEqual("contextual_recheck", at_floor["decision_status"])

    def test_repeated_structured_field_does_not_copy_a_confident_peer(self):
        regions = [
            {"label": "top form customer number"},
            {"label": "bottom form customer number"},
        ]
        finals = [
            {
                "final_transcription": "5252 731",
                "words": [{"text": "5252", "confidence": 0.99}],
            },
            {
                "final_transcription": "6252731",
                "confidence": 0.82,
                "words": [],
            },
        ]

        reconciled = pipeline.reconcile_repeated_structured_fields(finals, regions)

        self.assertEqual("6252731[?]", reconciled[1]["final_transcription"])
        self.assertEqual("cross_region_conflict_review", reconciled[1]["decision_status"])
        self.assertEqual("5252 731", reconciled[1]["alternative_transcriptions"][0])
        self.assertTrue(reconciled[1]["needs_review"])

    def test_repeated_structured_field_ignores_positional_label_words(self):
        finals = [
            {
                "final_transcription": "5252 731",
                "words": [{"text": "5252", "confidence": 0.99}],
            },
            {"final_transcription": "6252731", "words": []},
        ]
        regions = [
            {"label": "handwritten number on next line"},
            {"label": "handwritten number below lower table"},
        ]

        reconciled = pipeline.reconcile_repeated_structured_fields(finals, regions)

        self.assertEqual("6252731[?]", reconciled[1]["final_transcription"])
        self.assertEqual(
            "cross_region_conflict_review", reconciled[1]["decision_status"]
        )

    def test_repeated_structured_field_rejects_weak_or_non_identifier_matches(self):
        weak = pipeline.reconcile_repeated_structured_fields(
            [
                {"final_transcription": "5252731", "confidence": 0.89},
                {"final_transcription": "6252731", "confidence": 0.80},
            ],
            [
                {"label": "top form customer number"},
                {"label": "bottom form customer number"},
            ],
        )
        dates = pipeline.reconcile_repeated_structured_fields(
            [
                {"final_transcription": "12/7/26", "confidence": 0.99},
                {"final_transcription": "13/7/26", "confidence": 0.70},
            ],
            [{"label": "top date"}, {"label": "bottom date"}],
        )

        self.assertEqual("6252731", weak[1]["final_transcription"])
        self.assertEqual("13/7/26", dates[1]["final_transcription"])

    def test_repeated_date_uses_contextual_value_seen_in_another_region(self):
        finals = [
            {
                "final_transcription": "7/2/26",
                "contextual_evidence": {
                    "final_transcription": "7/7/26",
                    "alternative_transcriptions": ["7/2/26"],
                },
            },
            {"final_transcription": "7/7/26"},
        ]

        reconciled = pipeline.reconcile_repeated_dates(
            finals, [{"label": "date"}, {"label": "date"}],
        )

        self.assertEqual("7/7/26", reconciled[0]["final_transcription"])
        self.assertEqual(
            "cross_region_date_consensus", reconciled[0]["decision_status"]
        )
        self.assertEqual(
            2, reconciled[0]["cross_region_date_consensus"]["matched_region"]
        )

    def test_repeated_date_rejects_unsupported_or_multi_date_change(self):
        finals = [
            {
                "final_transcription": "10/7/23",
                "contextual_evidence": {"final_transcription": "10/7/26"},
            },
            {"final_transcription": "11/7/26"},
            {
                "final_transcription": "31/12/26 7/7/26",
                "contextual_evidence": {"final_transcription": "31/12/26 7/7/26"},
            },
        ]

        reconciled = pipeline.reconcile_repeated_dates(
            finals,
            [{"label": "date"}, {"label": "date"}, {"label": "dates"}],
        )

        self.assertEqual("10/7/23", reconciled[0]["final_transcription"])
        self.assertEqual("31/12/26 7/7/26", reconciled[2]["final_transcription"])

    def test_cu_numeric_consensus_repairs_unique_one_edit_token(self):
        reconciled = pipeline.reconcile_with_cu_numeric_tokens(
            [{
                "field_type": "numeric",
                "final_transcription": "767735",
                "notes": [],
            }],
            [{"label": "handwritten reference number"}],
            "reference 7637735",
        )

        self.assertEqual("7637735", reconciled[0]["final_transcription"])
        self.assertEqual("cu_numeric_consensus", reconciled[0]["decision_status"])

    def test_cu_numeric_consensus_rejects_present_untyped_date_or_ambiguous(self):
        present = pipeline.reconcile_with_cu_numeric_tokens(
            [{"field_type": "numeric", "final_transcription": "5252731"}],
            [{"label": "number"}],
            "5252731 6252731",
        )
        untyped = pipeline.reconcile_with_cu_numeric_tokens(
            [{"final_transcription": "767735"}],
            [{"label": "reference number"}],
            "7637735",
        )
        date = pipeline.reconcile_with_cu_numeric_tokens(
            [{"field_type": "numeric", "final_transcription": "10/7/23"}],
            [{"label": "date"}],
            "10/7/26",
        )
        ambiguous = pipeline.reconcile_with_cu_numeric_tokens(
            [{"field_type": "numeric", "final_transcription": "767735"}],
            [{"label": "number"}],
            "7637735 7677350",
        )

        self.assertEqual("5252731", present[0]["final_transcription"])
        self.assertEqual("767735", untyped[0]["final_transcription"])
        self.assertEqual("10/7/23", date[0]["final_transcription"])
        self.assertEqual("767735", ambiguous[0]["final_transcription"])

    def test_uncertain_repeated_number_uses_region_and_cu_agreement(self):
        reconciled = pipeline.reconcile_uncertain_repeated_numbers(
            [
                {"field_type": "unknown", "final_transcription": "5556790"},
                {
                    "field_type": "numeric",
                    "final_transcription": "5556[?]8",
                    "notes": [],
                },
            ],
            [{"label": "colored handwriting"}, {"label": "colored handwriting"}],
            "ID 5556790",
        )

        self.assertEqual("5556790", reconciled[1]["final_transcription"])
        self.assertEqual(
            "repeated_numeric_cu_consensus", reconciled[1]["decision_status"]
        )

    def test_uncertain_repeated_number_requires_uncertainty_and_unique_cu_support(self):
        no_question = pipeline.reconcile_uncertain_repeated_numbers(
            [
                {"final_transcription": "5556790"},
                {"field_type": "numeric", "final_transcription": "555678"},
            ],
            [{}, {}],
            "5556790",
        )
        ambiguous = pipeline.reconcile_uncertain_repeated_numbers(
            [
                {"final_transcription": "5556790"},
                {"final_transcription": "5556789"},
                {"field_type": "numeric", "final_transcription": "5556[?]8"},
            ],
            [{}, {}, {}],
            "5556790 5556789",
        )

        self.assertEqual("555678", no_question[1]["final_transcription"])
        self.assertEqual("5556[?]8", ambiguous[2]["final_transcription"])

    def test_contextual_acceptance_rejects_script_change_and_lost_structured_token(self):
        hebrew = {"final_transcription": "ביוב", "words": []}
        schedule = {"final_transcription": "O+ 1/8/26 31/8/26", "words": []}

        self.assertIs(hebrew, pipeline.accept_contextual_recheck(hebrew, {
            "classification": "mark_or_checkbox",
            "final_transcription": "circle mark", "confidence": 0.95,
        }))
        self.assertIs(schedule, pipeline.accept_contextual_recheck(schedule, {
            "classification": "handwritten_text",
            "final_transcription": "1/8/26 31/8/26", "confidence": 0.95,
        }))

    def test_contextual_hebrew_prompt_withholds_candidate_to_avoid_anchoring(self):
        page = Image.new("RGB", (1000, 800), "white")
        captured = {}

        def fake_call(content, **_kwargs):
            captured["prompt"] = content[0]["text"]
            return {"classification": "handwritten_text", "final_transcription": "בארותיים"}

        with patch.object(pipeline, "_responses_call", side_effect=fake_call):
            pipeline.contextual_region_recheck(
                page,
                {"box": [400, 300, 100, 60]},
                {"final_transcription": "ראשל״צ"},
            )

        self.assertIn("withheld for an independent Hebrew reread", captured["prompt"])
        self.assertNotIn("ראשל״צ", captured["prompt"])

    def test_spatial_numeric_cells_are_assembled_left_to_right(self):
        recheck = pipeline.assemble_spatial_numeric_cells({
            "field_type": "numeric",
            "final_transcription": "05686399 142 012123",
            "numeric_cells": [
                {"value": "142", "x": 0.50, "confidence": 0.9},
                {"value": "05686399", "x": 0.82, "confidence": 0.8},
                {"value": "012123", "x": 0.16, "confidence": 0.95},
            ],
        })

        self.assertEqual("012123 142 05686399", recheck["final_transcription"])
        self.assertTrue(recheck["spatial_numeric_assembly"])

    def test_spatial_numeric_cells_group_adjacent_single_digits(self):
        recheck = pipeline.assemble_spatial_numeric_cells({
            "field_type": "numeric",
            "final_transcription": "056 142 012",
            "numeric_cells": [
                {"value": "1", "x": 0.15},
                {"value": "2", "x": 0.20},
                {"value": "0", "x": 0.10},
                {"value": "142", "x": 0.50},
                {"value": "0", "x": 0.75},
                {"value": "5", "x": 0.80},
                {"value": "6", "x": 0.85},
            ],
        })

        self.assertEqual("012 142 056", recheck["final_transcription"])
        self.assertTrue(recheck["spatial_numeric_assembly"])

    def test_spatial_numeric_cells_ignore_mixed_script(self):
        original = {
            "field_type": "numeric",
            "final_transcription": "012123 142",
            "numeric_cells": [
                {"value": "012123", "x": 0.2},
                {"value": "בנק", "x": 0.8},
            ],
        }

        self.assertIs(original, pipeline.assemble_spatial_numeric_cells(original))

    def test_signature_semantics_clear_transcription(self):
        final = pipeline.finalize_region_semantics(
            {
                "classification": "signature_non_text",
                "final_transcription": "[?]",
                "words": [{"text": "[?]"}],
            },
            [],
            signature_flourish=True,
        )

        self.assertEqual("", final["final_transcription"])
        self.assertEqual([], final["words"])
        self.assertEqual("signature_non_text", final["decision_status"])

    def test_signature_geometry_rejects_separate_letter_components(self):
        signature = Image.new("RGB", (200, 80), "white")
        draw = ImageDraw.Draw(signature)
        draw.line((10, 40, 190, 40), fill=(0, 70, 220), width=5)
        draw.ellipse((70, 10, 120, 65), outline=(0, 70, 220), width=5)
        word = Image.new("RGB", (200, 80), "white")
        draw = ImageDraw.Draw(word)
        draw.ellipse((20, 15, 60, 65), outline=(0, 70, 220), width=5)
        draw.ellipse((85, 15, 125, 65), outline=(0, 70, 220), width=5)
        draw.ellipse((150, 15, 190, 65), outline=(0, 70, 220), width=5)

        self.assertTrue(pipeline.is_signature_flourish(signature))
        self.assertFalse(pipeline.is_signature_flourish(word))

    def test_black_signature_geometry_accepts_dominant_flourish(self):
        signature = Image.new("RGB", (400, 160), "white")
        draw = ImageDraw.Draw(signature)
        draw.line((10, 130, 360, 35), fill="black", width=9)
        draw.arc((110, 20, 300, 145), 20, 320, fill="black", width=8)
        word = Image.new("RGB", (400, 160), "white")
        draw = ImageDraw.Draw(word)
        for left in (30, 100, 170, 240, 310):
            draw.line((left, 130, left + 35, 35), fill="black", width=8)

        self.assertTrue(pipeline.is_black_signature_flourish(signature))
        self.assertFalse(pipeline.is_black_signature_flourish(word))

    def test_empty_signature_notes_promote_geometry_backed_signature(self):
        final = pipeline.finalize_region_semantics(
            {
                "classification": "handwritten_text",
                "final_transcription": "",
                "notes": ["נראה קו חתימה/שרבוט לא טקסטואלי"],
            },
            [],
            signature_flourish=True,
        )

        self.assertEqual("signature_non_text", final["classification"])
        self.assertEqual("signature_non_text", final["decision_status"])

    def test_signature_notes_without_geometry_remain_text(self):
        final = {
            "classification": "handwritten_text",
            "final_transcription": "",
            "notes": ["signature-like but uncertain"],
        }

        resolved = pipeline.finalize_region_semantics(
            final, [], signature_flourish=False,
        )

        self.assertEqual("handwritten_text", resolved["classification"])

    def test_mark_semantics_render_x_and_number(self):
        final = pipeline.finalize_region_semantics(
            {
                "classification": "mark_or_checkbox",
                "final_transcription": "1 X",
                "marks": ["X"],
                "digits": ["1"],
            },
            [],
        )

        self.assertEqual("סימון x ו המספר 1", final["final_transcription"])
        self.assertEqual("structured_mark", final["decision_status"])

    def test_empty_visual_result_preserves_multi_image_text_on_non_signature(self):
        usable = [{
            "source": "core4_multi",
            "ocr": {"full_transcription": "ביוב", "words": [{"text": "ביוב"}]},
        }]

        final = pipeline.finalize_region_semantics(
            {
                "classification": "mark_or_checkbox",
                "final_transcription": "",
                "marks": [],
                "words": [],
            },
            usable,
            signature_flourish=False,
        )

        self.assertEqual("ביוב", final["final_transcription"])
        self.assertEqual("handwritten_text", final["classification"])
        self.assertEqual("nonempty_faithful_fallback", final["decision_status"])

    def test_full_mode_ocr_uses_only_four_core_views(self):
        image = Image.new("RGB", (80, 40), "white")
        with tempfile.TemporaryDirectory() as temp_dir:
            tasks, all_images = pipeline.prepare_region_images(
                image,
                {"box": [0, 0, 80, 40]},
                temp_dir,
                "full",
                chunk_width=900,
                overlap=150,
            )

        self.assertEqual(list(pipeline.CORE_OCR_VIEWS), [name for name, _ in tasks])
        self.assertEqual(8, len(all_images))
        self.assertIn("03_adaptive_threshold.png", dict(all_images))
        self.assertIn("07_pen_stroke_isolated.png", dict(all_images))

    @patch.object(pipeline, "_responses_call")
    def test_multi_image_ocr_sends_four_labeled_views_in_one_call(self, responses_call):
        responses_call.return_value = {"full_transcription": "5252731", "words": []}
        images = [
            (name, Image.new("RGB", (20, 10), "white"))
            for name in pipeline.CORE_OCR_VIEWS
        ]

        result = pipeline.ocr_image_set(images, "read this")

        content = responses_call.call_args.args[0]
        self.assertEqual(4, sum(item["type"] == "input_image" for item in content))
        self.assertEqual(1, responses_call.call_args.kwargs["endpoint_limit"])
        self.assertEqual(
            list(pipeline.CORE_OCR_VIEWS),
            result["evidence_views"],
        )
        labels = [item["text"] for item in content if item["type"] == "input_text"]
        self.assertTrue(all(any(name in label for label in labels) for name in pipeline.CORE_OCR_VIEWS))

    @patch.object(pipeline, "_responses_call")
    def test_isolated_multiview_plan_aligns_two_blind_reads(self, responses_call):
        responses_call.side_effect = [
            {"transcription": "סניף"},
            {"transcription": "סניף"},
        ]
        views = [
            ("original_with_form_line", Image.new("RGB", (20, 10), "white")),
            ("blue_only", Image.new("RGB", (20, 10), "white")),
            ("blue_only_enlarged", Image.new("RGB", (60, 30), "white")),
        ]

        final, candidates, details = pipeline.run_isolated_multiview_plan({
            "lines": [{
                "line": 1,
                "mode": "word_groups",
                "units": [{"unit_rtl": 1, "views": views, "metadata": {}}],
            }],
            "draws": 2,
        })

        self.assertEqual("סניף", final["final_transcription"])
        self.assertFalse(final["needs_review"])
        self.assertEqual("isolated_multiview_consensus", final["decision_status"])
        self.assertEqual(2, len(candidates))
        self.assertEqual(2, details["successful_calls"])
        self.assertEqual(2, details["concurrency"])
        self.assertTrue(all(
            sum(item["type"] == "input_image" for item in call.args[0]) == 3
            for call in responses_call.call_args_list
        ))

    def test_isolated_multiview_pipeline_writes_standard_result_artifacts(self):
        input_path = self.output_dir / "lined.png"
        Image.new("RGB", (120, 80), "white").save(input_path)
        output_dir = self.output_dir / "isolated-output"
        final = {
            "classification": "handwritten_text",
            "final_transcription": "סניף פארק",
            "alternative_transcriptions": [],
            "words": [],
            "notes": ["experimental isolated multi-view consensus"],
            "needs_review": False,
            "decision_status": "isolated_multiview_consensus",
        }
        candidates = [
            {"source": "isolated_multiview_a", "ocr": {"full_transcription": "סניף פארק"}},
            {"source": "isolated_multiview_b", "ocr": {"full_transcription": "סניף פארק"}},
        ]
        plan = {"lines": [], "draws": 2}

        with patch.object(pipeline, "OCR_STRATEGY", pipeline.ISOLATED_MULTIVIEW_STRATEGY), \
             patch.object(pipeline, "detect_handwriting_regions", return_value=[{
                 "label": "lined handwriting", "box": [0, 0, 120, 80],
             }]), \
             patch.object(pipeline, "prepare_isolated_multiview_plan", return_value=plan), \
             patch.object(
                 pipeline,
                 "run_isolated_multiview_plan",
                 return_value=(final, candidates, {"final_transcription": "סניף פארק"}),
             ), \
             patch.object(pipeline, "visual_reconcile_ocr_results") as visual_reconcile:
            return_code = pipeline.run_pipeline(
                str(input_path),
                page=1,
                dpi=400,
                crops=None,
                output_dir=str(output_dir),
                chunk_width=900,
                overlap=150,
                ocr_mode="full",
                auto_detect=True,
                merge_md=False,
            )

        result = json.loads((output_dir / "handwriting_result.json").read_text())
        region = result["regions"][0]
        self.assertEqual(0, return_code)
        self.assertEqual("isolated-multiview", result["ocr_strategy"])
        self.assertEqual("סניף פארק", region["final_transcription"])
        self.assertEqual("isolated_multiview_consensus", region["decision_status"])
        self.assertTrue((output_dir / region["folder"] / "result.json").exists())
        visual_reconcile.assert_not_called()

    def test_pipeline_uses_cu_numeric_evidence_without_model_context(self):
        input_path = self.output_dir / "numeric-region.png"
        Image.new("RGB", (120, 80), "white").save(input_path)
        output_dir = self.output_dir / "cu-numeric-output"
        gpt_read = {
            "classification": "handwritten_text",
            "field_type": "numeric",
            "full_transcription": "058329",
            "words": [],
        }
        final = {
            "classification": "handwritten_text",
            "field_type": "numeric",
            "final_transcription": "058329",
            "words": [],
            "notes": [],
        }

        with patch.object(pipeline, "OCR_STRATEGY", "standard"), \
             patch.object(pipeline, "NEARBY_CU_CONTEXT", False), \
             patch.object(pipeline, "detect_handwriting_regions", return_value=[{
                 "label": "handwritten phone number", "box": [0, 0, 120, 80],
             }]), \
             patch.object(pipeline, "ocr_image_set", return_value=gpt_read), \
             patch.object(pipeline, "should_contextual_recheck", return_value=False), \
             patch.object(
                 pipeline, "visual_reconcile_ocr_results", return_value=final,
             ) as visual_reconcile, \
             patch.object(pipeline, "cu_ocr_markdown", return_value="phone 058328"), \
             patch.object(
                 pipeline, "build_merged_html", return_value=output_dir / "merged.html"
             ), \
             patch.object(pipeline, "build_merged_jpg", return_value=None), \
             patch.object(
                 pipeline, "build_merged_markdown", return_value=output_dir / "merged.md"
             ):
            return_code = pipeline.run_pipeline(
                str(input_path),
                page=1,
                dpi=400,
                crops=None,
                output_dir=str(output_dir),
                chunk_width=900,
                overlap=150,
                ocr_mode="full",
                auto_detect=True,
                merge_md=True,
            )

        result = json.loads((output_dir / "handwriting_result.json").read_text())
        self.assertEqual(0, return_code)
        self.assertEqual("", visual_reconcile.call_args.args[2])
        self.assertEqual("058328", result["regions"][0]["final_transcription"])
        self.assertEqual(
            "cu_numeric_consensus", result["regions"][0]["decision_status"]
        )

    def test_standard_pipeline_uses_only_vision_evidence(self):
        input_path = self.output_dir / "vision-only.png"
        Image.new("RGB", (120, 80), "white").save(input_path)
        output_dir = self.output_dir / "vision-only-output"
        reading = {
            "classification": "handwritten_text",
            "full_transcription": "שלום", "words": [],
        }
        final = {
            "classification": "handwritten_text",
            "final_transcription": "שלום", "words": [], "notes": [],
        }
        with patch.object(pipeline, "VOTE_ENABLED", False), \
             patch.object(pipeline, "OCR_STRATEGY", "standard"), \
             patch.object(pipeline, "requests") as requests, \
             patch.object(pipeline, "ocr_image_set", return_value=reading), \
             patch.object(pipeline, "should_contextual_recheck", return_value=False), \
             patch.object(
                 pipeline, "visual_reconcile_ocr_results", return_value=final,
             ) as reconcile:
            code = pipeline.run_pipeline(
                str(input_path), page=1, dpi=400, crops=[(0, 0, 120, 80)],
                output_dir=str(output_dir), chunk_width=900, overlap=150,
                ocr_mode="full", merge_md=False,
            )

        self.assertEqual(0, code)
        requests.get.assert_not_called()
        requests.post.assert_not_called()
        self.assertEqual(["core4_multi"], [
            result["source"] for result in reconcile.call_args.args[1]
        ])
        result = json.loads((output_dir / "handwriting_result.json").read_text())
        self.assertEqual("שלום", result["regions"][0]["final_transcription"])
        progress = json.loads((output_dir / "ocr_progress.json").read_text())
        for region in (result["regions"][0], progress["regions"][0]):
            self.assertNotIn("ivrit_ocr", region)
            self.assertNotIn("kraken_ocr", region)

    def test_removed_ocr_strategies_are_rejected(self):
        client = web_app.app.test_client()
        for strategy in ("standard-kraken-small", "standard-kraken-medium"):
            with self.subTest(strategy=strategy):
                response = client.post("/api/jobs", data={"ocr_strategy": strategy})
                self.assertEqual(400, response.status_code)
                self.assertEqual("Unsupported OCR strategy.", response.get_json()["error"])

    def test_saved_runs_with_retired_strategies_remain_readable(self):
        job = web_app._jobs[self.job_id]
        job.update(status="complete", historical=True, selected_pages=[1])
        client = web_app.app.test_client()
        for strategy in ("standard-kraken-small", "standard-kraken-medium"):
            with self.subTest(strategy=strategy):
                job["ocr_strategy"] = strategy
                with patch.object(web_app, "RESULTS_DIR", self.output_dir):
                    listing = client.get("/api/runs")
                snapshot = client.get(f"/api/jobs/{self.job_id}")
                self.assertEqual(200, listing.status_code)
                self.assertEqual(200, snapshot.status_code)
                for result in (listing.get_json()[0], snapshot.get_json()):
                    self.assertEqual(strategy, result["ocr_strategy"])
                    self.assertEqual(
                        f"Retired OCR strategy: {strategy}", result["ocr_strategy_label"],
                    )

    def test_responses_url_accepts_base_or_full_url(self):
        self.assertEqual(
            "https://example.test/openai/v1/responses",
            pipeline._responses_url("https://example.test/openai/v1/"),
        )
        self.assertEqual(
            "https://example.test/openai/v1/responses",
            pipeline._responses_url("https://example.test/openai/v1/responses"),
        )

    @patch.object(pipeline, "_bearer", return_value="token")
    @patch.object(pipeline.requests.Session, "post")
    def test_transient_error_immediately_fails_over_to_second_endpoint(self, post, _bearer):
        success = Mock(status_code=200)
        success.json.return_value = {
            "output_text": '{"ok": true}',
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        post.side_effect = [pipeline.requests.exceptions.ReadTimeout("slow"), success]
        targets = [
            {"name": "one", "url": "https://one/responses", "deployment": "gpt-5.4"},
            {"name": "two", "url": "https://two/responses", "deployment": "gpt-5.4"},
        ]

        with patch.object(pipeline, "GPT_TARGETS", targets), \
             patch.object(pipeline, "_target_cursor", 0):
            text = pipeline._responses_text(
                [{"type": "input_text", "text": "test"}], retries=0
            )

        self.assertEqual('{"ok": true}', text)
        self.assertEqual(2, post.call_count)
        self.assertEqual("https://one/responses", post.call_args_list[0].args[0])
        self.assertEqual("https://two/responses", post.call_args_list[1].args[0])

    @patch.object(pipeline, "_responses_call")
    def test_focused_recheck_uses_only_faithful_images(self, responses_call):
        responses_call.return_value = {
            "final_transcription": "גולדברג",
            "visible_letter_sequence": list("גולדברג"),
            "confidence": 0.82,
            "words": [],
        }
        images = [
            (source, Image.new("RGB", (10, 10)))
            for source in [
                "01_original_crop.png",
                "02_upscaled_contrast.png",
                "03_adaptive_threshold.png",
                "04_denoised.png",
                "05_thickened_strokes.png",
            ]
        ]

        result = pipeline.focused_faithful_recheck(images, "אלון זמיר")

        content = responses_call.call_args.args[0]
        labels = [item["text"] for item in content if item["type"] == "input_text"]
        self.assertEqual("גולדברג", result["final_transcription"])
        self.assertTrue(any("01_original_crop.png" in label for label in labels))
        self.assertFalse(any("05_thickened_strokes.png" in label for label in labels))
        self.assertEqual(0, responses_call.call_args.kwargs["retries"])
        self.assertEqual(12000, responses_call.call_args.kwargs["max_tokens"])

    def test_focused_recheck_cannot_degrade_supported_faithful_candidate(self):
        guarded = {
            "final_transcription": "על דבר",
            "decision_status": "faithful_candidate_fallback",
            "agreement_gate": {"faithful_supporting_sources": []},
        }
        usable = [{
            "source": "01_original_crop.png",
            "ocr": {"full_transcription": "על דבר"},
        }]

        accepted = pipeline.accept_focused_faithful_recheck(
            guarded, {"final_transcription": "[?]ל [?]", "words": []}, usable
        )

        self.assertEqual(guarded, accepted)

    def test_focused_recheck_accepts_clear_faithful_candidate(self):
        guarded = {
            "final_transcription": "[?]ל דבר",
            "decision_status": "faithful_candidate_fallback",
            "agreement_gate": {"faithful_supporting_sources": []},
        }
        usable = [{
            "source": "01_original_crop.png",
            "ocr": {"full_transcription": "על דבר"},
        }]

        accepted = pipeline.accept_focused_faithful_recheck(
            guarded, {"final_transcription": "על דבר", "words": [], "notes": []}, usable
        )

        self.assertEqual("על דבר", accepted["final_transcription"])
        self.assertEqual("focused_faithful_recheck", accepted["decision_status"])


if __name__ == "__main__":
    unittest.main()