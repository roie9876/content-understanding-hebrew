import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

import app as web_app
import compare_golden
import handwriting_pipeline as pipeline


class ConcurrentJobTests(unittest.TestCase):
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

    def test_luna_profile_sets_page_subprocess_environment(self):
        environment = web_app._model_environment("gpt-5.6-luna")

        self.assertEqual(
            "https://admin-9107-resource.services.ai.azure.com/openai/v1/",
            environment["GPT_ENDPOINT"],
        )
        self.assertEqual("gpt-5.6-luna", environment["GPT_DEPLOYMENT"])

    def test_gpt_54_pro_profile_sets_deployment_reasoning_and_timeout(self):
        environment = web_app._model_environment("gpt-5.4-pro")

        self.assertEqual(
            "https://medical-summary.services.ai.azure.com/openai/v1/",
            environment["GPT_ENDPOINT"],
        )
        self.assertEqual("gpt-5.4-pro", environment["GPT_DEPLOYMENT"])
        self.assertEqual("medium", environment["GPT_REASONING"])
        self.assertEqual("medium", environment["OCR_REASONING"])
        self.assertEqual("180", environment["GPT_READ_TIMEOUT"])

    def test_sol_profile_sets_page_subprocess_environment(self):
        environment = web_app._model_environment("gpt-5.6-sol")

        self.assertEqual(
            "https://admin-9107-resource.services.ai.azure.com/openai/v1/",
            environment["GPT_ENDPOINT"],
        )
        self.assertEqual("gpt-5.6-sol", environment["GPT_DEPLOYMENT"])
        self.assertEqual("low", environment["GPT_REASONING"])
        self.assertEqual("low", environment["OCR_REASONING"])

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
                            "model_profile": "gpt-5.4",
                            "pages": "2,4-5",
                        },
                    )
            payload = response.get_json()

            self.assertEqual(202, response.status_code)
            self.assertEqual([2, 4, 5], payload["selected_pages"])
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

        response = web_app.app.test_client().get(f"/api/jobs/{self.job_id}/pages/1/diagnostics")
        payload = response.get_json()

        self.assertEqual(200, response.status_code)
        self.assertEqual(1, payload["detected"]["count"])
        self.assertEqual("שלום", payload["progress"]["regions"][0]["candidates"][0]["text"])
        self.assertEqual("שלום", payload["final_regions"][0]["final_transcription"])

    def test_full_mode_uses_all_seven_enhancement_candidates(self):
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

        self.assertEqual(7, len(tasks))
        self.assertEqual(8, len(images))

    def test_handwriting_context_expands_truncated_field(self):
        page = Image.new("RGB", (3400, 4400), "white")

        expanded = pipeline.add_handwriting_context(page, [1579, 559, 180, 308])

        self.assertGreaterEqual(expanded[2], 700)
        self.assertGreater(expanded[1] + expanded[3], 1000)

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

    @patch("handwriting_pipeline._bearer", return_value="token")
    @patch("handwriting_pipeline.requests.post")
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
        self.assertEqual([415, 669, 911, 615], deduplicated[0]["box"])
        self.assertEqual([1285, 660, 1129, 615], deduplicated[1]["box"])

    def test_detection_dedup_merges_same_sized_duplicate_fields(self):
        regions = [
            {"label": "handwritten number in upper left field", "box": [431, 658, 895, 642]},
            {"label": "handwritten number in second upper left field", "box": [433, 745, 893, 657]},
        ]

        deduplicated = pipeline.deduplicate_detection_regions(regions, 3400, 4400)

        self.assertEqual(1, len(deduplicated))
        self.assertEqual([431, 658, 895, 744], deduplicated[0]["box"])

    def test_midpage_handwriting_line_gets_horizontal_context_only(self):
        regions = [
            {"label": "handwritten name entry in mid upper line", "box": [1411, 1352, 929, 336]},
            {"label": "small form field", "box": [1411, 900, 600, 400]},
        ]

        widened = pipeline.widen_midpage_handwriting_lines(regions, 3400, 4400)

        self.assertEqual([408, 1352, 2652, 336], widened[0]["box"])
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
        self.assertLess(expanded[2], 2700)
        self.assertEqual(1569, expanded[1])
        self.assertEqual(1964, expanded[1] + expanded[3])

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