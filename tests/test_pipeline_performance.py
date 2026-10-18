import copy
import hashlib
import json
import tempfile
import threading
import unittest
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image, ImageDraw

import handwriting_pipeline as pipeline


TEXT = "\u05e9\u05dc\u05d5\u05dd"


class HttpSessionTests(unittest.TestCase):
    def test_connections_are_reused_per_thread_and_closed(self):
        with patch.object(pipeline, "_http_local", threading.local()), \
             patch.object(pipeline, "_http_sessions", []), \
             patch.object(pipeline.requests, "Session", side_effect=lambda: Mock()) as factory:
            main_session = pipeline._http_session()
            self.assertIs(main_session, pipeline._http_session())
            with ThreadPoolExecutor(max_workers=1) as executor:
                worker_session = executor.submit(pipeline._http_session).result()
                self.assertIs(
                    worker_session, executor.submit(pipeline._http_session).result(),
                )
            self.assertIsNot(main_session, worker_session)
            self.assertEqual(2, factory.call_count)

            pipeline._close_http_sessions()

            main_session.close.assert_called_once_with()
            worker_session.close.assert_called_once_with()
            self.assertEqual([], pipeline._http_sessions)


class PipelineSchedulingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.input = self.root / "source.png"
        image = Image.new("RGB", (200, 100), "white")
        ImageDraw.Draw(image).rectangle((0, 0, 99, 99), fill=(230, 230, 230))
        image.save(self.input)
        self.run_count = 0

    def tearDown(self):
        self.temporary.cleanup()

    def _run(self, ocr, reconcile, **settings):
        self.run_count += 1
        output = self.root / f"run-{self.run_count}"
        configuration = {
            "MAX_WORKERS": 2,
            "OCR_STRATEGY": "standard",
            "MULTI_IMAGE_OCR": True,
            "VOTE_ENABLED": False,
            "NEARBY_CU_CONTEXT": False,
        }
        configuration.update(settings)
        with ExitStack() as stack:
            stack.enter_context(patch.multiple(pipeline, **configuration))
            stack.enter_context(patch.object(pipeline, "ocr_image_set", side_effect=ocr))
            stack.enter_context(patch.object(
                pipeline, "visual_reconcile_ocr_results", side_effect=reconcile,
            ))
            stack.enter_context(patch.object(pipeline, "should_contextual_recheck", return_value=False))
            stack.enter_context(patch.object(pipeline.time, "sleep"))
            stack.enter_context(patch.object(pipeline, "build_merged_jpg", return_value=None))
            code = pipeline.run_pipeline(
                str(self.input), page=1, dpi=400,
                crops=[(0, 0, 100, 100), (100, 0, 100, 100)],
                output_dir=str(output), chunk_width=900, overlap=150, ocr_mode="full",
            )
        self.assertEqual(0, code)
        return json.loads((output / "handwriting_result.json").read_text())

    @staticmethod
    def _reading(text=TEXT):
        return {
            "full_transcription": text, "classification": "handwritten_text",
            "words": [{"text": text, "confidence": 1.0}],
        }

    @staticmethod
    def _final(results):
        reading = results[0]["ocr"]
        return {
            "final_transcription": reading["full_transcription"],
            "classification": "handwritten_text", "words": [], "notes": [],
        }

    def test_ready_region_is_verified_before_slow_ocr_finishes_with_same_budget(self):
        verified_fast = threading.Event()
        slow_released_by_verification = []
        lock = threading.Lock()
        active = 0
        peak = 0
        calls = Counter()

        def enter(kind):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                calls[kind] += 1

        def leave():
            nonlocal active
            with lock:
                active -= 1

        def ocr(images, _prompt, _seed):
            enter("ocr")
            try:
                if dict(images)["01_original_crop.png"].getpixel((0, 0))[0] > 250:
                    slow_released_by_verification.append(verified_fast.wait(timeout=2))
                    return self._reading("12345")
                return self._reading()
            finally:
                leave()

        def reconcile(_images, results, _context):
            enter("verification")
            try:
                if results[0]["ocr"]["full_transcription"] == TEXT:
                    verified_fast.set()
                return self._final(results)
            finally:
                leave()

        result = self._run(ocr, reconcile)

        self.assertEqual([True], slow_released_by_verification)
        self.assertLessEqual(peak, 2)
        self.assertEqual(Counter(ocr=2, verification=2), calls)
        self.assertEqual([TEXT, "12345"], [
            region["final_transcription"] for region in result["regions"]
        ])

    def test_context_and_voting_modes_keep_their_full_evidence_barrier(self):
        for settings in (
            {"NEARBY_CU_CONTEXT": True},
            {"VOTE_ENABLED": True, "VOTE_SEEDS": 2},
        ):
            with self.subTest(settings=settings):
                completed = []
                observed = []
                expected = 4 if settings.get("VOTE_ENABLED") else 2

                def ocr(_images, _prompt, _seed):
                    completed.append(True)
                    return self._reading()

                def reconcile(_images, results, _context):
                    observed.append(len(completed))
                    return self._final(results)

                with patch.object(pipeline, "aggregate_votes", return_value={
                    "final_transcription": "[?]", "words": [],
                }):
                    self._run(ocr, reconcile, **settings)

                self.assertEqual(expected, len(completed))
                self.assertEqual([expected, expected], observed)

    def test_empty_region_still_gets_faithful_rescue_and_is_verified_once(self):
        calls = []
        verified = []

        def ocr(images, _prompt, seed):
            bright = dict(images)["01_original_crop.png"].getpixel((0, 0))[0] > 250
            calls.append((bright, seed, [name for name, _image in images]))
            if bright and seed is None:
                raise RuntimeError("transient OCR failure")
            return self._reading("12345" if bright else TEXT)

        def reconcile(_images, results, _context):
            verified.append(results[0]["source"])
            return self._final(results)

        result = self._run(ocr, reconcile)

        self.assertEqual(3, len(calls))
        rescue = next(call for call in calls if call[1] == 900)
        self.assertEqual(sorted(pipeline.RETRYABLE_OCR_VIEWS), sorted(rescue[2]))
        self.assertCountEqual(["core4_multi", "core2_faithful_retry"], verified)
        self.assertEqual(2, len(result["regions"]))

    def test_failed_faithful_rescue_still_gets_original_image_rescue(self):
        calls = []
        verified = []

        def ocr(images, _prompt, seed):
            bright = dict(images)["01_original_crop.png"].getpixel((0, 0))[0] > 250
            calls.append((bright, seed, [name for name, _image in images]))
            if bright and seed is None:
                return {"full_transcription": "", "words": []}
            if bright and seed == 900:
                raise RuntimeError("faithful rescue failed")
            return self._reading("12345" if bright else TEXT)

        def reconcile(_images, results, _context):
            verified.append([result["source"] for result in results])
            return self._final([result for result in results if result["ocr"].get(
                "full_transcription"
            )])

        with patch.object(pipeline, "OCR_FINAL_ORIGINAL_RESCUE", True), \
             patch.object(pipeline, "ocr_image", side_effect=lambda image, prompt, seed: ocr(
                 [("01_original_crop.png", image)], prompt, seed,
             )):
            result = self._run(ocr, reconcile)

        original = next(call for call in calls if call[1] == 901)
        self.assertEqual(["01_original_crop.png"], original[2])
        self.assertEqual(2, len(verified))
        self.assertTrue(any(
            "01_original_crop.png#final_rescue" in sources for sources in verified
        ))
        self.assertEqual(2, len(result["regions"]))

    def test_overlapping_and_barrier_schedules_have_identical_requests_and_results(self):
        captured = []
        response = {
            **self._reading(), "final_transcription": TEXT,
            "confidence": 1.0, "notes": [],
        }

        def respond(content, **kwargs):
            serialized = json.dumps(
                {"content": content, "kwargs": kwargs}, sort_keys=True,
            ).encode()
            captured.append(hashlib.sha256(serialized).hexdigest())
            return copy.deepcopy(response)

        outputs = []
        requests_by_schedule = []
        for context_barrier in (True, False):
            self.run_count += 1
            output = self.root / f"equivalent-{self.run_count}"
            captured.clear()
            with patch.multiple(
                pipeline, MAX_WORKERS=2, OCR_STRATEGY="standard",
                MULTI_IMAGE_OCR=True, VOTE_ENABLED=False,
                NEARBY_CU_CONTEXT=context_barrier,
            ), patch.object(pipeline, "_responses_call", side_effect=respond), \
                 patch.object(pipeline, "build_merged_jpg", return_value=None):
                code = pipeline.run_pipeline(
                    str(self.input), page=1, dpi=400,
                    crops=[(0, 0, 100, 100), (100, 0, 100, 100)],
                    output_dir=str(output), chunk_width=900, overlap=150, ocr_mode="full",
                )
            self.assertEqual(0, code)
            result = json.loads((output / "handwriting_result.json").read_text())
            result.pop("elapsed")
            result.pop("elapsed_seconds")
            outputs.append(result)
            requests_by_schedule.append(Counter(captured))

        self.assertGreaterEqual(sum(requests_by_schedule[0].values()), 6)
        self.assertEqual(requests_by_schedule[0], requests_by_schedule[1])
        self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
